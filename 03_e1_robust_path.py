# =============================================================================
# Program      : 03_e1_robust_path.py
# Version      : 3.0
# Description  : Experiment 1 — Robust Path (Path A) training and evaluation.
#
#                Trains ONLY the robust-ID branch: a small HyperNetwork
#                (key bits -> FiLM gamma/beta) that modulates a slice of
#                EnCodec's PRE-quantization latent (the encoder's raw
#                continuous output), which the frozen quantizer then
#                genuinely quantizes (always producing on-manifold output,
#                via its own straight-through-estimator training path), fed
#                to the frozen decoder. A small extractor CNN is trained
#                alongside it to recover the key bits from the watermarked
#                waveform. EnCodec itself (encoder, quantizer, decoder)
#                stays completely frozen throughout — only the HyperNetwork
#                and Extractor have trainable parameters.
#
#                This is Path A ONLY (no fragile path yet — that's E2).
#                Evaluated against the benign half of the threat model
#                (§4.4 of the blueprint): compression, resampling, EQ,
#                gain, and noise, at a range of severities.
#
#                PIVOTED IN v2.0: earlier versions (v1.0-v1.9) modulated
#                the POST-quantization embedding — decode()'d directly from
#                fixed integer codes, never having gone through an actual
#                quantization step. That fed the decoder an off-manifold
#                input it was never trained to expect. Four different
#                training configurations (tight FiLM bound, 5x wider bound,
#                a batch-size-validated fix, and a 10x-reweighted loss) all
#                produced statistically identical chance-level results —
#                evidence pointing at the insertion point itself, not a
#                hyperparameter. See the v2.0 changelog entry below for the
#                full reasoning and the self-test that validates it.
#
# PRE-FLIGHT SELF-TEST: introspects model.quantizer's training-mode forward
# API (method name/signature aren't hardcoded — the exact call is
# discovered and validated at runtime, since the raw `encodec` package's
# exact API surface hasn't been confirmed against a live installation),
# validates reconstruction, and crucially probes that gradient actually
# flows from the decoder's output back through quantization to the
# pre-quantization embedding. If any of that fails, the script stops with
# a diagnostic instead of silently training on a broken pipeline — check
# that output on the first run before trusting anything after it.
#
#                KNOWN SIMPLIFICATIONS in this first version (flagged, not
#                hidden): (1) only BER is computed, not detection AUC — AUC
#                needs unwatermarked negative examples, deferred to keep
#                scope manageable; (2) the "compression" augmentation used
#                DURING training is EnCodec re-encoding itself at a lower
#                bandwidth (fast, stays on GPU, no gradient to the embedder
#                through this branch since it's non-differentiable) — REAL
#                MP3 compression (via pydub/ffmpeg) is used only in the
#                post-training evaluation battery, where the I/O cost is
#                affordable; (3) insertion point is now pre-quantization
#                (was post-quantization through v1.9; see v2.0 changelog —
#                pre-/post-quantization/residual-VQ are the three options
#                E6 is meant to ablate, so this is one of the intended
#                experiment axes, not an arbitrary implementation detail).
#
# INPUT:
#                  PROJECT_DIR (/content/drive/MyDrive/paper/HyperFrag/):
#                    dataset_e0.h5    — from 01_e0_dataextract.py
#                    e1_robust_checkpoint.pth  — if present, resumes training
#
# STEPS:
#                  Step 1  Download dataset_e0.h5 from PROJECT_DIR, load all 1050
#                          clips into memory, seeded 90/10 train/val split
#                  Step 2  Load frozen pretrained EnCodec, run the
#                          pre-flight self-test (see above), determine the
#                          real latent channel count D and set
#                          D_ROBUST = D // 4
#                  Step 3  Build HyperNetRobust + ExtractorRobust (small,
#                          trainable); try to resume from
#                          e1_robust_checkpoint.pth in PROJECT_DIR
#                  Step 4  Train for N_EPOCHS: embed with random key bits,
#                          apply a randomly chosen differentiable benign
#                          augmentation (noise/gain/resample/EQ) AND the
#                          non-differentiable compression proxy, extract
#                          bits from clean + both augmented versions, BCE
#                          + multi-res STFT reconstruction loss
#                  Step 5  Checkpoint every CHECKPOINT_EVERY_EPOCHS epochs
#                          to PROJECT_DIR (model + optimizer + epoch, resumable)
#                  Step 6  Post-training: evaluate BER on the held-out val
#                          split against REAL benign transforms (real MP3
#                          at several bitrates, real resampling, gain,
#                          noise, EQ), save results + one diagnostic figure
#
# OUTPUT FILES : (stored at /content/drive/MyDrive/paper/HyperFrag/)
#                  e1_robust_checkpoint.pth  (overwritten each checkpoint)
#                      {hypernet_state, extractor_state, optimizer_state,
#                       epoch, d_robust, key_bits}
#                  exp1_robust_results.json
#                      {training: {final_loss, final_bit_acc},
#                       ber_by_transform: {transform_name: {severity: ber}}}
#                  fig_03_01_ber_by_transform.png
#                      BER vs. severity, one line per transform type
#
# GPU Required : YES
# Dependencies : torch, torchaudio, encodec, pydub (needs ffmpeg, Colab has
#                it), h5py, matplotlib, tqdm, numpy
#
# Change Log   :
#   v1.0  2026-07-16  Initial version
#   v1.1  2026-07-15  Patched file copy helpers: no timeout meant a stalled
#                      file operation could hang copy operations forever with
#                      zero output. Added timeout, retry with backoff, and
#                      periodic progress reporting. 
#   v1.2  2026-07-15  Fixed the actual bug the self-test was built to catch:
#                      model.quantizer.decode() expects codes as [n_q,B,T],
#                      but model.encode() returns [B,n_q,T] — confirmed by
#                      the self-test output itself (decoded embedding came
#                      back as [8,128,750] instead of [1,128,750], meaning
#                      quantizer.decode() read n_q=8 as the batch dim and
#                      used only 1 of 8 codebooks). Added the required
#                      .transpose(0,1) in both the self-test and embed().
#   v1.3  2026-07-15  Fixed "cudnn RNN backward can only be called in
#                      training mode": EnCodec's encoder/decoder each
#                      contain an LSTM, and cuDNN's fused LSTM kernel
#                      refuses to backward() through a layer that ran
#                      forward in eval mode. Disabled torch.backends.
#                      cudnn.enabled so PyTorch falls back to its native
#                      LSTM implementation, which has no such restriction.
#   v1.4  2026-07-15  Fixed CUDA OOM in evaluate(): it ran all 105 val
#                      clips through embed() (full EnCodec encoder/decoder,
#                      LSTM included) in ONE batch, vs. batch size 8
#                      throughout training — a 13x jump with no batching
#                      at all. Rewrote evaluate() to batch the embed() step
#                      and every transform+extraction step, with explicit
#                      torch.cuda.empty_cache() between batches. Also
#                      persisted per-epoch (loss, bit_acc) history in the
#                      checkpoint and results JSON — training finished all
#                      30 epochs at bit_acc=0.4965 (chance level for 32-bit
#                      prediction), which needs the full history to
#                      diagnose and is a separate, more important question
#                      than this OOM fix.
#   v1.5  2026-07-15  Fixed a version-skew crash: the checkpoint already on
#                      PROJECT_DIR was written by v1.3, before final_loss/
#                      final_bit_acc/epoch_history existed in the
#                      checkpoint dict — resuming into it (training already
#                      complete, loop skipped) left those None, crashing
#                      the final summary print's :.4f formatting. Now
#                      formats gracefully. NOTE: this also means the
#                      epoch_history from the original 30-epoch run is
#                      genuinely gone (that checkpoint format never stored
#                      it) — only a fresh run under this version will
#                      populate it.
#   v1.6  2026-07-15  Actual results came back: BER flat at ~0.51 (chance)
#                      across EVERY transform tested, including gain=1.0
#                      and 320kbps compression — the mildest cases
#                      available. Ruled out "learned but fragile" (would
#                      show BER climbing with severity) in favor of "never
#                      learned anything." Widened HyperNetRobust's FiLM
#                      bound 5x (gamma +-50%, beta +-0.5, was +-10%/+-0.1)
#                      since the old bound likely didn't leave enough room
#                      to encode 32 bits against the reconstruction loss.
#                      Considered redesigning the extractor to read
#                      EnCodec's own latent instead of raw waveform, but
#                      that requires re-encoding through model.encode()'s
#                      non-differentiable quantization step, which would
#                      sever the gradient path to the HyperNetwork
#                      entirely — rejected, not just left alone. Added
#                      FORCE_FRESH_START (True for this run) since the
#                      existing checkpoint holds chance-level weights and
#                      reports all 30 epochs already done, which would
#                      otherwise skip training and reuse those weights.
#   v1.7  2026-07-15  Wider FiLM bound (v1.6) made zero difference — 7
#                      epochs, still flat at chance, statistically
#                      indistinguishable from the tight-bound run. Two
#                      different signal strengths producing identical
#                      non-learning rules out "signal too weak" and points
#                      at something structural. Found the likely cause:
#                      BATCH_SIZE (8) equals n_q (8 codebooks at 6kbps), so
#                      codes.shape at the real training batch size is
#                      (8,8,750) — square in the first two dims. The
#                      original self-test only validated the
#                      codes.transpose(0,1) fix at batch=1, where
#                      codes.shape=(1,8,750) is asymmetric and a mixup
#                      would be obvious; at batch=8 a batch/codebook-axis
#                      mixup would be numerically invisible in an aggregate
#                      check. Rewrote the self-test to run at the actual
#                      BATCH_SIZE and added a cross-contamination check:
#                      modulate only example 0 with an unmistakable shift
#                      and verify every other example's output is exactly
#                      unchanged. If contamination is found, this exactly
#                      explains chance-level training regardless of FiLM
#                      bound — per-example gradient would be incoherently
#                      mixed across the batch every step.
#   v1.8  2026-07-15  Self-test passed at real batch size (contamination
#                      hypothesis ruled out too) but training still flat
#                      at chance through epoch 9. Ran an isolated,
#                      codec-free sanity check (03b_extractor_sanity_
#                      check.py) of ExtractorRobust + the training loop
#                      against a synthetic, obviously-strong injected
#                      signal: bit_acc climbed steadily (0.47 -> 0.62 over
#                      150 steps), confirming the optimizer/training-loop
#                      mechanics work fine — the bottleneck is specifically
#                      that whatever signal survives the EnCodec embed->
#                      decode pipeline is too weak or too slow-to-learn
#                      relative to LAMBDA_RECON's pull back toward no
#                      change. Rebalanced loss weights: LAMBDA_RECON
#                      1.0->0.1, LAMBDA_ROBUST 5.0->10.0, deliberately
#                      prioritizing "can it learn to embed/extract at all"
#                      over audio quality for now — quality gets dialed
#                      back in once bit_acc actually moves.
#
#   v1.9  2026-07-15  Download progress now uses tqdm instead of time-throttled
#                      prints, and skips re-downloading a file that already exists
#                      locally with a matching size (verified against the server via
#                      file size check, not just trusted by filename).
#   v2.0  2026-07-15  Reweighted loss (v1.8) made zero difference either —
#                      fourth configuration, fourth statistically identical
#                      chance-level result. Ruled out hyperparameters
#                      entirely at this point; the common factor across
#                      ALL four attempts was the insertion point itself.
#                      embed() modulated codes.decode()'d from FIXED
#                      integer codes — a continuous vector that never went
#                      through an actual quantization step, i.e. off the
#                      manifold of inputs the decoder was ever trained on.
#                      EnCodec's paper confirms a straight-through
#                      estimator exists for gradient through quantization,
#                      but only via the training-mode forward path
#                      (quantizer(x, frame_rate, bandwidth) -> QuantizedResult),
#                      not encode()/decode(). Rewrote embed() to modulate
#                      the ENCODER's raw pre-quantization output instead,
#                      then let the real quantizer do its actual job —
#                      always producing on-manifold output. Rewrote the
#                      self-test to introspect and validate this new path
#                      specifically, including a gradient probe (does
#                      d(decoder(quantizer(raw_emb)))/d(raw_emb) actually
#                      have nonzero norm) — since if the straight-through
#                      estimator isn't active on this call path either,
#                      this pivot doesn't help and we'd want to know before
#                      training, not after another 2 hours.
#   v2.1  2026-07-15  The v2.0 gradient probe did exactly its job: caught
#                      "RuntimeError: element 0 of tensors does not require
#                      grad" before training started. Cause: model.quantizer
#                      inherited .eval() from the frozen model, and its
#                      internal straight-through path is gated on
#                      self.training — same category of bug as the earlier
#                      LSTM/cuDNN issue, different symptom. Implemented the
#                      straight-through estimator manually instead of
#                      relying on the library's internal one: quantizer
#                      call stays under no_grad() (matches its frozen
#                      nature, no risk of side-effecting EMA codebook
#                      updates), then emb + (quantized - emb).detach()
#                      gives forward=quantized, backward=identity,
#                      independent of any library-internal train/eval
#                      branching. Applied to both the self-test's gradient
#                      probe and embed() itself.
#   v2.2  2026-07-15  Five configurations in a row (post-quant tight bound,
#                      post-quant wide bound, batch-validated fix,
#                      reweighted loss, pre-quant with verified gradient
#                      flow) all produced statistically identical
#                      chance-level bit_acc through epoch 10 — pipeline
#                      correctness is no longer in question, so this isn't
#                      another bug fix. TEMPORARY diagnostic: KEY_BITS
#                      32->1, N_EPOCHS 30->8, to distinguish "32 bits is
#                      more capacity than this training budget supports"
#                      (1 bit should learn quickly if so) from "the
#                      straight-through gradient isn't a locally useful
#                      signal at this perturbation scale" (1 bit would
#                      also fail to learn if so). REVERT KEY_BITS and
#                      N_EPOCHS to 32/30 once this check has been read.
#   v2.3  2026-07-15  Self-test's gradient probe reported a strong,
#                      healthy norm (87.87) — the v2.0/v2.1 pivot is
#                      validated and working. Hit a NEW OOM crash in
#                      compression_proxy() (a separate, correctly
#                      no_grad()-wrapped encoder call for the non-
#                      differentiable augmentation) — but the real cause
#                      was upstream: embed()'s OWN encoder call
#                      (model.encoder(wav_batch_1ch)) ran outside no_grad()
#                      to get gradient to the HyperNetwork, which
#                      unnecessarily retained the ENTIRE encoder's
#                      intermediate activations for a backward pass that
#                      was never needed — we only need gradient to reach
#                      gamma/beta via the FiLM multiply, not through the
#                      encoder's own many conv+LSTM layers. By the time
#                      compression_proxy() ran its own encode() call, that
#                      accumulated graph had exhausted the T4's 14.56GB.
#                      Wrapped embed()'s encoder call in no_grad() — same
#                      principle already applied to the quantizer call.
#   v2.4  2026-07-15  1-bit diagnostic SUCCEEDED: bit_acc 0.64->0.99 by
#                      epoch 2, BER=0.0 across every benign transform
#                      tested. Confirms the v2.0-v2.3 pivot works and 32
#                      bits was a capacity/training-budget mismatch, not a
#                      pipeline defect. TEMPORARY: KEY_BITS 1->16 to find
#                      the practical capacity ceiling for this training
#                      budget before committing to a final bit count for
#                      the full 30-epoch production run. N_EPOCHS stays at
#                      8 for this check.
#   v2.5  2026-07-15  16 bits completely flat through epoch 4 — same
#                      signature as the original 32-bit failure, no
#                      partial drift at all (unlike 1-bit, which was
#                      already at 0.64 by epoch 1). Binary-searching
#                      toward the capacity ceiling: KEY_BITS 16->4.
#   v2.6  2026-07-15  4 bits confirmed a real, clean result: bit_acc
#                      0.51->0.91 over 8 epochs, monotonic, still climbing
#                      at the end, BER ~0.02 on real held-out benign
#                      transforms. Narrowing further between "clearly
#                      works" (4) and "zero movement" (16): KEY_BITS 4->8.
#   v2.7  2026-07-15  8 bits flat through 4 epochs (0.50-0.50) — same dead
#                      signature as 16 bits, not 4's immediate drift.
#                      Ceiling confirmed between 4 and 8. KEY_BITS 8->6.
#   v3.0  2026-07-15  Capacity search concluded. Full 30-epoch run at 6
#                      bits: real, generalizing learning (bit_acc
#                      0.50->0.87, BER~0.11 on held-out transforms,
#                      tracking training accuracy closely — not
#                      overfitting) but still oscillating/not fully
#                      settled even at epoch 30, and clearly worse than
#                      4-bit's near-perfect result at a QUARTER the epoch
#                      budget. Sharp capacity cliff confirmed between 4
#                      and 6-8 for this training budget (945 clips, batch
#                      8, this LR), not a gradual slope. COMMITTED:
#                      KEY_BITS=4, N_EPOCHS=30 for the production E1
#                      result — this run's output is what goes in Table 2.
#
# !pip install torch torchaudio encodec pydub h5py matplotlib tqdm
# =============================================================================

!pip install -q encodec pydub h5py matplotlib tqdm

import torch
import sys

if not torch.cuda.is_available():
    print("\n[ERROR] GPU not detected!")
    print("This script requires a GPU")
    print("Please switch your Colab runtime to a T4 GPU and restart.")
    sys.exit(1)
print("CUDA available: True. Proceeding...")

# EnCodec's encoder/decoder each contain an LSTM bottleneck (not just conv
# layers). The whole model is frozen via model.eval() below, which is
# correct for the conv weights, but cuDNN's fused LSTM kernel refuses to
# run backward() through a layer whose forward pass ran in eval mode —
# that's a hard cuDNN restriction, not something specific to this script.
# We still need gradients to flow THROUGH the frozen decoder to reach the
# HyperNetwork, so disable cuDNN's RNN kernel and fall back to PyTorch's
# native LSTM implementation, which has no such restriction. Slightly
# slower than the fused kernel, but this model's LSTM is small (750-frame
# sequences, modest hidden size) so the difference should be minor —
# revisit only if epoch time turns out to be a real bottleneck.
torch.backends.cudnn.enabled = False

import os
import json
import random
import shutil
import time
import datetime
import tempfile

import numpy as np
import h5py
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from encodec import EncodecModel

DEVICE = torch.device("cuda")


def now():
    return datetime.datetime.now().strftime("%H:%M:%S")


# --- Google Drive Helper Functions --------------------------------------------
PROJECT_DIR = "/content/drive/MyDrive/paper/HyperFrag/"
MAX_RETRIES = 3
TIMEOUT_SECONDS = 60


def copy_to_project(local_filepath, remote_filename):
    """Copy a local file to PROJECT_DIR (Google Drive)."""
    project_filepath = os.path.join(PROJECT_DIR, remote_filename)
    os.makedirs(os.path.dirname(project_filepath), exist_ok=True)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            shutil.copy2(local_filepath, project_filepath)
            return
        except Exception as e:
            print(f"[PROJECT_DIR] copy attempt {attempt}/{MAX_RETRIES} for "
                  f"{remote_filename} failed: {e}")
            if attempt == MAX_RETRIES:
                raise
            time.sleep(5)


def copy_from_project(remote_filename, local_filepath, verbose=True, skip_if_exists=True):
    """Copy a file from PROJECT_DIR (Google Drive) to local path."""
    project_filepath = os.path.join(PROJECT_DIR, remote_filename)
    if not os.path.isfile(project_filepath):
        print(f"[PROJECT_DIR] {remote_filename} not found in PROJECT_DIR")
        return False
    
    # Skip if local copy already exists and sizes match
    if skip_if_exists and os.path.exists(local_filepath):
        local_size = os.path.getsize(local_filepath)
        remote_size = os.path.getsize(project_filepath)
        if remote_size == local_size:
            print(f"[PROJECT_DIR] {local_filepath} already exists locally "
                  f"({local_size / 1e6:.1f} MB), skipping copy.")
            return True
        print(f"[PROJECT_DIR] {local_filepath} exists locally but size differs from "
              f"PROJECT_DIR ({local_size} vs {remote_size} bytes) — re-copying.")
    
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            os.makedirs(os.path.dirname(local_filepath), exist_ok=True)
            total_size = os.path.getsize(project_filepath)
            
            with open(local_filepath, "wb") as f, tqdm(
                total=total_size, unit="B", unit_scale=True, unit_divisor=1024,
                desc=f"[PROJECT_DIR] {remote_filename}", disable=not verbose,
            ) as pbar:
                with open(project_filepath, "rb") as src:
                    while True:
                        chunk = src.read(8192)
                        if not chunk:
                            break
                        f.write(chunk)
                        pbar.update(len(chunk))
            return True
        except Exception as e:
            print(f"[PROJECT_DIR] download attempt {attempt}/{MAX_RETRIES} for "
                  f"{remote_filename} failed: {e}")
            if attempt == MAX_RETRIES:
                return False
            time.sleep(5)
    return False


def project_file_exists(remote_filename):
    """Check if a file exists in PROJECT_DIR (Google Drive)."""
    project_filepath = os.path.join(PROJECT_DIR, remote_filename)
    return os.path.isfile(project_filepath)


# --- Config ---------------------------------------------------------------
LOCAL_SCRATCH = "/content/scratch"
os.makedirs(LOCAL_SCRATCH, exist_ok=True)
LOCAL_H5 = f"{LOCAL_SCRATCH}/dataset_e0.h5"
LOCAL_CKPT = f"{LOCAL_SCRATCH}/e1_robust_checkpoint.pth"
LOCAL_RESULTS = f"{LOCAL_SCRATCH}/exp1_robust_results.json"
LOCAL_FIG = f"{LOCAL_SCRATCH}/fig_03_01_ber_by_transform.png"

SR = 24000
TARGET_BANDWIDTH_KBPS = 6.0  # same project-wide operating point as 02_e0_codec_sanity.py
KEY_BITS = 4   # FINAL — capacity search done. 1 bit: instant, trivial. 4 bits: clean
# convergence, BER~0.02 at just 8 epochs. 6 bits: real but slow, BER~0.11 even at 30
# epochs, still not settled. 8/16/32 bits: complete failure, chance level. 4 is the
# committed production value for this training budget.
BATCH_SIZE = 8
N_EPOCHS = 30  # back to full production budget
CHECKPOINT_EVERY_EPOCHS = 4
LR = 2e-4
LAMBDA_RECON = 0.1   # was 1.0 — deliberately weakened for now (see v1.8 changelog)
LAMBDA_ROBUST = 10.0  # was 5.0 — deliberately strengthened for now
VAL_FRACTION = 0.10
SEED = 20260716
# TRUE for this run on purpose: the checkpoint already in PROJECT_DIR holds weights
# trained under the old (too-tight) FiLM bound and reports epoch=29 (all 30
# done), so a normal resume would skip training entirely and reuse those
# chance-level weights. Set back to False after this run confirms the wider
# bound actually learns something, so future interruptions resume normally
# instead of re-training from scratch every time.
FORCE_FRESH_START = True

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# --- Step 1: load corpus into memory ---------------------------------------
def load_corpus():
    print(f"[{now()}] Downloading dataset_e0.h5 from PROJECT_DIR...")
    if not copy_from_project("dataset_e0.h5", LOCAL_H5):
        raise SystemExit("dataset_e0.h5 not found in PROJECT_DIR — run 01_e0_dataextract.py first.")

    ids, sources, wavs = [], [], []
    with h5py.File(LOCAL_H5, "r") as h5f:
        for gid in ("jamendo", "musdb18"):
            for tid in h5f[gid].keys():
                ids.append(tid)
                sources.append(gid)
                wavs.append(h5f[gid][tid][:])
    wavs = np.stack(wavs).astype(np.float32)  # [N, T]
    print(f"[{now()}] Loaded {len(ids)} clips into memory ({wavs.nbytes / 1e9:.2f} GB).")

    idx = np.arange(len(ids))
    rng = np.random.RandomState(SEED)
    rng.shuffle(idx)
    n_val = int(len(idx) * VAL_FRACTION)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    print(f"[{now()}] Split: {len(train_idx)} train, {len(val_idx)} val.")
    return wavs, ids, sources, train_idx, val_idx


# --- Step 2: load frozen EnCodec + pre-flight self-test ---------------------
def load_codec_and_selftest(sample_wavs_np):
    """PIVOTED 2026-07-15 (v2.0): four training configurations (tight FiLM
    bound, 5x wider bound, batch-validated fix, 10x-reweighted loss) all
    produced statistically identical chance-level results. The common
    factor across all of them: injection happened AFTER quantization, on
    codes.decode()'d output that never went through an actual quantization
    step — an arbitrary continuous vector the decoder was never trained to
    expect, since it was only ever trained on genuine quantizer outputs.
    EnCodec's own paper confirms a straight-through estimator exists for
    gradient flow through quantization, but only via the training-mode
    forward path, not the encode()/decode() split. This self-test
    introspects and validates THAT path — modulating the encoder's raw
    pre-quantization output, then letting the real quantizer do its actual
    job — before any training time is spent on it."""
    print(f"[{now()}] Loading pretrained EnCodec (24kHz), target bandwidth "
          f"{TARGET_BANDWIDTH_KBPS} kbps...")
    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(TARGET_BANDWIDTH_KBPS)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model = model.to(DEVICE)

    B_test = sample_wavs_np.shape[0]
    wav_t = torch.from_numpy(sample_wavs_np).float().unsqueeze(1).to(DEVICE)  # [B_test,1,T]

    print(f"[{now()}] Running pre-flight self-test of the PRE-quantization "
          f"injection path (encoder -> FiLM -> quantizer.forward -> decoder)...")
    print(f"[selftest] type(model.quantizer) = {type(model.quantizer)}")
    print(f"[selftest] dir(model.quantizer) = "
          f"{[m for m in dir(model.quantizer) if not m.startswith('_')]}")
    print(f"[selftest] model.frame_rate = {getattr(model, 'frame_rate', 'MISSING ATTRIBUTE')}")
    print(f"[selftest] model.bandwidth = {getattr(model, 'bandwidth', 'MISSING ATTRIBUTE')}")

    with torch.no_grad():
        encoded_frames = model.encode(wav_t)
        wav_via_blackbox = model.decode(encoded_frames)

        raw_emb = model.encoder(wav_t)
        print(f"[selftest] encoder output (pre-quantization) shape: {tuple(raw_emb.shape)}, "
              f"mean={raw_emb.mean().item():.4f}, std={raw_emb.std().item():.4f}, "
              f"min={raw_emb.min().item():.4f}, max={raw_emb.max().item():.4f}")

        frame_rate = getattr(model, "frame_rate", None)
        bandwidth = getattr(model, "bandwidth", None) or TARGET_BANDWIDTH_KBPS
        qres = None
        last_err = None
        for attempt_desc, attempt_fn in [
            ("model.quantizer(raw_emb, frame_rate, bandwidth)",
             lambda: model.quantizer(raw_emb, frame_rate, bandwidth)),
            ("model.quantizer(raw_emb, frame_rate=frame_rate, bandwidth=bandwidth)",
             lambda: model.quantizer(raw_emb, frame_rate=frame_rate, bandwidth=bandwidth)),
            ("model.quantizer.forward(raw_emb, frame_rate, bandwidth)",
             lambda: model.quantizer.forward(raw_emb, frame_rate, bandwidth)),
        ]:
            try:
                qres = attempt_fn()
                print(f"[selftest] quantizer call succeeded via: {attempt_desc}")
                break
            except Exception as e:
                last_err = e
                print(f"[selftest]   tried {attempt_desc} -> failed: {e}")
        if qres is None:
            print(f"[selftest] [FATAL] every quantizer forward-call signature tried failed. "
                  f"Last error: {last_err}")
            print(f"[selftest]   Inspect dir(model.quantizer) above and adjust embed() manually "
                  f"to call the correct training-mode method.")
            raise SystemExit("Self-test failed — could not find a working quantizer forward call.")

        print(f"[selftest] quantizer result type: {type(qres)}")
        print(f"[selftest]   attributes: {[a for a in dir(qres) if not a.startswith('_')]}")
        if not hasattr(qres, "quantized"):
            print(f"[selftest] [FATAL] result has no .quantized attribute — inspect the "
                  f"attributes list above and adjust embed() to use the correct field.")
            raise SystemExit("Self-test failed — quantizer result has no .quantized field.")
        quantized = qres.quantized
        print(f"[selftest]   quantized.shape: {tuple(quantized.shape)}")

        wav_via_prequant_path = model.decoder(quantized)
        n = min(wav_via_blackbox.shape[-1], wav_via_prequant_path.shape[-1])
        diff = (wav_via_blackbox[..., :n] - wav_via_prequant_path[..., :n]).abs().max().item()
        print(f"[selftest] max abs diff, model.decode() vs encoder->quantizer->decoder path: "
              f"{diff:.6f} (some difference is EXPECTED here — model.decode() uses codes from "
              f"model.encode()'s own internal call, this path re-quantizes independently; a "
              f"huge/NaN value would indicate a real problem, a small nonzero value does not)")

    # Gradient probe: does gradient reach raw_emb through a MANUALLY
    # implemented straight-through estimator (forward = quantized value,
    # backward = identity), rather than relying on the library's own ST
    # path, which turned out to be gated on model.quantizer.training —
    # inherited as False from the frozen model's .eval() call, exactly the
    # same category of issue as the earlier LSTM/cuDNN one. Implementing ST
    # ourselves sidesteps that entirely and avoids ever calling the
    # quantizer outside no_grad(), which also rules out any risk of
    # accidentally re-enabling EMA codebook updates