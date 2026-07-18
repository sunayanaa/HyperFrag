# =============================================================================
# Program      : 04_e2_fragile_path.py
# Version      : 1.0
# Description  : Experiment 2 — Fragile Path (Path B) training and evaluation.
#
#                This is the paper's actual novelty claim (per Mr CG's
#                scope-narrowing advice and the blueprint). Trains ONLY the
#                fragile-integrity branch: a HyperNetwork (key bits -> FiLM
#                gamma/beta) modulates a DIFFERENT slice of EnCodec's
#                pre-quantization latent than E1 used — channels
#                [D_total//4 : 2*D_total//4] (E1 used [0:D_total//4]) — so
#                the two paths don't collide when E3 combines them. Same
#                proven pipeline as E1: pre-quantization injection, manual
#                straight-through estimator, same architecture shapes for
#                the HyperNetwork/Extractor (renamed *Fragile).
#
#                The design goal is the OPPOSITE of E1 in one respect: Path
#                B should SURVIVE benign transforms (same augmentation set
#                as E1: noise/gain/resample/EQ/compression-proxy) but BREAK
#                under generative re-synthesis. Real generative-model
#                evaluation (MusicGen) is reserved for E4 — here, training
#                uses a cheap, well-motivated proxy for "regeneration loses
#                fine structure": Griffin-Lim magnitude-only reconstruction
#                (STFT -> discard phase -> iterative phase reconstruction
#                -> ISTFT). This directly operationalizes the blueprint's
#                own theoretical framing (§4.6, citing Griffin-Lim 1984) —
#                not an arbitrary choice of proxy.
#
#                LOSS: adds a hinge-style "fragile-break" term on top of
#                E1's recon + survive terms:
#                  L_break = relu(MARGIN - BCE(extractor(regen(x_wm)), K))
#                MARGIN = ln(2) (chance-level BCE for a 2-class prediction).
#                This only penalizes the network if bits are STILL
#                recoverable after regeneration (BCE below chance-level
#                confusion) — it does NOT reward pushing BCE arbitrarily
#                high, which would be an unbounded, unstable objective
#                (equivalent to training toward "confidently wrong," not
#                "uninformative"). Hinge keeps this numerically stable.
#
#                KEY_BITS=4, matching E1's proven-reliable capacity for
#                this training budget — no need to re-run E1's capacity
#                search, same architecture/budget, same conclusion applies.
#
#                N_EPOCHS=8 for this FIRST run, not 30. The hinge
#                fragile-break term is a genuinely new loss component,
#                untested at this scale — better to confirm it behaves
#                (does BER-under-regen actually rise while BER-under-
#                benign stays low?) with a short run before committing to
#                a full 30-epoch production run, same discipline that
#                caught real problems early in E1.
#
# PRE-FLIGHT SELF-TEST: slimmer than E1's — the pre-quantization API call
# signature is already known-good from E1's discovery (model.quantizer(
# raw_emb, frame_rate, bandwidth) -> QuantizedResult with .quantized). This
# only re-runs the gradient probe (manual straight-through), since that's
# cheap and still worth confirming fresh in a new Colab session before
# training starts.
#
# INPUT:
#                  PROJECT_DIR (/content/drive/MyDrive/paper/HyperFrag/):
#                    dataset_e0.h5    — from 01_e0_dataextract.py
#
# STEPS:
#                  Step 1  Download dataset_e0.h5 from PROJECT_DIR (tqdm progress,
#                          skip if already local with matching size)
#                  Step 2  Load frozen EnCodec, slim self-test (gradient
#                          probe only) of the pre-quantization + manual
#                          straight-through path
#                  Step 3  Train HyperNetFragile + ExtractorFragile:
#                          recon loss + survive loss (clean + benign-
#                          augmented) + hinge fragile-break loss (Griffin-
#                          Lim regen proxy)
#                  Step 4  Checkpoint every CHECKPOINT_EVERY_EPOCHS epochs
#                          to PROJECT_DIR
#                  Step 5  Evaluate on held-out split: BER under the SAME
#                          benign-transform battery as E1 (want LOW), and
#                          BER under Griffin-Lim regen at a few severities
#                          (want HIGH, near chance ~0.5)
#
# OUTPUT FILES : (stored at /content/drive/MyDrive/paper/HyperFrag/)
#                  e2_fragile_checkpoint.pth  (overwritten each checkpoint)
#                  exp2_fragile_results.json
#                      {training: {final_loss, final_bit_acc, epoch_history},
#                       ber_benign: {same structure as E1's ber_by_transform},
#                       ber_regen: {n_iter: ber}}
#                  fig_04_01_survive_vs_break.png
#                      Bar chart: BER under benign transforms (should be
#                      low) vs. BER under regen proxy at each severity
#                      (should be high) — the core "survives vs breaks"
#                      visual for this experiment
#
# GPU Required : YES
# Dependencies : torch, torchaudio, encodec, pydub, h5py, matplotlib, tqdm
#
# Change Log   :
#   v1.0  2026-07-16  Initial version. 
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

# Same fix as 03_e1_robust_path.py — EnCodec's encoder/decoder contain an
# LSTM; cuDNN's fused kernel can't backward() through a layer that ran
# forward in eval mode. Fall back to PyTorch's native LSTM implementation.
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
LOCAL_CKPT = f"{LOCAL_SCRATCH}/e2_fragile_checkpoint.pth"
LOCAL_RESULTS = f"{LOCAL_SCRATCH}/exp2_fragile_results.json"
LOCAL_FIG = f"{LOCAL_SCRATCH}/fig_04_01_survive_vs_break.png"

SR = 24000
TARGET_BANDWIDTH_KBPS = 6.0  # same project-wide operating point as 02/03
KEY_BITS = 4   # matches E1's proven-reliable capacity for this training budget
BATCH_SIZE = 8
N_EPOCHS = 30   # FIRST run — see header for why not 30 yet
CHECKPOINT_EVERY_EPOCHS = 4
LR = 2e-4
LAMBDA_RECON = 0.1
LAMBDA_SURVIVE = 10.0
LAMBDA_BREAK = 5.0
MARGIN = 0.6931471805599453  # ln(2) — chance-level BCE, the hinge target
VAL_FRACTION = 0.10
SEED = 20260716
FORCE_FRESH_START = True  # new program, no existing checkpoint to conflict with

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


# --- Step 2: load frozen EnCodec + slim self-test ---------------------------
def load_codec_and_selftest(sample_wavs_np):
    """Slimmer than E1's — the API call signature is already known-good
    from E1's discovery. Only re-confirms the gradient probe (manual
    straight-through) fresh in this session, since that's cheap and still
    worth checking before training starts."""
    print(f"[{now()}] Loading pretrained EnCodec (24kHz), target bandwidth "
          f"{TARGET_BANDWIDTH_KBPS} kbps...")
    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(TARGET_BANDWIDTH_KBPS)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model = model.to(DEVICE)

    frame_rate = getattr(model, "frame_rate", 75)
    bandwidth = getattr(model, "bandwidth", TARGET_BANDWIDTH_KBPS)

    wav_t = torch.from_numpy(sample_wavs_np).float().unsqueeze(1).to(DEVICE)
    with torch.no_grad():
        raw_emb = model.encoder(wav_t)
    d_total = raw_emb.shape[1]
    print(f"[{now()}] Encoder output shape: {tuple(raw_emb.shape)} (D_total={d_total})")

    raw_emb_leaf = raw_emb.clone().detach().requires_grad_(True)
    with torch.no_grad():
        qres = model.quantizer(raw_emb_leaf, frame_rate, bandwidth)
    quantized_st = raw_emb_leaf + (qres.quantized - raw_emb_leaf).detach()
    wav_probe = model.decoder(quantized_st)
    wav_probe.sum().backward()
    grad_norm = raw_emb_leaf.grad.norm().item() if raw_emb_leaf.grad is not None else 0.0
    print(f"[selftest] gradient probe (manual straight-through) — "
          f"||d(decoder(...))/d(raw_emb)||: {grad_norm:.6f}")
    if grad_norm < 1e-8:
        print(f"[selftest] [FATAL] no gradient through the pre-quantization path in this "
              f"session — this worked in 03_e1_robust_path.py, so something about this "
              f"environment differs. Do not proceed to training.")
        raise SystemExit("Self-test failed — no gradient through quantizer.")
    print(f"[selftest] PASSED. Proceeding to training.")

    return model, d_total, frame_rate, bandwidth


# --- Trainable modules (same architecture as E1, renamed for Path B) -------
class HyperNetFragile(nn.Module):
    def __init__(self, key_bits, d_fragile, hidden=128):
        super().__init__()
        self.d_fragile = d_fragile
        self.net = nn.Sequential(
            nn.Linear(key_bits, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2 * d_fragile),
        )

    def forward(self, key_bits):
        out = self.net(key_bits)
        gamma_raw, beta_raw = out[:, :self.d_fragile], out[:, self.d_fragile:]
        gamma = 1.0 + 0.5 * torch.tanh(gamma_raw)  # same bound as E1's final working config
        beta = 0.5 * torch.tanh(beta_raw)
        return gamma, beta


class ExtractorFragile(nn.Module):
    def __init__(self, key_bits, channels=32):
        super().__init__()
        def block(cin, cout, stride):
            return nn.Sequential(
                nn.Conv1d(cin, cout, kernel_size=9, stride=stride, padding=4),
                nn.BatchNorm1d(cout), nn.ReLU(),
            )
        self.net = nn.Sequential(
            block(1, channels, 4),
            block(channels, channels * 2, 4),
            block(channels * 2, channels * 4, 4),
            block(channels * 4, channels * 4, 4),
            block(channels * 4, channels * 4, 4),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(channels * 4, key_bits)

    def forward(self, wav):  # wav: [B, T]
        h = self.net(wav.unsqueeze(1)).squeeze(-1)
        return self.head(h)


# --- Embedding pipeline (Path B: channels [D_total//4 : 2*D_total//4]) ------
def embed_fragile(model, wav_batch_1ch, hypernet, key_bits, d_start, d_end, frame_rate, bandwidth):
    """wav_batch_1ch: [B,1,T]. Returns watermarked waveform [B,1,T].
    Same proven pipeline as E1's embed(): encoder outside grad tracking
    (frozen, only need its VALUE), FiLM on a channel slice, manual
    straight-through through the quantizer, frozen decoder. The only
    difference from E1 is WHICH channel slice gets modulated."""
    with torch.no_grad():
        raw_emb = model.encoder(wav_batch_1ch)  # [B, D, T]
    gamma, beta = hypernet(key_bits)
    gamma = gamma.unsqueeze(-1)
    beta = beta.unsqueeze(-1)
    emb_fragile_part = raw_emb[:, d_start:d_end, :] * gamma + beta
    emb_mod = torch.cat([raw_emb[:, :d_start, :], emb_fragile_part, raw_emb[:, d_end:, :]], dim=1)
    with torch.no_grad():
        qres = model.quantizer(emb_mod, frame_rate, bandwidth)
    emb_mod_st = emb_mod + (qres.quantized - emb_mod).detach()
    wav_wm = model.decoder(emb_mod_st)
    return wav_wm


def match_length(wav, target_len):
    cur = wav.shape[-1]
    if cur == target_len:
        return wav
    if cur > target_len:
        return wav[..., :target_len]
    return F.pad(wav, (0, target_len - cur))


# --- Differentiable benign augmentations (same as E1) -----------------------
def differentiable_eq(wav, sr=SR, n_fft=1024, hop=256, max_db=6.0, n_bands=6):
    window = torch.hann_window(n_fft, device=wav.device)
    spec = torch.stft(wav, n_fft=n_fft, hop_length=hop, window=window, return_complex=True)
    n_bins = spec.shape[-2]
    band_gains_db = (torch.rand(wav.shape[0], 1, n_bands, device=wav.device) * 2 - 1) * max_db
    gain_curve_db = F.interpolate(band_gains_db, size=n_bins, mode="linear", align_corners=True)
    gain_curve_db = gain_curve_db.squeeze(1).unsqueeze(-1)
    gain_lin = 10 ** (gain_curve_db / 20)
    spec_eq = spec * gain_lin
    wav_eq = torch.istft(spec_eq, n_fft=n_fft, hop_length=hop, window=window, length=wav.shape[-1])
    return wav_eq


def augment_diff(wav):
    choice = random.choice(["identity", "noise", "gain", "resample", "eq"])
    if choice == "identity":
        return wav
    if choice == "noise":
        snr_db = random.uniform(20, 40)
        sig_power = wav.pow(2).mean(dim=-1, keepdim=True)
        noise_power = sig_power / (10 ** (snr_db / 10))
        return wav + torch.randn_like(wav) * noise_power.sqrt()
    if choice == "gain":
        return wav * random.uniform(0.7, 1.3)
    if choice == "resample":
        low_sr = random.choice([16000, 22050])
        down = torchaudio.functional.resample(wav, SR, low_sr)
        up = torchaudio.functional.resample(down, low_sr, SR)
        return match_length(up, wav.shape[-1])
    return differentiable_eq(wav)  # "eq"


# --- Non-differentiable "benign compression" proxy (same as E1) -------------
def compression_proxy(model, wav_1ch, low_bandwidths=(1.5, 3.0)):
    bw = random.choice(low_bandwidths)
    model.set_target_bandwidth(bw)
    with torch.no_grad():
        frames = model.encode(wav_1ch)
        out = model.decode(frames)
    model.set_target_bandwidth(TARGET_BANDWIDTH_KBPS)  # restore
    return out


# --- NEW: generative-regeneration proxy (Griffin-Lim, magnitude-only) ------
def regen_proxy_griffinlim(wav, n_fft=1024, hop=256, n_iter=32):
    """Discards phase entirely (keeps only magnitude), then iteratively
    reconstructs phase from scratch via Griffin-Lim — a cheap, well-
    motivated stand-in for "a generative model loses fine structure but
    preserves macro spectral content," directly operationalizing the
    blueprint's own theoretical framing (§4.6). Non-differentiable by
    design (wrapped in no_grad() at the call site) — this only needs to
    shape the EXTRACTOR's response, not backprop into the embedder."""
    window = torch.hann_window(n_fft, device=wav.device)
    spec = torch.stft(wav, n_fft=n_fft, hop_length=hop, window=window, return_complex=True)
    mag = spec.abs()
    wav_regen = torchaudio.functional.griffinlim(
        mag, window=window, n_fft=n_fft, hop_length=hop, win_length=n_fft,
        power=1.0, n_iter=n_iter, momentum=0.99, length=wav.shape[-1], rand_init=True,
    )
    return wav_regen


# --- Multi-resolution STFT loss (batched, same as E1) ------------------------
def multi_resolution_stft_loss_batch(ref, deg,
                                      resolutions=((1024, 120, 600), (2048, 240, 1200), (512, 50, 240))):
    total = 0.0
    for n_fft, hop, win in resolutions:
        window = torch.hann_window(win, device=ref.device)
        ref_spec = torch.stft(ref, n_fft=n_fft, hop_length=hop, win_length=win,
                               window=window, return_complex=True)
        deg_spec = torch.stft(deg, n_fft=n_fft, hop_length=hop, win_length=win,
                               window=window, return_complex=True)
        ref_mag = torch.clamp(ref_spec.abs(), min=1e-7)
        deg_mag = torch.clamp(deg_spec.abs(), min=1e-7)
        sc = torch.norm(ref_mag - deg_mag, p="fro", dim=(-2, -1)) / torch.norm(ref_mag, p="fro", dim=(-2, -1))
        mag = torch.mean(torch.abs(torch.log(ref_mag) - torch.log(deg_mag)), dim=(-2, -1))
        total = total + (sc + mag).mean()
    return total / len(resolutions)


# --- Training ----------------------------------------------------------------
def train(model, hypernet, extractor, wavs, train_idx, d_start, d_end, frame_rate, bandwidth):
    optimizer = torch.optim.Adam(
        list(hypernet.parameters()) + list(extractor.parameters()), lr=LR)
    bce_loss_fn = nn.BCEWithLogitsLoss()
    bce_loss_fn_noreduce = nn.BCEWithLogitsLoss(reduction="mean")

    start_epoch = 0
    final_loss, final_bit_acc = None, None
    epoch_history = []
    if FORCE_FRESH_START:
        print(f"[{now()}] FORCE_FRESH_START is True — ignoring any existing "
              f"checkpoint in PROJECT_DIR and starting from scratch.")
    elif copy_from_project("e2_fragile_checkpoint.pth", LOCAL_CKPT):
        ckpt = torch.load(LOCAL_CKPT, map_location=DEVICE)
        hypernet.load_state_dict(ckpt["hypernet_state"])
        extractor.load_state_dict(ckpt["extractor_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        epoch_history = ckpt.get("epoch_history", [])
        final_loss = ckpt.get("final_loss")
        final_bit_acc = ckpt.get("final_bit_acc")
        print(f"[{now()}] Resumed from checkpoint at epoch {ckpt['epoch']}, "
              f"continuing from epoch {start_epoch}.")
    else:
        print(f"[{now()}] No checkpoint found, starting fresh.")

    def checkpoint(epoch):
        torch.save({
            "hypernet_state": hypernet.state_dict(),
            "extractor_state": extractor.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "epoch_history": epoch_history,
            "final_loss": final_loss,
            "final_bit_acc": final_bit_acc,
        }, LOCAL_CKPT)
        copy_to_project(LOCAL_CKPT, "e2_fragile_checkpoint.pth")
        print(f"[{now()}] Checkpoint saved (epoch {epoch}) and uploaded to PROJECT_DIR.")

    n_train = len(train_idx)

    for epoch in range(start_epoch, N_EPOCHS):
        print(f"[{now()}] Epoch {epoch + 1}/{N_EPOCHS} starting...")
        perm = np.random.permutation(train_idx)
        epoch_losses, epoch_bit_accs, epoch_bce_regen = [], [], []

        for start in range(0, n_train, BATCH_SIZE):
            batch_idx = perm[start:start + BATCH_SIZE]
            if len(batch_idx) < 2:  # BatchNorm needs >=2 samples
                continue
            wav_batch = torch.from_numpy(wavs[batch_idx]).float().to(DEVICE)  # [B,T]
            key_bits_pm1 = (torch.randint(0, 2, (len(batch_idx), KEY_BITS), device=DEVICE) * 2 - 1).float()
            target = (key_bits_pm1 > 0).float()

            x_wm = embed_fragile(model, wav_batch.unsqueeze(1), hypernet, key_bits_pm1,
                                  d_start, d_end, frame_rate, bandwidth)
            x_wm_flat = x_wm.squeeze(1)

            recon = multi_resolution_stft_loss_batch(wav_batch, x_wm_flat)

            # Survive: clean + one random differentiable benign augmentation
            x_wm_aug_diff = augment_diff(x_wm_flat)
            with torch.no_grad():
                x_wm_aug_nondiff = compression_proxy(model, x_wm.detach()).squeeze(1)
                x_wm_aug_nondiff = match_length(x_wm_aug_nondiff, x_wm_flat.shape[-1])

            bits_clean = extractor(x_wm_flat)
            bits_aug_diff = extractor(match_length(x_wm_aug_diff, x_wm_flat.shape[-1]))
            bits_aug_nondiff = extractor(x_wm_aug_nondiff)
            bce_survive = (bce_loss_fn(bits_clean, target)
                           + bce_loss_fn(bits_aug_diff, target)
                           + bce_loss_fn(bits_aug_nondiff, target)) / 3.0

            # Break: Griffin-Lim regeneration proxy, hinge loss
            with torch.no_grad():
                x_wm_regen = regen_proxy_griffinlim(x_wm_flat.detach())
                x_wm_regen = match_length(x_wm_regen, x_wm_flat.shape[-1])
            bits_regen = extractor(x_wm_regen)
            bce_regen = bce_loss_fn_noreduce(bits_regen, target)
            l_break = F.relu(MARGIN - bce_regen)

            loss = (LAMBDA_RECON * recon
                    + LAMBDA_SURVIVE * bce_survive
                    + LAMBDA_BREAK * l_break)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bit_acc = ((bits_clean > 0).float() == target).float().mean().item()
            epoch_losses.append(loss.item())
            epoch_bit_accs.append(bit_acc)
            epoch_bce_regen.append(bce_regen.item())

        final_loss = float(np.mean(epoch_losses)) if epoch_losses else None
        final_bit_acc = float(np.mean(epoch_bit_accs)) if epoch_bit_accs else None
        mean_bce_regen = float(np.mean(epoch_bce_regen)) if epoch_bce_regen else None
        epoch_history.append({
            "epoch": epoch + 1, "loss": final_loss, "bit_acc": final_bit_acc,
            "bce_regen": mean_bce_regen,
        })
        print(f"[{now()}] Epoch {epoch + 1}/{N_EPOCHS}  loss={final_loss:.4f}  "
              f"bit_acc(clean)={final_bit_acc:.4f}  bce_regen={mean_bce_regen:.4f} "
              f"(target >= {MARGIN:.4f})")

        if (epoch + 1) % CHECKPOINT_EVERY_EPOCHS == 0 or (epoch + 1) == N_EPOCHS:
            checkpoint(epoch)

    return final_loss, final_bit_acc, epoch_history


# --- Evaluation: benign battery (want LOW BER) + regen battery (want HIGH) --
def real_mp3_roundtrip(wav_np, bitrate_kbps):
    from pydub import AudioSegment
    import soundfile as sf
    with tempfile.NamedTemporaryFile(suffix=".wav") as wav_f, \
         tempfile.NamedTemporaryFile(suffix=".mp3") as mp3_f:
        sf.write(wav_f.name, wav_np, SR)
        AudioSegment.from_wav(wav_f.name).export(mp3_f.name, format="mp3", bitrate=f"{bitrate_kbps}k")
        seg = AudioSegment.from_mp3(mp3_f.name)
        out = np.array(seg.get_array_of_samples()).astype(np.float32)
        if seg.channels > 1:
            out = out.reshape(-1, seg.channels).mean(axis=1)
        out = out / (2 ** (8 * seg.sample_width - 1))
        if seg.frame_rate != SR:
            out = torchaudio.functional.resample(
                torch.from_numpy(out).unsqueeze(0), seg.frame_rate, SR).squeeze(0).numpy()
        return out


def evaluate(model, hypernet, extractor, wavs, val_idx, d_start, d_end, frame_rate, bandwidth,
             eval_batch_size=BATCH_SIZE):
    print(f"[{now()}] Evaluating on {len(val_idx)} held-out clips — benign transforms "
          f"(want LOW BER) and Griffin-Lim regen proxy (want HIGH BER, near chance ~0.5)...")
    hypernet.eval()
    extractor.eval()
    torch.cuda.empty_cache()

    x_wm_chunks, target_chunks = [], []
    with torch.no_grad():
        for start in range(0, len(val_idx), eval_batch_size):
            batch_idx = val_idx[start:start + eval_batch_size]
            wav_batch = torch.from_numpy(wavs[batch_idx]).float().to(DEVICE)
            key_bits_pm1 = (torch.randint(0, 2, (len(batch_idx), KEY_BITS), device=DEVICE) * 2 - 1).float()
            target = (key_bits_pm1 > 0).float()
            x_wm = embed_fragile(model, wav_batch.unsqueeze(1), hypernet, key_bits_pm1,
                                  d_start, d_end, frame_rate, bandwidth).squeeze(1)
            x_wm_chunks.append(x_wm.cpu().numpy())
            target_chunks.append(target.cpu().numpy())
            del wav_batch, key_bits_pm1, target, x_wm
            torch.cuda.empty_cache()
    x_wm_np = np.concatenate(x_wm_chunks, axis=0)
    target_np = np.concatenate(target_chunks, axis=0)
    print(f"[{now()}]   Watermarked {len(x_wm_np)} val clips. Scoring...")

    def ber_for_transform(transform_fn):
        n_wrong, n_total = 0, 0
        for start in range(0, len(x_wm_np), eval_batch_size):
            chunk_np = x_wm_np[start:start + eval_batch_size]
            chunk_target = target_np[start:start + eval_batch_size]
            chunk_t = torch.from_numpy(chunk_np).float().to(DEVICE)
            with torch.no_grad():
                variant = transform_fn(chunk_t)
                variant = match_length(variant, chunk_t.shape[-1])
                logits = extractor(variant)
            pred = (logits > 0).float().cpu().numpy()
            n_wrong += (pred != chunk_target).sum()
            n_total += chunk_target.size
            del chunk_t, variant, logits
        torch.cuda.empty_cache()
        return float(n_wrong / n_total)

    def ber_for_mp3(bitrate_kbps):
        n_wrong, n_total = 0, 0
        for start in range(0, len