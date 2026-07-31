# =============================================================================
# Program      : 13_anti_forensics_adaptive_adversary.py
# Version      : 1.4
# Description  : Adaptive Anti-Forensics Adversary (T-IFS expansion).
#
#                Every prior attack in this project either forges a NEW
#                watermark (07, overwriting) or simply regenerates content
#                and observes what happens (04/06/12, generative attacks).
#                Neither models an adversary who specifically KNOWS Path
#                B's fragility design and actively tries to DEFEAT it —
#                i.e., someone who wants to regenerate content while
#                making it still LOOK benignly-processed to our detector.
#                This is the standard "anti-forensics" question in a
#                forensics venue: does the fragility signal survive a
#                sophisticated, adaptive, gradient-based evasion attempt,
#                not just passive regeneration.
#
#                THREAT MODEL (gray-box, consistent with 07's precedent):
#                the adversary knows the published architecture (channels
#                [D/4:D/2], HyperNetwork+FiLM, same frozen EnCodec) but not
#                our trained weights or key. Critically, they do NOT know
#                the true embedded key either — this rules out the naive
#                "optimize the extractor to output the true bits" attack,
#                since the adversary has no target to optimize toward.
#
#                THE REALISTIC ATTACK THIS SCRIPT IMPLEMENTS: a
#                self-consistency evasion attack, standard in adversarial
#                ML (analogous to crafting adversarial examples on a
#                substitute model and testing transfer to a black-box
#                target). Phase 1: the adversary reproduces our published
#                methodology to train their OWN complete Path B clone
#                (own key, own weights, same architecture) — reusing
#                07's proven attacker-training pattern. Phase 2, per
#                clip: the adversary observes what their OWN surrogate
#                extractor reports on the defender's ORIGINAL (pre-
#                regeneration) watermarked audio — call this the
#                "should-look-like" target. They regenerate the audio
#                (DAC, full-clip, real), then run PGD to find a small
#                waveform perturbation making their OWN surrogate's
#                output on the regenerated+perturbed audio match that
#                target. This requires no knowledge of the true key at
#                all — only self-consistency against their own model.
#                The actual test: does this perturbation, crafted
#                entirely against the attacker's surrogate, ALSO fool the
#                DEFENDER's real trained extractor (from 05's joint
#                checkpoint)? If the defender's real BER (vs. the TRUE
#                key) drops back toward 0, evasion transferred and
#                succeeded. If it stays near chance, the fragility signal
#                survives this adaptive attack.
#
#                PGD OPERATES ENTIRELY IN THE WAVEFORM DOMAIN, against the
#                attacker's own fully-differentiable extractor — no
#                gradient needs to flow through EnCodec, DAC, or any
#                frozen/non-differentiable component. This is standard
#                Linf-bounded PGD, computationally cheap relative to
#                everything else in this project (no autoregressive
#                generation, no codec forward passes inside the PGD loop).
#                Tested across a range of perturbation budgets (epsilon)
#                to show the security margin as a function of attack
#                strength, not just a single point.
#
# PRE-FLIGHT SELF-TEST: PGD is genuinely new to this project (nothing
# else here does gradient-based adversarial optimization). Verifies the
# attacker's own surrogate loss actually DECREASES over PGD iterations on
# one clip before running the full battery — if the optimization isn't
# working at all, every downstream number would be meaningless.
#
# INPUT:
#                  PROJECT_DIR (/content/drive/MyDrive/paper/HyperFrag/):
#                    dataset_e0.h5              — from 01_e0_dataextract.py
#                    e3_joint_checkpoint.pth    — from 05_e3_joint_path.py
#                                                 (REQUIRED — the defender)
#
# STEPS:
#                  Step 1  Download dataset_e0.h5 + e3_joint_checkpoint.pth
#                          from PROJECT_DIR
#                  Step 2  Load frozen EnCodec, slim self-test
#                  Step 3  PHASE 1: train the attacker's own Path B clone
#                          (same recipe as 04/07, N_EPOCHS=30 directly —
#                          proven recipe, no cautious-first-run needed)
#                  Step 4  PHASE 2, self-test: confirm PGD reduces the
#                          attacker's own loss on one clip
#                  Step 5  PHASE 2, full battery: for each held-out clip
#                          and each epsilon, regenerate via DAC, run PGD
#                          against the attacker's surrogate, then score
#                          BOTH the attacker's own extractor (sanity — did
#                          the attack even work against its own target?)
#                          AND the defender's real extractor (the actual
#                          question — did it transfer?)
#
# OUTPUT FILES : (stored at /content/drive/MyDrive/paper/HyperFrag/)
#                  e13_attacker_checkpoint.pth  (attacker's trained clone)
#                  exp13_anti_forensics_results.json
#                      {attacker_training: {...},
#                       no_attack_baseline: {defender_ber_regen_only},
#                       per_epsilon: {eps: {attacker_own_ber, defender_ber,
#                                            mean_perturbation_l2}}}
#                  fig_13_01_evasion_transfer.png
#                      Defender's real BER vs. epsilon — does evasion
#                      transfer, and how does it scale with attack budget?
#
# GPU Required : YES
# Dependencies : torch, torchaudio, encodec, descript-audio-codec, h5py,
#                matplotlib, tqdm
#
# Change Log   :
#   v1.0  2026-07-28  Initial version
#   v1.1  2026-07-28  First production run's attacker converged to only
#                      ~0.62 bit_acc (loss barely moved over 30 epochs),
#                      far below the ~0.98 this exact recipe reaches in
#                      04/07. Diagnosed as likely RNG-state divergence:
#                      nn.Module weight init draws from the global torch
#                      RNG at construction time, and this script's
#                      self-test consumes a different number of prior
#                      random draws than 04/07's setup code does, so the
#                      "same" SEED lands on a different, apparently
#                      unlucky, initial-weight draw here. Fixed by
#                      explicitly re-seeding (ATTACKER_INIT_SEED, distinct
#                      from the corpus-split SEED) immediately before
#                      constructing the attacker's modules, decoupling
#                      their initialization from whatever preceded it.
#                      This is a diagnostic fix, not a confirmed root
#                      cause — if this run ALSO converges weakly, the
#                      problem is something else and this hypothesis is
#                      wrong.
#   v1.2  2026-07-28  v1.1's re-seed did NOT fix convergence (bit_acc
#                      0.607, essentially unchanged from v1.0's 0.622) —
#                      the RNG-divergence hypothesis is REJECTED, not
#                      confirmed. Exhaustively diffed this script's
#                      training loop against 04's working one: hyper-
#                      parameters, channel slicing, architecture classes,
#                      loss formula (including the /3.0 divisor),
#                      optimizer step order, and zero_grad() placement
#                      all match exactly — no discrepancy found. Added
#                      back mean_bce_regen tracking/logging, which 04 has
#                      and this script was missing — the hinge/break term
#                      is the one part of this recipe untested by 07's
#                      successful Path-A attacker (which never touches
#                      Griffin-Lim or the hinge loss at all), so it is the
#                      remaining suspect, but UNCONFIRMED. This is a
#                      diagnostic-visibility addition, not a fix — watch
#                      bce_regen's trajectory against 04's healthy
#                      ~0.70-0.99-and-climbing range this run.
#   v1.3  2026-07-28  ROOT CAUSE FOUND AND FIXED, confirmed via a
#                      standalone diagnostic script (debug_13_vs_04_
#                      divergence.py) that reproduced the exact failure
#                      using code believed identical to 04's, then a
#                      direct diff against 04's actual file located the
#                      real discrepancy: embed_fragile() wrapped its
#                      ENTIRE body — including hypernet's FiLM computation
#                      and the decoder call — inside a single
#                      torch.no_grad() block, and was missing the manual
#                      straight-through estimator line entirely (decoder
#                      was called on qres.quantized directly). Consequence:
#                      hypernet's output had requires_grad=False from the
#                      start, so hypernet NEVER received gradient and
#                      NEVER trained, in any of the prior runs — only the
#                      extractor trained, on a fixed, untrained, arbitrary
#                      embedding pattern with no real relationship to the
#                      key bits. This fully explains every symptom seen:
#                      bit_acc stuck near 0.5-0.6 regardless of random
#                      seed, bce_regen weakly-but-not-pathologically
#                      positive, and PGD failing (a poorly-trained
#                      extractor has meaningless gradients). Neither the
#                      RNG-seed hypothesis (v1.1) nor the hinge-loss
#                      hypothesis (v1.2) was the actual cause — both were
#                      reasonable given the evidence at the time, but
#                      wrong. Fixed by splitting into two separate
#                      no_grad() blocks (encoder-only, quantizer-only)
#                      with hypernet's FiLM computation and the manual
#                      straight-through estimator (emb_mod_st = emb_mod +
#                      (qres.quantized - emb_mod).detach()) outside both,
#                      matching 04's proven-correct structure exactly.
#   v1.4  2026-07-28  v1.3's fix confirmed working (attacker converged to
#                      0.984 bit_acc, healthy trajectory matching 04/07).
#                      Real results showed a well-trained attacker achieves
#                      near-perfect self-consistency evasion against its
#                      own surrogate (own BER down to 0.03-0.08) but this
#                      barely transfers to the defender (BER 0.405->0.369).
#                      However, the defender was embedded using
#                      embed_fragile — Path B alone, leaving Path A's
#                      channels at their natural, unmodulated values —
#                      not the TRUE deployed configuration where both
#                      paths are modulated simultaneously (05's actual
#                      joint embed). This explained why the no-attack
#                      baseline (0.405) differed from E4's originally-
#                      reported joint-config DAC baseline (0.516) for the
#                      same attack. Added embed_joint() (inference-only,
#                      matching 05/06/12's proven pattern — no_grad()
#                      around everything is correct here since nothing in
#                      this call needs gradient, unlike embed_fragile's
#                      training use case) and switched the defender's
#                      embedding to it, now modulating both paths with
#                      independently-keyed defender_hypernet_a and
#                      defender_hypernet_b. Attacker training/PGD logic
#                      unchanged — the attacker still only clones Path B,
#                      since the evasion target is specifically Path B's
#                      fragility signal.
#
# !pip install torch torchaudio encodec descript-audio-codec h5py matplotlib tqdm
# =============================================================================

!pip install -q encodec descript-audio-codec h5py matplotlib tqdm

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not torch.cuda.is_available():
    print("\n[ERROR] GPU not detected!")
    print("This script requires a GPU (attacker training needs it, even though")
    print("the PGD loop itself is cheap).")
    import sys
    sys.exit(1)
print("CUDA available: True. Proceeding...")

torch.backends.cudnn.enabled = False  # same LSTM/eval-mode fix as every training script here

import os
import json
import random
import shutil
import time
import datetime

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

SR = 24000
TARGET_BANDWIDTH_KBPS = 6.0
KEY_BITS = 4
VAL_FRACTION = 0.10
SEED = 20260716  # SAME seed — identical 105-clip split as every other experiment
ATTACKER_INIT_SEED = 999  # deliberately DIFFERENT from SEED — see v1.1 changelog. A
# fixed (not time-based) value so this run is itself reproducible once we know it works.
N_EVAL_CLIPS = 105

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


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


LOCAL_SCRATCH = "/content/scratch"
os.makedirs(LOCAL_SCRATCH, exist_ok=True)
LOCAL_H5 = f"{LOCAL_SCRATCH}/dataset_e0.h5"
LOCAL_E3_CKPT = f"{LOCAL_SCRATCH}/e3_joint_checkpoint.pth"
LOCAL_ATTACKER_CKPT = f"{LOCAL_SCRATCH}/e13_attacker_checkpoint.pth"
LOCAL_RESULTS = f"{LOCAL_SCRATCH}/exp13_anti_forensics_results.json"
LOCAL_FIG = f"{LOCAL_SCRATCH}/fig_13_01_evasion_transfer.png"

# Attacker training (Phase 1) — same recipe as 04/07, proven repeatedly
BATCH_SIZE = 8
N_EPOCHS = 30
CHECKPOINT_EVERY_EPOCHS = 4
LR = 2e-4
LAMBDA_RECON = 0.1
LAMBDA_SURVIVE = 10.0
LAMBDA_BREAK = 5.0
MARGIN = 0.6931471805599453  # ln(2)
FORCE_FRESH_START = True

# PGD evasion (Phase 2) — new to this project
EPSILONS = [0.005, 0.01, 0.02, 0.04]  # Linf bound, waveform normalized to [-1,1]
PGD_STEPS = 40
PGD_STEP_SIZE_FRACTION = 0.1  # step size = epsilon * this fraction, standard PGD convention


# --- Corpus ---------------------------------------------------------------
def load_corpus():
    print(f"[{now()}] Downloading dataset_e0.h5 from PROJECT_DIR...")
    if not copy_from_project("dataset_e0.h5", LOCAL_H5):
        raise SystemExit("dataset_e0.h5 not found in PROJECT_DIR — run 01_e0_dataextract.py first.")
    ids, wavs = [], []
    with h5py.File(LOCAL_H5, "r") as h5f:
        for gid in ("jamendo", "musdb18"):
            for tid in h5f[gid].keys():
                ids.append(tid)
                wavs.append(h5f[gid][tid][:])
    wavs = np.stack(wavs).astype(np.float32)
    idx = np.arange(len(ids))
    rng = np.random.RandomState(SEED)
    rng.shuffle(idx)
    n_val = int(len(idx) * VAL_FRACTION)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    print(f"[{now()}] Split: {len(train_idx)} train (attacker's own training pool), "
          f"{len(val_idx)} val (same held-out clips as every other experiment).")
    return wavs, train_idx, val_idx


# --- Frozen EnCodec + slim self-test ----------------------------------------
def load_codec_and_selftest(sample_wavs_np):
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

    with torch.no_grad():
        raw_emb = model.encoder(torch.from_numpy(sample_wavs_np).float().unsqueeze(1).to(DEVICE))
    d_total = raw_emb.shape[1]
    d_b_start, d_b_end = d_total // 4, 2 * (d_total // 4)  # Path B's channel slice

    raw_emb_leaf = raw_emb.clone().detach().requires_grad_(True)
    with torch.no_grad():
        qres = model.quantizer(raw_emb_leaf, frame_rate, bandwidth)
    quantized_st = raw_emb_leaf + (qres.quantized - raw_emb_leaf).detach()
    wav_probe = model.decoder(quantized_st)
    wav_probe.sum().backward()
    grad_norm = raw_emb_leaf.grad.norm().item() if raw_emb_leaf.grad is not None else 0.0
    print(f"[selftest] gradient probe: {grad_norm:.6f}")
    if grad_norm < 1e-8:
        raise SystemExit("Self-test failed — no gradient through quantizer in this session.")
    print(f"[selftest] PASSED. D_total={d_total}, Path B channels [{d_b_start}:{d_b_end}].")

    return model, d_total, d_b_start, d_b_end, frame_rate, bandwidth


# --- Trainable modules (identical architecture to Path B) --------------------
class HyperNet(nn.Module):
    def __init__(self, key_bits, d_channels, hidden=128):
        super().__init__()
        self.d_channels = d_channels
        self.net = nn.Sequential(
            nn.Linear(key_bits, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2 * d_channels),
        )

    def forward(self, key_bits):
        out = self.net(key_bits)
        gamma_raw, beta_raw = out[:, :self.d_channels], out[:, self.d_channels:]
        gamma = 1.0 + 0.5 * torch.tanh(gamma_raw)
        beta = 0.5 * torch.tanh(beta_raw)
        return gamma, beta


class Extractor(nn.Module):
    """Fully differentiable end-to-end — this is what makes it usable as
    a PGD surrogate; no frozen/non-differentiable components inside it."""
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

    def forward(self, wav):
        h = self.net(wav.unsqueeze(1)).squeeze(-1)
        return self.head(h)


def embed_fragile(model, wav_batch_1ch, hypernet, key_bits, d_start, d_end, frame_rate, bandwidth):
    """CRITICAL: only the frozen encoder and the frozen quantizer go inside
    no_grad — hypernet's FiLM computation must stay OUTSIDE it, or hypernet
    never receives gradient and never trains (this exact bug caused every
    failed attacker-training run so far; see the v1.3 changelog entry for
    the full diagnosis). The manual straight-through estimator is what lets
    gradient flow from the decoder's output back to hypernet's parameters
    despite the quantizer being non-differentiable — decoder must be
    called on emb_mod_st, never on qres.quantized directly."""
    with torch.no_grad():
        raw_emb = model.encoder(wav_batch_1ch)
    gamma, beta = hypernet(key_bits)
    gamma, beta = gamma.unsqueeze(-1), beta.unsqueeze(-1)
    emb_fragile_part = raw_emb[:, d_start:d_end, :] * gamma + beta
    emb_mod = torch.cat([raw_emb[:, :d_start, :], emb_fragile_part, raw_emb[:, d_end:, :]], dim=1)
    with torch.no_grad():
        qres = model.quantizer(emb_mod, frame_rate, bandwidth)
    emb_mod_st = emb_mod + (qres.quantized - emb_mod).detach()
    return model.decoder(emb_mod_st)


def embed_joint(model, wav_batch_1ch, hypernet_a, hypernet_b, key_a, key_b,
                 d_a_start, d_a_end, d_b_start, d_b_end, frame_rate, bandwidth):
    """The DEFENDER's true deployed configuration: both paths modulated
    simultaneously, matching 05_e3_joint_path.py exactly — used ONLY for
    the defender's embedding (inference, already-trained checkpoint, no
    backward() anywhere here), never for the attacker (who only clones
    Path B). Unlike embed_fragile, wrapping everything in no_grad() here
    is correct, not a bug — nothing in this call needs gradient."""
    with torch.no_grad():
        raw_emb = model.encoder(wav_batch_1ch)
        gamma_a, beta_a = hypernet_a(key_a)
        gamma_b, beta_b = hypernet_b(key_b)
        gamma_a, beta_a = gamma_a.unsqueeze(-1), beta_a.unsqueeze(-1)
        gamma_b, beta_b = gamma_b.unsqueeze(-1), beta_b.unsqueeze(-1)
        emb_a = raw_emb[:, d_a_start:d_a_end, :] * gamma_a + beta_a
        emb_b = raw_emb[:, d_b_start:d_b_end, :] * gamma_b + beta_b
        emb_mod = torch.cat([emb_a, emb_b, raw_emb[:, d_b_end:, :]], dim=1)
        qres = model.quantizer(emb_mod, frame_rate, bandwidth)
        return model.decoder(qres.quantized)


def match_length(wav, target_len):
    cur = wav.shape[-1]
    if cur == target_len:
        return wav
    if cur > target_len:
        return wav[..., :target_len]
    return F.pad(wav, (0, target_len - cur))


# --- Augmentations for attacker's own training (identical to 04) -----------
def differentiable_eq(wav, sr=SR, n_fft=1024, hop=256, max_db=6.0, n_bands=6):
    window = torch.hann_window(n_fft, device=wav.device)
    spec = torch.stft(wav, n_fft=n_fft, hop_length=hop, window=window, return_complex=True)
    n_bins = spec.shape[-2]
    band_gains_db = (torch.rand(wav.shape[0], 1, n_bands, device=wav.device) * 2 - 1) * max_db
    gain_curve_db = F.interpolate(band_gains_db, size=n_bins, mode="linear", align_corners=True)
    gain_curve_db = gain_curve_db.squeeze(1).unsqueeze(-1)
    gain_lin = 10 ** (gain_curve_db / 20)
    return torch.istft(spec * gain_lin, n_fft=n_fft, hop_length=hop, window=window, length=wav.shape[-1])


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
    return differentiable_eq(wav)


def compression_proxy(model, wav_1ch, low_bandwidths=(1.5, 3.0)):
    bw = random.choice(low_bandwidths)
    model.set_target_bandwidth(bw)
    with torch.no_grad():
        frames = model.encode(wav_1ch)
        out = model.decode(frames)
    model.set_target_bandwidth(TARGET_BANDWIDTH_KBPS)
    return out


def griffinlim_regen(wav, n_fft=1024, hop=256, n_iter=32):
    window = torch.hann_window(n_fft, device=wav.device)
    spec = torch.stft(wav, n_fft=n_fft, hop_length=hop, window=window, return_complex=True)
    mag = spec.abs()
    return torchaudio.functional.griffinlim(
        mag, window=window, n_fft=n_fft, hop_length=hop, win_length=n_fft,
        power=1.0, n_iter=n_iter, momentum=0.99, length=wav.shape[-1], rand_init=True,
    )


def multi_resolution_stft_loss_batch(ref, deg, resolutions=((1024, 120, 600), (2048, 240, 1200), (512, 50, 240))):
    total = 0.0
    for n_fft, hop, win in resolutions:
        window = torch.hann_window(win, device=ref.device)
        ref_spec = torch.stft(ref, n_fft=n_fft, hop_length=hop, win_length=win, window=window, return_complex=True)
        deg_spec = torch.stft(deg, n_fft=n_fft, hop_length=hop, win_length=win, window=window, return_complex=True)
        ref_mag = torch.clamp(ref_spec.abs(), min=1e-7)
        deg_mag = torch.clamp(deg_spec.abs(), min=1e-7)
        sc = torch.norm(ref_mag - deg_mag, p="fro", dim=(-2, -1)) / torch.norm(ref_mag, p="fro", dim=(-2, -1))
        mag = torch.mean(torch.abs(torch.log(ref_mag) - torch.log(deg_mag)), dim=(-2, -1))
        total = total + (sc + mag).mean()
    return total / len(resolutions)


def load_dac():
    import dac
    dac_model_path = dac.utils.download(model_type="24khz")
    dac_model = dac.DAC.load(dac_model_path).to(DEVICE)
    dac_model.eval()
    return dac_model


def dac_regen(dac_model, wav_1ch):
    with torch.no_grad():
        x = dac_model.preprocess(wav_1ch, SR)
        z, codes, latents, _, _ = dac_model.encode(x)
        y = dac_model.decode(z)
    return match_length(y, wav_1ch.shape[-1])


# --- PHASE 1: train the attacker's own Path B clone (mirrors 04/07) --------
def train_attacker(model, hypernet, extractor, wavs, train_idx, d_start, d_end, frame_rate, bandwidth):
    optimizer = torch.optim.Adam(list(hypernet.parameters()) + list(extractor.parameters()), lr=LR)
    bce_loss_fn = nn.BCEWithLogitsLoss()

    start_epoch = 0
    final_loss, final_bit_acc = None, None
    epoch_history = []
    if FORCE_FRESH_START:
        print(f"[{now()}] FORCE_FRESH_START is True — training attacker clone from scratch.")
    elif copy_from_project("e13_attacker_checkpoint.pth", LOCAL_ATTACKER_CKPT):
        ckpt = torch.load(LOCAL_ATTACKER_CKPT, map_location=DEVICE)
        hypernet.load_state_dict(ckpt["hypernet_state"])
        extractor.load_state_dict(ckpt["extractor_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        epoch_history = ckpt.get("epoch_history", [])
        print(f"[{now()}] Resumed attacker training from epoch {ckpt['epoch']}.")
    else:
        print(f"[{now()}] No attacker checkpoint found, starting fresh.")

    def checkpoint(epoch):
        torch.save({
            "hypernet_state": hypernet.state_dict(),
            "extractor_state": extractor.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "epoch_history": epoch_history,
        }, LOCAL_ATTACKER_CKPT)
        copy_to_project(LOCAL_ATTACKER_CKPT, "e13_attacker_checkpoint.pth")
        print(f"[{now()}] Attacker checkpoint saved (epoch {epoch}) and saved to PROJECT_DIR.")

    n_train = len(train_idx)
    for epoch in range(start_epoch, N_EPOCHS):
        print(f"[{now()}] [ATTACKER TRAINING] Epoch {epoch + 1}/{N_EPOCHS} starting...")
        perm = np.random.permutation(train_idx)
        epoch_losses, epoch_bit_accs, epoch_bce_regen = [], [], []

        for start in range(0, n_train, BATCH_SIZE):
            batch_idx = perm[start:start + BATCH_SIZE]
            if len(batch_idx) < 2:
                continue
            wav_batch = torch.from_numpy(wavs[batch_idx]).float().to(DEVICE)
            key_pm1 = (torch.randint(0, 2, (len(batch_idx), KEY_BITS), device=DEVICE) * 2 - 1).float()
            target = (key_pm1 > 0).float()

            x_wm = embed_fragile(model, wav_batch.unsqueeze(1), hypernet, key_pm1, d_start, d_end, frame_rate, bandwidth)
            x_wm_flat = x_wm.squeeze(1)

            recon = multi_resolution_stft_loss_batch(wav_batch, x_wm_flat)
            x_wm_aug_diff = augment_diff(x_wm_flat)
            with torch.no_grad():
                x_wm_aug_nondiff = compression_proxy(model, x_wm.detach()).squeeze(1)
                x_wm_aug_nondiff = match_length(x_wm_aug_nondiff, x_wm_flat.shape[-1])

            bits_clean = extractor(x_wm_flat)
            bits_aug_diff = extractor(match_length(x_wm_aug_diff, x_wm_flat.shape[-1]))
            bits_aug_nondiff = extractor(x_wm_aug_nondiff)
            bce_survive = (bce_loss_fn(bits_clean, target) + bce_loss_fn(bits_aug_diff, target)
                           + bce_loss_fn(bits_aug_nondiff, target)) / 3.0

            with torch.no_grad():
                x_wm_regen = griffinlim_regen(x_wm_flat.detach())
                x_wm_regen = match_length(x_wm_regen, x_wm_flat.shape[-1])
            bits_regen = extractor(x_wm_regen)
            bce_regen = bce_loss_fn(bits_regen, target)
            l_break = F.relu(MARGIN - bce_regen)

            loss = LAMBDA_RECON * recon + LAMBDA_SURVIVE * bce_survive + LAMBDA_BREAK * l_break

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
        epoch_history.append({"epoch": epoch + 1, "loss": final_loss, "bit_acc": final_bit_acc,
                               "bce_regen": mean_bce_regen})
        print(f"[{now()}] [ATTACKER] Epoch {epoch + 1}/{N_EPOCHS}  loss={final_loss:.4f}  "
              f"bit_acc={final_bit_acc:.4f}  bce_regen={mean_bce_regen:.4f} (target >= {MARGIN:.4f}, "
              f"04's healthy range was ~0.70-0.99 and climbing — watch for this diverging)")

        if (epoch + 1) % CHECKPOINT_EVERY_EPOCHS == 0 or (epoch + 1) == N_EPOCHS:
            checkpoint(epoch)

    return final_loss, final_bit_acc, epoch_history


# --- PHASE 2: PGD self-consistency evasion attack ---------------------------
def pgd_evasion(attacker_extractor, regen_audio, target_logits, epsilon, n_steps=PGD_STEPS):
    """Finds delta (Linf-bounded by epsilon) added to regen_audio such that
    attacker_extractor(regen_audio + delta) matches target_logits — the
    attacker's own extractor's output on the ORIGINAL pre-regeneration
    audio. No knowledge of the true key is used anywhere in this function."""
    step_size = epsilon * PGD_STEP_SIZE_FRACTION
    delta = torch.zeros_like(regen_audio, requires_grad=True)
    target = target_logits.detach()

    for _ in range(n_steps):
        logits = attacker_extractor(regen_audio + delta)
        loss = F.mse_loss(logits, target)
        grad = torch.autograd.grad(loss, delta)[0]
        with torch.no_grad():
            delta -= step_size * grad.sign()  # descend the loss (MSE, not BCE — minimizing directly)
            delta.clamp_(-epsilon, epsilon)
        delta.requires_grad_(True)

    return delta.detach()


def selftest_pgd(attacker_hypernet, attacker_extractor, model, dac_model,
                  sample_wav_np, d_start, d_end, frame_rate, bandwidth):
    print(f"[{now()}] Self-test: does PGD actually reduce the attacker's own loss?")
    wav_t = torch.from_numpy(sample_wav_np).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
    key_pm1 = (torch.randint(0, 2, (1, KEY_BITS), device=DEVICE) * 2 - 1).float()

    with torch.no_grad():
        x_wm = embed_fragile(model, wav_t, attacker_hypernet, key_pm1, d_start, d_end, frame_rate, bandwidth)
        target_logits = attacker_extractor(x_wm.squeeze(1))
        x_regen = dac_regen(dac_model, x_wm)
        x_regen_flat = match_length(x_regen.squeeze(1), x_wm.shape[-1])
        loss_before = F.mse_loss(attacker_extractor(x_regen_flat), target_logits).item()

    epsilon = EPSILONS[len(EPSILONS) // 2]  # a middle epsilon for the self-test
    delta = pgd_evasion(attacker_extractor, x_regen_flat, target_logits, epsilon)
    with torch.no_grad():
        loss_after = F.mse_loss(attacker_extractor(x_regen_flat + delta), target_logits).item()

    print(f"[selftest] MSE to attacker's own target — before PGD: {loss_before:.4f}, "
          f"after PGD ({PGD_STEPS} steps, eps={epsilon}): {loss_after:.4f}")
    if loss_after >= loss_before:
        print(f"[selftest] [FATAL] PGD did not reduce the loss at all — the optimization "
              f"is not working. Every downstream number would be meaningless.")
        raise SystemExit("Self-test failed — PGD loop is not optimizing correctly.")
    print(f"[selftest] PASSED. PGD is reducing the attacker's own loss as expected.")


# --- Main ---------------------------------------------------------------
def main():
    wavs, train_idx, val_idx = load_corpus()
    model, d_total, d_start, d_end, frame_rate, bandwidth = load_codec_and_selftest(
        wavs[train_idx[:BATCH_SIZE]])

    # --- PHASE 1: train the attacker ---
    # DIAGNOSTIC FIX (v1.1): the first production run converged to only
    # ~0.62 bit_acc, far below the ~0.98 this exact recipe reaches in
    # 04/07. Most likely cause: nn.Module weight init draws from the
    # GLOBAL torch RNG state at construction time — even with the same
    # SEED, if this script's self-test consumed a different number of
    # prior random draws than 04/07's did, the attacker's initial weights
    # end up completely different, and small networks are genuinely
    # sensitive to this. Explicit re-seed here decouples attacker weight
    # initialization from whatever random-state history preceded it,
    # testing that hypothesis directly rather than guessing.
    torch.manual_seed(ATTACKER_INIT_SEED)
    print(f"[{now()}] Re-seeded torch RNG to {ATTACKER_INIT_SEED} immediately before attacker "
          f"module construction (diagnostic fix for the prior run's weak convergence — "
          f"see v1.1 changelog).")
    attacker_hypernet = HyperNet(KEY_BITS, d_end - d_start).to(DEVICE)
    attacker_extractor = Extractor(KEY_BITS).to(DEVICE)
    n_params = sum(p.numel() for net in (attacker_hypernet, attacker_extractor) for p in net.parameters())
    print(f"[{now()}] Attacker clone: {n_params:,} trainable parameters "
          f"(gray-box: own key, own weights, same published architecture).")

    final_loss, final_bit_acc, epoch_history = train_attacker(
        model, attacker_hypernet, attacker_extractor, wavs, train_idx, d_start, d_end, frame_rate, bandwidth)
    torch.cuda.empty_cache()
    attacker_hypernet.eval()
    attacker_extractor.eval()

    # --- Load the DEFENDER's real trained system (TRUE joint config, both paths) ---
    print(f"[{now()}] Downloading e3_joint_checkpoint.pth from PROJECT_DIR (REQUIRED — the defender)...")
    if not copy_from_project("e3_joint_checkpoint.pth", LOCAL_E3_CKPT):
        raise SystemExit("e3_joint_checkpoint.pth not found in PROJECT_DIR — run "
                          "05_e3_joint_path.py to completion first.")
    e3_ckpt = torch.load(LOCAL_E3_CKPT, map_location=DEVICE)
    d_a_start, d_a_end = 0, d_total // 4  # Path A's slice — d_start/d_end above is Path B's
    defender_hypernet_a = HyperNet(KEY_BITS, d_a_end - d_a_start).to(DEVICE)
    defender_extractor_a = Extractor(KEY_BITS).to(DEVICE)
    defender_hypernet_a.load_state_dict(e3_ckpt["hypernet_a_state"])
    defender_extractor_a.load_state_dict(e3_ckpt["extractor_a_state"])
    defender_hypernet_a.eval()
    defender_extractor_a.eval()
    defender_hypernet_b = HyperNet(KEY_BITS, d_end - d_start).to(DEVICE)
    defender_extractor_b = Extractor(KEY_BITS).to(DEVICE)
    defender_hypernet_b.load_state_dict(e3_ckpt["hypernet_b_state"])
    defender_extractor_b.load_state_dict(e3_ckpt["extractor_b_state"])
    defender_hypernet_b.eval()
    defender_extractor_b.eval()
    print(f"[{now()}] Loaded defender's E3 Path A AND Path B (true joint config) "
          f"from epoch {e3_ckpt.get('epoch')}.")

    dac_model = load_dac()

    # --- PHASE 2 self-test ---
    selftest_pgd(attacker_hypernet, attacker_extractor, model, dac_model,
                 wavs[val_idx[0]], d_start, d_end, frame_rate, bandwidth)

    # --- PHASE 2 full battery ---
    eval_val_idx = val_idx[:N_EVAL_CLIPS]
    print(f"[{now()}] Running full anti-forensics evasion battery on {len(eval_val_idx)} clips, "
          f"{len(EPSILONS)} epsilon levels...")

    no_attack_wrong, no_attack_total = 0, 0  # defender BER under DAC regen, no evasion attempt at all
    per_epsilon = {str(eps): {"attacker_own_wrong": 0, "attacker_own_total": 0,
                               "defender_wrong": 0, "defender_total": 0,
                               "perturbation_l2_sum": 0.0} for eps in EPSILONS}

    for i in tqdm(range(len(eval_val_idx)), desc="anti-forensics battery"):
        idx = eval_val_idx[i]
        wav_t = torch.from_numpy(wavs[idx]).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
        defender_key_a_pm1 = (torch.randint(0, 2, (1, KEY_BITS), device=DEVICE) * 2 - 1).float()
        defender_key_pm1 = (torch.randint(0, 2, (1, KEY_BITS), device=DEVICE) * 2 - 1).float()
        defender_target = (defender_key_pm1 > 0).float()

        with torch.no_grad():
            x_wm_defender = embed_joint(model, wav_t, defender_hypernet_a, defender_hypernet_b,
                                         defender_key_a_pm1, defender_key_pm1,
                                         d_a_start, d_a_end, d_start, d_end, frame_rate, bandwidth)
            x_regen = dac_regen(dac_model, x_wm_defender)
            x_regen_flat = match_length(x_regen.squeeze(1), x_wm_defender.shape[-1])

            # Baseline: defender's BER under plain regeneration, no evasion at all
            no_attack_logits = defender_extractor_b(x_regen_flat)
            no_attack_pred = (no_attack_logits > 0).float()
            no_attack_wrong += (no_attack_pred != defender_target).sum().item()
            no_attack_total += KEY_BITS

            # Attacker's own target: what THEIR surrogate saw on the pre-regen audio
            attacker_target_logits = attacker_extractor(x_wm_defender.squeeze(1))

        for eps in EPSILONS:
            delta = pgd_evasion(attacker_extractor, x_regen_flat, attacker_target_logits, eps)
            x_evaded = x_regen_flat + delta

            with torch.no_grad():
                # Sanity: did the attack even work against its OWN surrogate?
                attacker_own_logits = attacker_extractor(x_evaded)
                # this is a self-consistency target, not a true key — score as "did it move toward the target"
                attacker_own_pred = (attacker_own_logits > 0).float()
                attacker_own_target_pred = (attacker_target_logits > 0).float()
                per_epsilon[str(eps)]["attacker_own_wrong"] += (
                    attacker_own_pred != attacker_own_target_pred).sum().item()
                per_epsilon[str(eps)]["attacker_own_total"] += KEY_BITS

                # THE ACTUAL QUESTION: does it transfer to the defender's real extractor?
                defender_logits = defender_extractor_b(x_evaded)
                defender_pred = (defender_logits > 0).float()
                per_epsilon[str(eps)]["defender_wrong"] += (defender_pred != defender_target).sum().item()
                per_epsilon[str(eps)]["defender_total"] += KEY_BITS
                per_epsilon[str(eps)]["perturbation_l2_sum"] += delta.norm().item()

    no_attack_ber = float(no_attack_wrong / no_attack_total)
    per_epsilon_results = {}
    for eps in EPSILONS:
        d = per_epsilon[str(eps)]
        per_epsilon_results[str(eps)] = {
            "attacker_own_ber_vs_self_target": float(d["attacker_own_wrong"] / d["attacker_own_total"]),
            "defender_ber": float(d["defender_wrong"] / d["defender_total"]),
            "mean_perturbation_l2": float(d["perturbation_l2_sum"] / len(eval_val_idx)),
        }

    results = {
        "defender_config": "true joint embedding (both Path A and Path B modulated "
                            "simultaneously, matching 05_e3_joint_path.py's deployed "
                            "configuration) — v1.4, was Path-B-alone in v1.0-1.3",
        "attacker_training": {"final_loss": final_loss, "final_bit_acc": final_bit_acc,
                               "epoch_history": epoch_history},
        "no_attack_baseline_defender_ber_regen_only": no_attack_ber,
        "per_epsilon": per_epsilon_results,
        "key_bits": KEY_BITS, "n_eval_clips": len(eval_val_idx),
        "pgd_steps": PGD_STEPS, "epsilons": EPSILONS,
    }
    with open(LOCAL_RESULTS, "w") as f:
        json.dump(results, f, indent=2)
    copy_to_project(LOCAL_RESULTS, "exp13_anti_forensics_results.json")

    print(f"[{now()}] No-attack baseline (defender BER under plain DAC regen): {no_attack_ber:.4f}")
    for eps in EPSILONS:
        r = per_epsilon_results[str(eps)]
        print(f"[{now()}] eps={eps}: attacker-own BER vs. self-target={r['attacker_own_ber_vs_self_target']:.4f}, "
              f"DEFENDER BER (the real question)={r['defender_ber']:.4f}, "
              f"mean perturbation L2={r['mean_perturbation_l2']:.4f}")

    fig, ax = plt.subplots(figsize=(9, 5))
    eps_vals = EPSILONS
    defender_bers = [per_epsilon_results[str(e)]["defender_ber"] for e in eps_vals]
    attacker_bers = [per_epsilon_results[str(e)]["attacker_own_ber_vs_self_target"] for e in eps_vals]
    ax.plot(eps_vals, defender_bers, "o-", color="#C44E52", label="Defender's real BER (the actual question)")
    ax.plot(eps_vals, attacker_bers, "s--", color="#4C72B0", alpha=0.7,
            label="Attacker's own BER vs. self-target (sanity check)")
    ax.axhline(no_attack_ber, color="gray", linestyle=":", label=f"No-evasion baseline ({no_attack_ber:.3f})")
    ax.axhline(0.5, color="black", linestyle="--", alpha=0.3, label="chance")
    ax.set_xlabel("PGD perturbation budget (epsilon, Linf)")
    ax.set_ylabel("BER")
    ax.set_title(f"Adaptive anti-forensics evasion attack — n={len(eval_val_idx)} held-out clips\n"
                 f"does a gray-box, gradient-based evasion attack transfer to the real defender?")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(LOCAL_FIG, dpi=300)
    plt.close(fig)
    copy_to_project(LOCAL_FIG, "fig_13_01_evasion_transfer.png")

    print(f"[{now()}] DONE. e13_attacker_checkpoint.pth, exp13_anti_forensics_results.json, "
          f"fig_13_01_evasion_transfer.png all saved to PROJECT_DIR.")


if __name__ == "__main__":
    main()