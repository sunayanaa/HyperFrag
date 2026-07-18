# =============================================================================
# Program      : 05_e3_joint_path.py
# Version      : 1.0
# Description  : Experiment 3 — Joint Training (Path A + Path B together).
#
#                E1 and E2 each produced their OWN independently
#                watermarked audio (Path A only touched channels [0:32],
#                Path B only touched [32:64], the other path's slice left
#                at its natural encoder value). This is the actual proposed
#                system: ONE watermarked audio carrying BOTH signals
#                simultaneously — a single embed_joint() call applies
#                Path A's FiLM to channels [0:32] AND Path B's FiLM to
#                channels [32:64] in the SAME forward pass, through the
#                SAME frozen decoder. The two paths interact through that
#                shared nonlinear decoder — that interaction is exactly
#                what this experiment measures, not a bug to route around.
#
#                Two questions this answers:
#                  1. Does joint training hurt EITHER path's own
#                     performance vs. training it alone (E1: BER=0 benign;
#                     E2: BER=0 benign, BER~0.5 under regen)?
#                  2. What's the perceptual-quality cost of carrying BOTH
#                     signals at once, vs. E0's unwatermarked ceiling
#                     (ViSQOL 4.40, PESQ 2.40, STFT-loss 1.10)? E1/E2 only
#                     reported BER, never PESQ/ViSQOL of the watermarked
#                     audio itself — this is the first place that gap gets
#                     closed, deliberately here rather than duplicating the
#                     measurement in both E1 and E2 separately.
#
#                Reuses E1/E2's architecture, loss terms, and augmentations
#                unchanged — HyperNet/Extractor classes are structurally
#                identical to their E1/E2 counterparts, KEY_BITS=4 for both
#                (proven capacity). Only embed() and the loss combination
#                are new.
#
#                N_EPOCHS=8 for this FIRST run, not 30 — genuinely unknown
#                whether the two paths interfere through the shared
#                decoder, same "check before committing" discipline as
#                E1's capacity search and E2's hinge-loss check.
#
# PRE-FLIGHT SELF-TEST: slim, like E2's — API call signature already known-
# good. Confirms gradient reaches BOTH hypernet_a and hypernet_b through
# the joint embed path (not just one, which a channel-concat bug could
# silently break for one path while leaving the other fine).
#
# INPUT:
#                  PROJECT_DIR (/content/drive/MyDrive/paper/HyperFrag/):
#                    dataset_e0.h5    — from 01_e0_dataextract.py
#
# STEPS:
#                  Step 1  Download dataset_e0.h5 from PROJECT_DIR
#                  Step 2  Load frozen EnCodec, slim self-test (gradient
#                          reaches both hypernets through the joint path)
#                  Step 3  Train HyperNet/Extractor for Path A and Path B
#                          TOGETHER on one shared watermarked signal:
#                          shared recon loss + Path A survive loss + Path
#                          B survive loss + Path B hinge break loss
#                  Step 4  Checkpoint every CHECKPOINT_EVERY_EPOCHS epochs
#                          to PROJECT_DIR
#                  Step 5  Evaluate: Path A benign battery (compare to E1's
#                          BER=0), Path B benign+regen battery (compare to
#                          E2's BER=0/~0.5), and perceptual quality
#                          (PESQ/ViSQOL/STFT-loss) of the joint watermarked
#                          audio vs. E0's ceiling
#
# OUTPUT FILES : (stored at /content/drive/MyDrive/paper/HyperFrag/)
#                  e3_joint_checkpoint.pth  (overwritten each checkpoint)
#                  exp3_joint_results.json
#                      {training: {...}, ber_a_benign: {...},
#                       ber_b_benign: {...}, ber_b_regen: {...},
#                       perceptual: {pesq_wb, visqol_moslqo, mrstft_loss}}
#                  fig_05_01_interference_check.png
#                      3-panel: Path A/B BER under benign, Path B BER under
#                      regen, perceptual quality (E0 ceiling vs. E3 joint)
#
# GPU Required : YES
# Dependencies : torch, torchaudio, encodec, pydub, pesq, visqol-python,
#                h5py, matplotlib, tqdm
#
# Change Log   :
#   v1.0  2026-07-16  Initial version. 
#
# !pip install torch torchaudio encodec pydub pesq visqol-python h5py matplotlib tqdm
# =============================================================================

!pip install -q encodec pydub pesq visqol-python h5py matplotlib tqdm

import torch
import sys

if not torch.cuda.is_available():
    print("\n[ERROR] GPU not detected!")
    print("This script requires a GPU")
    print("Please switch your Colab runtime to a T4 GPU and restart.")
    sys.exit(1)
print("CUDA available: True. Proceeding...")

# Same fix as 03/04 — EnCodec's encoder/decoder contain an LSTM; cuDNN's
# fused kernel can't backward() through a layer that ran forward in eval mode.
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
from pesq import pesq as pesq_fn
from visqol import VisqolApi

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
LOCAL_CKPT = f"{LOCAL_SCRATCH}/e3_joint_checkpoint.pth"
LOCAL_RESULTS = f"{LOCAL_SCRATCH}/exp3_joint_results.json"
LOCAL_FIG = f"{LOCAL_SCRATCH}/fig_05_01_interference_check.png"

SR = 24000
TARGET_BANDWIDTH_KBPS = 6.0
KEY_BITS_A = 4   # Path A (robust) — matches E1's proven capacity
KEY_BITS_B = 4   # Path B (fragile) — matches E2's proven capacity
BATCH_SIZE = 8
N_EPOCHS = 30   # FIRST run — see header for why not 30 yet
CHECKPOINT_EVERY_EPOCHS = 4
LR = 2e-4
LAMBDA_RECON = 0.1
LAMBDA_SURVIVE = 10.0
LAMBDA_BREAK = 5.0
MARGIN = 0.6931471805599453  # ln(2)
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
    wavs = np.stack(wavs).astype(np.float32)
    print(f"[{now()}] Loaded {len(ids)} clips into memory ({wavs.nbytes / 1e9:.2f} GB).")

    idx = np.arange(len(ids))
    rng = np.random.RandomState(SEED)
    rng.shuffle(idx)
    n_val = int(len(idx) * VAL_FRACTION)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    print(f"[{now()}] Split: {len(train_idx)} train, {len(val_idx)} val.")
    return wavs, ids, sources, train_idx, val_idx


# --- Step 2: load frozen EnCodec + slim joint self-test ---------------------
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

    wav_t = torch.from_numpy(sample_wavs_np).float().unsqueeze(1).to(DEVICE)
    with torch.no_grad():
        raw_emb = model.encoder(wav_t)
    d_total = raw_emb.shape[1]
    d_a_start, d_a_end = 0, d_total // 4
    d_b_start, d_b_end = d_total // 4, 2 * (d_total // 4)
    print(f"[{now()}] D_total={d_total}. Path A channels [{d_a_start}:{d_a_end}], "
          f"Path B channels [{d_b_start}:{d_b_end}].")

    print(f"[{now()}] Running joint self-test: does gradient reach BOTH "
          f"hypernet_a and hypernet_b through the shared embed_joint() path?")
    d_a = d_a_end - d_a_start
    d_b = d_b_end - d_b_start
    gamma_a_leaf = torch.ones(sample_wavs_np.shape[0], d_a, 1, device=DEVICE, requires_grad=True)
    beta_a_leaf = torch.zeros(sample_wavs_np.shape[0], d_a, 1, device=DEVICE, requires_grad=True)
    gamma_b_leaf = torch.ones(sample_wavs_np.shape[0], d_b, 1, device=DEVICE, requires_grad=True)
    beta_b_leaf = torch.zeros(sample_wavs_np.shape[0], d_b, 1, device=DEVICE, requires_grad=True)

    emb_a = raw_emb[:, d_a_start:d_a_end, :] * gamma_a_leaf + beta_a_leaf
    emb_b = raw_emb[:, d_b_start:d_b_end, :] * gamma_b_leaf + beta_b_leaf
    emb_mod = torch.cat([emb_a, emb_b, raw_emb[:, d_b_end:, :]], dim=1)
    with torch.no_grad():
        qres = model.quantizer(emb_mod, frame_rate, bandwidth)
    emb_mod_st = emb_mod + (qres.quantized - emb_mod).detach()
    wav_probe = model.decoder(emb_mod_st)
    wav_probe.sum().backward()

    grad_a = gamma_a_leaf.grad.norm().item() if gamma_a_leaf.grad is not None else 0.0
    grad_b = gamma_b_leaf.grad.norm().item() if gamma_b_leaf.grad is not None else 0.0
    print(f"[selftest] gradient norm reaching Path A's gamma: {grad_a:.6f}")
    print(f"[selftest] gradient norm reaching Path B's gamma: {grad_b:.6f}")
    if grad_a < 1e-8 or grad_b < 1e-8:
        print(f"[selftest] [FATAL] gradient missing for at least one path through the "
              f"joint embed — a channel-concat bug could silently break one path while "
              f"leaving the other fine. Do not proceed to training.")
        raise SystemExit("Self-test failed — gradient missing for at least one path.")
    print(f"[selftest] PASSED. Both paths receive gradient through the joint embed. "
          f"Proceeding to training.")

    return model, d_total, d_a_start, d_a_end, d_b_start, d_b_end, frame_rate, bandwidth


# --- Trainable modules (identical architecture to E1/E2) --------------------
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


# --- Joint embedding pipeline ------------------------------------------------
def embed_joint(model, wav_batch_1ch, hypernet_a, hypernet_b, key_bits_a, key_bits_b,
                 d_a_start, d_a_end, d_b_start, d_b_end, frame_rate, bandwidth):
    """Both paths' FiLM applied in ONE forward pass, through the SAME decoder —
    this shared nonlinear decoding is exactly what E3 exists to test."""
    with torch.no_grad():
        raw_emb = model.encoder(wav_batch_1ch)
    gamma_a, beta_a = hypernet_a(key_bits_a)
    gamma_b, beta_b = hypernet_b(key_bits_b)
    gamma_a, beta_a = gamma_a.unsqueeze(-1), beta_a.unsqueeze(-1)
    gamma_b, beta_b = gamma_b.unsqueeze(-1), beta_b.unsqueeze(-1)

    emb_a = raw_emb[:, d_a_start:d_a_end, :] * gamma_a + beta_a
    emb_b = raw_emb[:, d_b_start:d_b_end, :] * gamma_b + beta_b
    emb_mod = torch.cat([emb_a, emb_b, raw_emb[:, d_b_end:, :]], dim=1)

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


# --- Augmentations (identical to E1/E2) -------------------------------------
def differentiable_eq(wav, sr=SR, n_fft=1024, hop=256, max_db=6.0, n_bands=6):
    window = torch.hann_window(n_fft, device=wav.device)
    spec = torch.stft(wav, n_fft=n_fft, hop_length=hop, window=window, return_complex=True)
    n_bins = spec.shape[-2]
    band_gains_db = (torch.rand(wav.shape[0], 1, n_bands, device=wav.device) * 2 - 1) * max_db
    gain_curve_db = F.interpolate(band_gains_db, size=n_bins, mode="linear", align_corners=True)
    gain_curve_db = gain_curve_db.squeeze(1).unsqueeze(-1)
    gain_lin = 10 ** (gain_curve_db / 20)
    spec_eq = spec * gain_lin
    return torch.istft(spec_eq, n_fft=n_fft, hop_length=hop, window=window, length=wav.shape[-1])


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


def regen_proxy_griffinlim(wav, n_fft=1024, hop=256, n_iter=32):
    window = torch.hann_window(n_fft, device=wav.device)
    spec = torch.stft(wav, n_fft=n_fft, hop_length=hop, window=window, return_complex=True)
    mag = spec.abs()
    return torchaudio.functional.griffinlim(
        mag, window=window, n_fft=n_fft, hop_length=hop, win_length=n_fft,
        power=1.0, n_iter=n_iter, momentum=0.99, length=wav.shape[-1], rand_init=True,
    )


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


# --- Perceptual quality metrics (same as 02_e0_codec_sanity.py) -------------
def resample_np(wav_np, orig_sr, new_sr):
    wav_t = torch.from_numpy(wav_np).float().unsqueeze(0)
    return torchaudio.functional.resample(wav_t, orig_sr, new_sr).squeeze(0).numpy()


def compute_pesq_wb(ref_np, deg_np):
    ref_16k = resample_np(ref_np, SR, 16000)
    deg_16k = resample_np(deg_np, SR, 16000)
    return float(pesq_fn(16000, ref_16k, deg_16k, "wb"))


def compute_visqol(api, ref_np, deg_np):
    ref_48k = resample_np(ref_np, SR, 48000)
    deg_48k = resample_np(deg_np, SR, 48000)
    result = api.measure_from_arrays(ref_48k, deg_48k, sample_rate=48000)
    return float(result.moslqo)


# --- Training ----------------------------------------------------------------
def train(model, hypernet_a, extractor_a, hypernet_b, extractor_b, wavs, train_idx,
          d_a_start, d_a_end, d_b_start, d_b_end, frame_rate, bandwidth):
    optimizer = torch.optim.Adam(
        list(hypernet_a.parameters()) + list(extractor_a.parameters())
        + list(hypernet_b.parameters()) + list(extractor_b.parameters()), lr=LR)
    bce_loss_fn = nn.BCEWithLogitsLoss()

    start_epoch = 0
    final_loss = None
    final_bit_acc_a, final_bit_acc_b = None, None
    epoch_history = []
    if FORCE_FRESH_START:
        print(f"[{now()}] FORCE_FRESH_START is True — ignoring any existing "
              f"checkpoint in PROJECT_DIR and starting from scratch.")
    elif copy_from_project("e3_joint_checkpoint.pth", LOCAL_CKPT):
        ckpt = torch.load(LOCAL_CKPT, map_location=DEVICE)
        hypernet_a.load_state_dict(ckpt["hypernet_a_state"])
        extractor_a.load_state_dict(ckpt["extractor_a_state"])
        hypernet_b.load_state_dict(ckpt["hypernet_b_state"])
        extractor_b.load_state_dict(ckpt["extractor_b_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        epoch_history = ckpt.get("epoch_history", [])
        print(f"[{now()}] Resumed from checkpoint at epoch {ckpt['epoch']}, "
              f"continuing from epoch {start_epoch}.")
    else:
        print(f"[{now()}] No checkpoint found, starting fresh.")

    def checkpoint(epoch):
        torch.save({
            "hypernet_a_state": hypernet_a.state_dict(),
            "extractor_a_state": extractor_a.state_dict(),
            "hypernet_b_state": hypernet_b.state_dict(),
            "extractor_b_state": extractor_b.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "epoch_history": epoch_history,
        }, LOCAL_CKPT)
        copy_to_project(LOCAL_CKPT, "e3_joint_checkpoint.pth")
        print(f"[{now()}] Checkpoint saved (epoch {epoch}) and saved to PROJECT_DIR.")

    n_train = len(train_idx)

    for epoch in range(start_epoch, N_EPOCHS):
        print(f"[{now()}] Epoch {epoch + 1}/{N_EPOCHS} starting...")
        perm = np.random.permutation(train_idx)
        epoch_losses, epoch_bit_accs_a, epoch_bit_accs_b, epoch_bce_regen = [], [], [], []

        for start in range(0, n_train, BATCH_SIZE):
            batch_idx = perm[start:start + BATCH_SIZE]
            if len(batch_idx) < 2:
                continue
            wav_batch = torch.from_numpy(wavs[batch_idx]).float().to(DEVICE)
            key_a = (torch.randint(0, 2, (len(batch_idx), KEY_BITS_A), device=DEVICE) * 2 - 1).float()
            key_b = (torch.randint(0, 2, (len(batch_idx), KEY_BITS_B), device=DEVICE) * 2 - 1).float()
            target_a = (key_a > 0).float()
            target_b = (key_b > 0).float()

            x_wm = embed_joint(model, wav_batch.unsqueeze(1), hypernet_a, hypernet_b, key_a, key_b,
                                d_a_start, d_a_end, d_b_start, d_b_end, frame_rate, bandwidth)
            x_wm_flat = x_wm.squeeze(1)

            recon = multi_resolution_stft_loss_batch(wav_batch, x_wm_flat)

            # One shared benign-augmented copy, read by BOTH extractors —
            # reflects a real pipeline where the same delivered audio is
            # checked for both watermarks, and halves augmentation compute.
            x_wm_aug_diff = augment_diff(x_wm_flat)
            with torch.no_grad():
                x_wm_aug_nondiff = compression_proxy(model, x_wm.detach()).squeeze(1)
                x_wm_aug_nondiff = match_length(x_wm_aug_nondiff, x_wm_flat.shape[-1])
            x_wm_aug_diff_m = match_length(x_wm_aug_diff, x_wm_flat.shape[-1])

            bits_a_clean = extractor_a(x_wm_flat)
            bits_a_aug_diff = extractor_a(x_wm_aug_diff_m)
            bits_a_aug_nondiff = extractor_a(x_wm_aug_nondiff)
            bce_a = (bce_loss_fn(bits_a_clean, target_a)
                     + bce_loss_fn(bits_a_aug_diff, target_a)
                     + bce_loss_fn(bits_a_aug_nondiff, target_a)) / 3.0

            bits_b_clean = extractor_b(x_wm_flat)
            bits_b_aug_diff = extractor_b(x_wm_aug_diff_m)
            bits_b_aug_nondiff = extractor_b(x_wm_aug_nondiff)
            bce_b_survive = (bce_loss_fn(bits_b_clean, target_b)
                              + bce_loss_fn(bits_b_aug_diff, target_b)
                              + bce_loss_fn(bits_b_aug_nondiff, target_b)) / 3.0

            with torch.no_grad():
                x_wm_regen = regen_proxy_griffinlim(x_wm_flat.detach())
                x_wm_regen = match_length(x_wm_regen, x_wm_flat.shape[-1])
            bits_b_regen = extractor_b(x_wm_regen)
            bce_b_regen = bce_loss_fn(bits_b_regen, target_b)
            l_break = F.relu(MARGIN - bce_b_regen)

            loss = (LAMBDA_RECON * recon
                    + LAMBDA_SURVIVE * (bce_a + bce_b_survive)
                    + LAMBDA_BREAK * l_break)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bit_acc_a = ((bits_a_clean > 0).float() == target_a).float().mean().item()
            bit_acc_b = ((bits_b_clean > 0).float() == target_b).float().mean().item()
            epoch_losses.append(loss.item())
            epoch_bit_accs_a.append(bit_acc_a)
            epoch_bit_accs_b.append(bit_acc_b)
            epoch_bce_regen.append(bce_b_regen.item())

        final_loss = float(np.mean(epoch_losses)) if epoch_losses else None
        final_bit_acc_a = float(np.mean(epoch_bit_accs_a)) if epoch_bit_accs_a else None
        final_bit_acc_b = float(np.mean(epoch_bit_accs_b)) if epoch_bit_accs_b else None
        mean_bce_regen = float(np.mean(epoch_bce_regen)) if epoch_bce_regen else None
        epoch_history.append({
            "epoch": epoch + 1, "loss": final_loss,
            "bit_acc_a": final_bit_acc_a, "bit_acc_b": final_bit_acc_b,
            "bce_regen_b": mean_bce_regen,
        })
        print(f"[{now()}] Epoch {epoch + 1}/{N_EPOCHS}  loss={final_loss:.4f}  "
              f"bit_acc_A={final_bit_acc_a:.4f}  bit_acc_B={final_bit_acc_b:.4f}  "
              f"bce_regen_B={mean_bce_regen:.4f} (target >= {MARGIN:.4f})")

        if (epoch + 1) % CHECKPOINT_EVERY_EPOCHS == 0 or (epoch + 1) == N_EPOCHS:
            checkpoint(epoch)

    return final_loss, final_bit_acc_a, final_bit_acc_b, epoch_history


# --- Evaluation ---------------------------------------------------------
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


def evaluate(model, hypernet_a, extractor_a, hypernet_b, extractor_b, wavs, val_idx,
             d_a_start, d_a_end, d_b_start, d_b_end, frame_rate, bandwidth,
             eval_batch_size=BATCH_SIZE):
    print(f"[{now()}] Evaluating on {len(val_idx)} held-out clips: Path A benign, "
          f"Path B benign+regen, and perceptual quality of the joint signal...")
    for m in (hypernet_a, extractor_a, hypernet_b, extractor_b):
        m.eval()
    torch.cuda.empty_cache()

    orig_np, x_wm_np, target_a_np, target_b_np = [], [], [], []
    with torch.no_grad():
        for start in range(0, len(val_idx), eval_batch_size):
            batch_idx = val