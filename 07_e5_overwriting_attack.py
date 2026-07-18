# =============================================================================
# Program      : 07_e5_overwriting_attack.py
# Version      : 1.0
# Description  : Experiment 5 — Overwriting-Attack Robustness.
#
#                Per the blueprint's threat model (gray-box attacker): the
#                attacker knows the PUBLISHED architecture — HyperNetwork-
#                conditioned FiLM on channels [0:D//4] of EnCodec's pre-
#                quantization latent, same frozen EnCodec, same insertion
#                point — but does NOT know our trained HyperNetwork_A
#                weights or our secret key K_A. This script simulates that
#                attacker faithfully rather than testing a strawman: Phase
#                1 trains an independent attacker HyperNetwork+Extractor
#                from scratch, using the EXACT same architecture and
#                training recipe as 03_e1_robust_path.py (a competent
#                gray-box attacker who read the paper would just reproduce
#                its methodology for their own key). N_EPOCHS=30 directly
#                for this phase, not a cautious 8-epoch check first — this
#                recipe is already proven to converge cleanly at K=4 bits
#                in E1, no new risk here.
#
#                Phase 2 is the actual attack: the trained attacker model
#                takes ALREADY-WATERMARKED audio (not the original clean
#                track — a realistic attacker only has access to
#                distributed/published audio) and re-modulates the SAME
#                channel slice with their own forged key K_attack, on top
#                of whatever is already there. This is tested against TWO
#                conditions, directly answering the blueprint's "attack
#                success rate with and without Path B active":
#                  (1) audio watermarked by E1's checkpoint (Path A alone)
#                  (2) audio watermarked by E3's checkpoint (Path A + Path
#                      B jointly)
#                For each, we measure whether the ORIGINAL key K_A still
#                recovers (defense held) and whether the ATTACKER's key
#                K_attack now recovers instead (attack succeeded) — these
#                are reported as separate BER curves, not collapsed into a
#                single number, since "disrupted but not hijacked" is a
#                real third outcome. For condition (2) only, we also check
#                Path B's BER before/after the attack — Path B's channels
#                were never the attack's target, so any rise in its BER
#                would be a genuine bonus finding: incidental tamper
#                detection from an attack aimed elsewhere.
#
# PRE-FLIGHT SELF-TEST: slim — the pre-quantization + manual straight-
# through pipeline is proven correct and stable across E1/E2/E3/E4 by this
# point. Only re-confirms the gradient probe fresh in this session before
# spending 30 epochs training the attacker model.
#
# INPUT:
#                  PROJECT_DIR (/content/drive/MyDrive/paper/HyperFrag/):
#                    dataset_e0.h5              — from 01_e0_dataextract.py
#                    e1_robust_checkpoint.pth   — from 03_e1_robust_path.py
#                                                 (REQUIRED for condition 1)
#                    e3_joint_checkpoint.pth    — from 05_e3_joint_path.py
#                                                 (REQUIRED for condition 2)
#
# STEPS:
#                  Step 1  Download dataset_e0.h5 from PROJECT_DIR
#                  Step 2  Load frozen EnCodec, slim self-test
#                  Step 3  PHASE 1: train attacker HyperNetwork+Extractor
#                          (same architecture/recipe as E1, K_ATTACK_BITS=4,
#                          N_EPOCHS=30) — the attacker's OWN training,
#                          using generic training audio, no knowledge of
#                          our checkpoints at all
#                  Step 4  PHASE 2: load E1 + E3 checkpoints from PROJECT_DIR
#                          (REQUIRED), fix one attacker key, attack held-out
#                          clips watermarked by each, score BER vs. K_A
#                          (defense) and vs. K_attack (attack success) for
#                          both conditions, plus Path B collateral BER for
#                          condition 2
#
# OUTPUT FILES : (stored at /content/drive/MyDrive/paper/HyperFrag/)
#                  e5_attacker_checkpoint.pth  (attacker's trained model)
#                  exp5_overwriting_results.json
#                      {attacker_training: {...},
#                       condition1_e1_only: {ber_vs_original, ber_vs_attack},
#                       condition2_e3_joint: {ber_vs_original, ber_vs_attack,
#                                              path_b_ber_before, path_b_ber_after}}
#                  fig_07_01_overwriting_attack.png
#                      Bar chart: BER vs original key / BER vs attacker key,
#                      for both conditions, plus Path B collateral BER
#
# GPU Required : YES
# Dependencies : torch, torchaudio, encodec, h5py, matplotlib, tqdm
#
# Change Log   :
#   v1.0  2026-07-17  Initial version
#
# !pip install torch torchaudio encodec h5py matplotlib tqdm
# =============================================================================

!pip install -q encodec h5py matplotlib tqdm

import torch
import sys

if not torch.cuda.is_available():
    print("\n[ERROR] GPU not detected!")
    print("This script requires a GPU")
    print("Please switch your Colab runtime to a T4 GPU and restart.")
    sys.exit(1)
print("CUDA available: True. Proceeding...")

torch.backends.cudnn.enabled = False  # same LSTM/eval-mode fix as 03/04/05/06

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
LOCAL_ATTACKER_CKPT = f"{LOCAL_SCRATCH}/e5_attacker_checkpoint.pth"
LOCAL_E1_CKPT = f"{LOCAL_SCRATCH}/e1_robust_checkpoint.pth"
LOCAL_E3_CKPT = f"{LOCAL_SCRATCH}/e3_joint_checkpoint.pth"
LOCAL_RESULTS = f"{LOCAL_SCRATCH}/exp5_overwriting_results.json"
LOCAL_FIG = f"{LOCAL_SCRATCH}/fig_07_01_overwriting_attack.png"

SR = 24000
TARGET_BANDWIDTH_KBPS = 6.0
KEY_BITS = 4          # matches Path A's key size — attacker forges a same-size key
BATCH_SIZE = 8
N_EPOCHS = 30          # attacker's own training — proven recipe from E1, no cautious-first-run needed
CHECKPOINT_EVERY_EPOCHS = 4
LR = 2e-4
LAMBDA_RECON = 0.1
LAMBDA_ROBUST = 10.0   # matches E1's final working weights
VAL_FRACTION = 0.10
SEED = 20260716
FORCE_FRESH_START = True   # new attacker model, no checkpoint to conflict with
N_EVAL_CLIPS = 105          # attack evaluation uses the full held-out split — no
# generative model involved here, so unlike E4 there's no speed reason to subsample.
BER_MATCH_THRESHOLD = 0.25   # <=1 of 4 bits wrong counts as "recovered this key"
                              # for the headline survival/success-rate numbers

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


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
    print(f"[{now()}] Loaded {len(ids)} clips into memory ({wavs.nbytes / 1e9:.2f} GB).")

    idx = np.arange(len(ids))
    rng = np.random.RandomState(SEED)
    rng.shuffle(idx)
    n_val = int(len(idx) * VAL_FRACTION)
    val_idx, train_idx = idx[:n_val], idx[n_val:]  # SAME split as E1/E2/E3/E4
    print(f"[{now()}] Split: {len(train_idx)} train (attacker's own training pool), "
          f"{len(val_idx)} val (same held-out clips E1/E3 reported on).")
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

    wav_t = torch.from_numpy(sample_wavs_np).float().unsqueeze(1).to(DEVICE)
    with torch.no_grad():
        raw_emb = model.encoder(wav_t)
    d_total = raw_emb.shape[1]
    d_a_start, d_a_end = 0, d_total // 4  # Path A's channel slice — the attack target

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
    print(f"[selftest] PASSED. D_total={d_total}, attack target channels [{d_a_start}:{d_a_end}].")

    return model, d_total, d_a_start, d_a_end, frame_rate, bandwidth


# --- Trainable modules (identical architecture to Path A, per gray-box knowledge) -
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


def embed(model, wav_batch_1ch, hypernet, key_bits, d_start, d_end, frame_rate, bandwidth):
    """Identical pipeline to E1/E3's embed() — used here for BOTH the
    attacker's own training (embedding into clean audio) AND the actual
    attack (embedding into ALREADY-watermarked audio, when wav_batch_1ch
    is x_wm instead of the original clean track)."""
    with torch.no_grad():
        raw_emb = model.encoder(wav_batch_1ch)
    gamma, beta = hypernet(key_bits)
    gamma, beta = gamma.unsqueeze(-1), beta.unsqueeze(-1)
    emb_target = raw_emb[:, d_start:d_end, :] * gamma + beta
    emb_mod = torch.cat([emb_target, raw_emb[:, d_end:, :]], dim=1)
    with torch.no_grad():
        qres = model.quantizer(emb_mod, frame_rate, bandwidth)
    emb_mod_st = emb_mod + (qres.quantized - emb_mod).detach()
    return model.decoder(emb_mod_st)


def match_length(wav, target_len):
    cur = wav.shape[-1]
    if cur == target_len:
        return wav
    if cur > target_len:
        return wav[..., :target_len]
    return F.pad(wav, (0, target_len - cur))


# --- Augmentations (identical to E1) ----------------------------------------
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


# --- PHASE 1: train the attacker's own model (mirrors 03_e1_robust_path.py) -
def train_attacker(model, hypernet, extractor, wavs, train_idx, d_start, d_end, frame_rate, bandwidth):
    optimizer = torch.optim.Adam(
        list(hypernet.parameters()) + list(extractor.parameters()), lr=LR)
    bce_loss_fn = nn.BCEWithLogitsLoss()

    start_epoch = 0
    final_loss, final_bit_acc = None, None
    epoch_history = []
    if FORCE_FRESH_START:
        print(f"[{now()}] FORCE_FRESH_START is True — training attacker model from scratch.")
    elif copy_from_project("e5_attacker_checkpoint.pth", LOCAL_ATTACKER_CKPT):
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
        copy_to_project(LOCAL_ATTACKER_CKPT, "e5_attacker_checkpoint.pth")
        print(f"[{now()}] Attacker checkpoint saved (epoch {epoch}) and saved to PROJECT_DIR.")

    n_train = len(train_idx)

    for epoch in range(start_epoch, N_EPOCHS):
        print(f"[{now()}] [ATTACKER TRAINING] Epoch {epoch + 1}/{N_EPOCHS} starting...")
        perm = np.random.permutation(train_idx)
        epoch_losses, epoch_bit_accs = [], []

        for start in range(0, n_train, BATCH_SIZE):
            batch_idx = perm[start:start + BATCH_SIZE]
            if len(batch_idx) < 2:
                continue
            wav_batch = torch.from_numpy(wavs[batch_idx]).float().to(DEVICE)
            key_bits_pm1 = (torch.randint(0, 2, (len(batch_idx), KEY_BITS), device=DEVICE) * 2 - 1).float()
            target = (key_bits_pm1 > 0).float()

            x_wm = embed(model, wav_batch.unsqueeze(1), hypernet, key_bits_pm1, d_start, d_end, frame_rate, bandwidth)
            x_wm_flat = x_wm.squeeze(1)

            recon = multi_resolution_stft_loss_batch(wav_batch, x_wm_flat)
            x_wm_aug_diff = augment_diff(x_wm_flat)
            with torch.no_grad():
                x_wm_aug_nondiff = compression_proxy(model, x_wm.detach()).squeeze(1)
                x_wm_aug_nondiff = match_length(x_wm_aug_nondiff, x_wm_flat.shape[-1])

            bits_clean = extractor(x_wm_flat)
            bits_aug_diff = extractor(match_length(x_wm_aug_diff, x_wm_flat.shape[-1]))
            bits_aug_nondiff = extractor(x_wm_aug_nondiff)
            bce = (bce_loss_fn(bits_clean, target) + bce_loss_fn(bits_aug_diff, target)
                   + bce_loss_fn(bits_aug_nondiff, target)) / 3.0
            loss = LAMBDA_RECON * recon + LAMBDA_ROBUST * bce

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            bit_acc = ((bits_clean > 0).float() == target).float().mean().item()
            epoch_losses.append(loss.item())
            epoch_bit_accs.append(bit_acc)

        final_loss = float(np.mean(epoch_losses)) if epoch_losses else None
        final_bit_acc = float(np.mean(epoch_bit_accs)) if epoch_bit_accs else None
        epoch_history.append({"epoch": epoch + 1, "loss": final_loss, "bit_acc": final_bit_acc})
        print(f"[{now()}] [ATTACKER] Epoch {epoch + 1}/{N_EPOCHS}  loss={final_loss:.4f}  bit_acc={final_bit_acc:.4f}")

        if (epoch + 1) % CHECKPOINT_EVERY_EPOCHS == 0 or (epoch + 1) == N_EPOCHS:
            checkpoint(epoch)

    return final_loss, final_bit_acc, epoch_history


# --- PHASE 2: the actual attack ---------------------------------------------
def load_defender_checkpoint(remote_name, local_path, required_keys):
    print(f"[{now()}] Downloading {remote_name} from PROJECT_DIR (REQUIRED)...")
    if not copy_from_project(remote_name, local_path):
        raise SystemExit(f"{remote_name} not found in PROJECT_DIR — this script evaluates "
                          f"attacks against ALREADY-TRAINED checkpoints, it doesn't "
                          f"produce them. Run the corresponding earlier program first.")
    ckpt = torch.load(local_path, map_location=DEVICE)
    for k in required_keys:
        if k not in ckpt:
            raise SystemExit(f"{remote_name} is missing expected key '{k}' — wrong "
                              f"checkpoint file, or an incompatible version.")
    return ckpt


def run_attack(model, attacker_hypernet, k_attack, defender_hypernet_a, defender_extractor_a,
                wavs, val_idx, d_a_start, d_a_end, frame_rate, bandwidth,
                defender_extractor_b=None, d_b_start=None, d_b_end=None,
                defender_hypernet_b=None):
    """Embeds legitimately with the defender's model, then overwrites with
    the attacker's model on the SAME channel slice. Returns BER vs. the
    original key, BER vs. the attacker's forged key, and (if Path B info
    is provided) Path B's BER before/after the attack."""
    n = len(val_idx)
    ber_vs_original_wrong, ber_vs_attack_wrong, total_bits = 0, 0, 0
    path_b_before_wrong, path_b_after_wrong, path_b_total = 0, 0, 0

    for start in range(0, n, BATCH_SIZE):
        batch_idx = val_idx[start:start + BATCH_SIZE]
        if len(batch_idx) < 1:
            continue
        wav_batch = torch.from_numpy(wavs[batch_idx]).float().to(DEVICE)
        key_a_pm1 = (torch.randint(0, 2, (len(batch_idx), KEY_BITS), device=DEVICE) * 2 - 1).float()
        target_a = (key_a_pm1 > 0).float()

        with torch.no_grad():
            if defender_hypernet_b is not None:
                key_b_pm1 = (torch.randint(0, 2, (len(batch_idx), KEY_BITS), device=DEVICE) * 2 - 1).float()
                target_b = (key_b_pm1 > 0).float()
                raw_emb = model.encoder(wav_batch.unsqueeze(1))
                gamma_a, beta_a = defender_hypernet_a(key_a_pm1)
                gamma_b, beta_b = defender_hypernet_b(key_b_pm1)
                gamma_a, beta_a = gamma_a.unsqueeze(-1), beta_a.unsqueeze(-1)
                gamma_b, beta_b = gamma_b.unsqueeze(-1), beta_b.unsqueeze(-1)
                emb_a = raw_emb[:, d_a_start:d_a_end, :] * gamma_a + beta_a
                emb_b = raw_emb[:, d_b_start:d_b_end, :] * gamma_b + beta_b
                emb_mod = torch.cat([emb_a, emb_b, raw_emb[:, d_b_end:, :]], dim=1)
                qres = model.quantizer(emb_mod, frame_rate, bandwidth)
                x_wm = model.decoder(qres.quantized).squeeze(1)
            else:
                x_wm = embed(model, wav_batch.unsqueeze(1), defender_hypernet_a, key_a_pm1,
                              d_a_start, d_a_end, frame_rate, bandwidth).squeeze(1)

            # Path B BEFORE the attack (condition 2 only)
            if defender_extractor_b is not None:
                logits_b_before = defender_extractor_b(x_wm)
                pred_b_before = (logits_b_before > 0).float()
                path_b_before_wrong += (pred_b_before != target_b).sum().item()

            # THE ATTACK: attacker overwrites x_wm (already-watermarked audio)
            # with their own forged key, on the SAME channel slice.
            k_attack_batch = k_attack.repeat(len(batch_idx), 1)
            x_attacked = embed(model, x_wm.unsqueeze(1), attacker_hypernet, k_attack_batch,
                                d_a_start, d_a_end, frame_rate, bandwidth).squeeze(1)

            logits_a_post = defender_extractor_a(x_attacked)
            pred_a_post = (logits_a_post > 0).float()
            ber_vs_original_wrong += (pred_a_post != target_a).sum().item()
            target_attack = (k_attack_batch > 0).float()
            ber_vs_attack_wrong += (pred_a_post != target_attack).sum().item()
            total_bits += target_a.numel()

            if defender_extractor_b is not None:
                logits_b_after = defender_extractor_b(x_attacked)
                pred_b_after = (logits_b_after > 0).float()
                path_b_after_wrong += (pred_b_after != target_b).sum().item()
                path_b_total += target_b.numel()

    result = {
        "ber_vs_original": ber_vs_original_wrong / total_bits,
        "ber_vs_attack": ber_vs_attack_wrong / total_bits,
    }
    if defender_extractor_b is not None:
        result["path_b_ber_before_attack"] = path_b_before_wrong / path_b_total
        result["path_b_ber_after_attack"] = path_b_after_wrong / path_b_total
    return result


# --- Main ---------------------------------------------------------------
def main():
    wavs, train_idx, val_idx = load_corpus()
    model, d_total, d_a_start, d_a_end, frame_rate, bandwidth = load_codec_and_selftest(
        wavs[train_idx[:BATCH_SIZE]])
    d_b_start, d_b_end = d_total // 4, 2 * (d_total // 4)  # Path B's slice, for condition 2 only

    # --- PHASE 1: train the attacker ---
    attacker_hypernet = HyperNet(KEY_BITS, d_a_end - d_a_start).to(DEVICE)
    attacker_extractor = Extractor(KEY_BITS).to(DEVICE)
    n_params = sum(p.numel() for net in (attacker_hypernet, attacker_extractor) for p in net.parameters())
    print(f"[{now()}] Attacker model: {n_params:,} trainable parameters "
          f"(same architecture as Path A, per gray-box knowledge of the published method).")

    final_loss, final_bit_acc, epoch_history = train_attacker(
        model, attacker_hypernet, attacker_extractor, wavs, train_idx,
        d_a_start, d_a_end, frame_rate, bandwidth)
    torch.cuda.empty_cache()
    attacker_hypernet.eval()
    attacker_extractor.eval()

    # --- PHASE 2: the attack itself ---
    print(f"[{now()}] PHASE 2: loading defender checkpoints from PROJECT_DIR for the attack...")
    e1_ckpt = load_defender_checkpoint("e1_robust_checkpoint.pth", LOCAL_E1_CKPT,
                                        ["hypernet_state", "extractor_state"])
    e3_ckpt = load_defender_checkpoint("e3_joint_checkpoint.pth", LOCAL_E3_CKPT,
                                        ["hypernet_a_state", "extractor_a_state",
                                         "hypernet_b_state", "extractor_b_state"])

    e1_hypernet_a = HyperNet(KEY_BITS, d_a_end - d_a_start).to(DEVICE)
    e1_extractor_a = Extractor(KEY_BITS).to(DEVICE)
    e1_hypernet_a.load_state_dict(e1_ckpt["hypernet_state"])
    e1_extractor_a.load_state_dict(e1_ckpt["extractor_state"])
    e1_hypernet_a.eval()
    e1_extractor_a.eval()

    e3_hypernet_a = HyperNet(KEY_BITS, d_a_end - d_a_start).to(DEVICE)
    e3_extractor_a = Extractor(KEY_BITS).to(DEVICE)
    e3_hypernet_b = HyperNet(KEY_BITS, d_b_end - d_b_start).to(DEVICE)
    e3_extractor_b = Extractor(KEY_BITS).to(DEVICE)
    e3_hypernet_a.load_state_dict(e3_ckpt