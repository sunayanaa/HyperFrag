# =============================================================================
# Program      : 08_e6_ablations.py
# Version      : 1.0
# Description  : Experiment 6 — Ablations (fixed-FiLM arm only).
#
#                The blueprint's E6 has four arms. Two require zero new
#                training — they're already sitting in prior results:
#                  "remove Path B"  = E1 (Path A trained alone)
#                  "remove Path A"  = E2 (Path B trained alone)
#                A third — insertion point (pre- vs. post-quantization) —
#                is already answered more thoroughly than a single clean
#                ablation run could manage: E1's entire v1.0-v2.1
#                debugging arc WAS this comparison (5 independent
#                configurations flat at chance under post-quantization,
#                vs. pre-quantization converging cleanly once the manual
#                straight-through estimator was in place). Citing that
#                history rather than re-spending a training run to
#                reconfirm it. "Residual-VQ stage" insertion is being
#                scoped OUT — it requires intervening between individual
#                RVQ codebook layers, a materially more invasive change to
#                EnCodec's internals than anything done so far, and isn't
#                a good use of remaining budget to attempt blind. Flagged
#                as future work, not silently dropped.
#
#                This script covers the one arm that DOES need new
#                training: does the HyperNetwork's depth/nonlinearity
#                (2 hidden layers, ReLU) actually matter, or would a
#                trivial single LINEAR layer mapping key bits directly to
#                FiLM parameters work just as well? Everything else
#                (Extractor architecture, channel slice [0:D//4] — Path
#                A's role, same as E1 — augmentations, loss, training
#                recipe) is IDENTICAL to E1, so any performance difference
#                is attributable to the HyperNetwork's capacity alone, not
#                a confound from some other change.
#
#                N_EPOCHS=8 for this FIRST run — genuinely untested
#                convergence behavior for this architecture, same
#                discipline as every other new variant introduced so far.
#
# PRE-FLIGHT SELF-TEST: slim — the pre-quantization + manual straight-
# through pipeline itself is proven stable by this point (E1-E5). Only
# re-confirms the gradient probe fresh in this session.
#
# INPUT:
#                  PROJECT_DIR (/content/drive/MyDrive/paper/HyperFrag/):
#                    dataset_e0.h5    — from 01_e0_dataextract.py
#
# STEPS:
#                  Step 1  Download dataset_e0.h5 from PROJECT_DIR
#                  Step 2  Load frozen EnCodec, slim self-test
#                  Step 3  Train FixedFiLM (single linear layer) +
#                          Extractor on Path A's channel slice, using E1's
#                          exact loss/augmentation/hyperparameter recipe
#                  Step 4  Checkpoint every CHECKPOINT_EVERY_EPOCHS epochs
#                          to PROJECT_DIR
#                  Step 5  Evaluate on held-out split with the SAME benign
#                          battery as E1, for direct comparison
#
# OUTPUT FILES : (stored at /content/drive/MyDrive/paper/HyperFrag/)
#                  e6_fixedfilm_checkpoint.pth
#                  exp6_ablation_fixedfilm_results.json
#                      {training: {...}, ber_benign: {...}}
#                  fig_08_01_hypernet_vs_fixedfilm.png
#                      Bar chart: FixedFiLM's benign BER (this experiment)
#                      vs. E1's HyperNetwork benign BER (cited), per
#                      transform — the actual capacity-matters-or-not
#                      comparison this ablation exists to answer
#
# GPU Required : YES
# Dependencies : torch, torchaudio, encodec, pydub, h5py, matplotlib, tqdm
#
# Change Log   :
#   v1.0  2026-07-17  Initial version
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

torch.backends.cudnn.enabled = False  # same LSTM/eval-mode fix as 03-07

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
LOCAL_CKPT = f"{LOCAL_SCRATCH}/e6_fixedfilm_checkpoint.pth"
LOCAL_RESULTS = f"{LOCAL_SCRATCH}/exp6_ablation_fixedfilm_results.json"
LOCAL_FIG = f"{LOCAL_SCRATCH}/fig_08_01_hypernet_vs_fixedfilm.png"

SR = 24000
TARGET_BANDWIDTH_KBPS = 6.0
KEY_BITS = 4
BATCH_SIZE = 8
N_EPOCHS = 30   # FIRST run will be 8 , then 30
CHECKPOINT_EVERY_EPOCHS = 4
LR = 2e-4
LAMBDA_RECON = 0.1
LAMBDA_ROBUST = 10.0
VAL_FRACTION = 0.10
SEED = 20260716
FORCE_FRESH_START = True

# E1's known results, for direct comparison in the output and figure —
# not re-measured here, just carried forward for context.
E1_BER_BENIGN = {
    "compression_kbps": {32: 0.0, 64: 0.0, 128: 0.0, 192: 0.0, 320: 0.0},
    "resample_hz": {16000: 0.0, 22050: 0.0, 32000: 0.0, 44100: 0.0, 48000: 0.0},
    "noise_snr_db": {10: 0.0, 20: 0.0, 30: 0.0, 40: 0.0},
    "gain": {0.7: 0.0, 1.0: 0.0, 1.3: 0.0},
    "eq": {"max_6db": 0.0},
}

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
    val_idx, train_idx = idx[:n_val], idx[n_val:]  # SAME split as E1-E5
    print(f"[{now()}] Split: {len(train_idx)} train, {len(val_idx)} val "
          f"(same held-out clips E1 reported BER=0 on).")
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
    d_start, d_end = 0, d_total // 4  # Path A's channel slice, same as E1

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
    print(f"[selftest] PASSED. D_total={d_total}, channels [{d_start}:{d_end}] (Path A's slice).")

    return model, d_start, d_end, frame_rate, bandwidth


# --- Trainable modules -------------------------------------------------------
class FixedFiLM(nn.Module):
    """THE ABLATION: a single linear layer, no hidden layers, no
    nonlinearity — replaces HyperNetRobust's 2-hidden-layer ReLU MLP.
    Tests whether the HyperNetwork's extra depth/capacity is actually
    doing anything, or whether a trivial linear map from key bits to FiLM
    parameters performs just as well."""
    def __init__(self, key_bits, d_channels):
        super().__init__()
        self.d_channels = d_channels
        self.linear = nn.Linear(key_bits, 2 * d_channels)

    def forward(self, key_bits):
        out = self.linear(key_bits)
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


def embed(model, wav_batch_1ch, film_module, key_bits, d_start, d_end, frame_rate, bandwidth):
    """Identical pipeline to E1's embed() — only the conditioning module
    (film_module) differs: FixedFiLM here instead of HyperNetRobust."""
    with torch.no_grad():
        raw_emb = model.encoder(wav_batch_1ch)
    gamma, beta = film_module(key_bits)
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


# --- Training (mirrors 03_e1_robust_path.py exactly, module swapped) --------
def train(model, film_module, extractor, wavs, train_idx, d_start, d_end, frame_rate, bandwidth):
    optimizer = torch.optim.Adam(
        list(film_module.parameters()) + list(extractor.parameters()), lr=LR)
    bce_loss_fn = nn.BCEWithLogitsLoss()

    start_epoch = 0
    final_loss, final_bit_acc = None, None
    epoch_history = []
    if FORCE_FRESH_START:
        print(f"[{now()}] FORCE_FRESH_START is True — starting from scratch.")
    elif copy_from_project("e6_fixedfilm_checkpoint.pth", LOCAL_CKPT):
        ckpt = torch.load(LOCAL_CKPT, map_location=DEVICE)
        film_module.load_state_dict(ckpt["film_state"])
        extractor.load_state_dict(ckpt["extractor_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        epoch_history = ckpt.get("epoch_history", [])
        print(f"[{now()}] Resumed from checkpoint at epoch {ckpt['epoch']}.")
    else:
        print(f"[{now()}] No checkpoint found, starting fresh.")

    def checkpoint(epoch):
        torch.save({
            "film_state": film_module.state_dict(),
            "extractor_state": extractor.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "epoch_history": epoch_history,
        }, LOCAL_CKPT)
        copy_to_project(LOCAL_CKPT, "e6_fixedfilm_checkpoint.pth")
        print(f"[{now()}] Checkpoint saved (epoch {epoch}) and saved to PROJECT_DIR.")

    n_train = len(train_idx)

    for epoch in range(start_epoch, N_EPOCHS):
        print(f"[{now()}] Epoch {epoch + 1}/{N_EPOCHS} starting...")
        perm = np.random.permutation(train_idx)
        epoch_losses, epoch_bit_accs = [], []

        for start in range(0, n_train, BATCH_SIZE):
            batch_idx = perm[start:start + BATCH_SIZE]
            if len(batch_idx) < 2:
                continue
            wav_batch = torch.from_numpy(wavs[batch_idx]).float().to(DEVICE)
            key_bits_pm1 = (torch.randint(0, 2, (len(batch_idx), KEY_BITS), device=DEVICE) * 2 - 1).float()
            target = (key_bits_pm1 > 0).float()

            x_wm = embed(model, wav_batch.unsqueeze(1), film_module, key_bits_pm1, d_start, d_end, frame_rate, bandwidth)
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
        print(f"[{now()}] Epoch {epoch + 1}/{N_EPOCHS}  loss={final_loss:.4f}  bit_acc={final_bit_acc:.4f}")

        if (epoch + 1) % CHECKPOINT_EVERY_EPOCHS == 0 or (epoch + 1) == N_EPOCHS:
            checkpoint(epoch)

    return final_loss, final_bit_acc, epoch_history


# --- Evaluation (same benign battery as E1, for direct comparison) ---------
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


def evaluate(model, film_module, extractor, wavs, val_idx, d_start, d_end, frame_rate, bandwidth,
             eval_batch_size=BATCH_SIZE):
    print(f"[{now()}] Evaluating on {len(val_idx)} held-out clips (same benign battery as E1)...")
    film_module.eval()
    extractor.eval()
    torch.cuda.empty_cache()

    x_wm_chunks, target_chunks = [], []
    with torch.no_grad():
        for start in range(0, len(val_idx), eval_batch_size):
            batch_idx = val_idx[start:start + eval_batch_size]
            wav_batch = torch.from_numpy(wavs[batch_idx]).float().to(DEVICE)
            key_bits_pm1 = (torch.randint(0, 2, (len(batch_idx), KEY_BITS), device=DEVICE) * 2 - 1).float()
            target = (key_bits_pm1 > 0).float()
            x_wm = embed(model, wav_batch.unsqueeze(1), film_module, key_bits_pm1, d_start, d_end,
                         frame_rate, bandwidth).squeeze(1)
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
                variant = match_length(transform_fn(chunk_t), chunk_t.shape[-1])
                logits = extractor(variant)
            pred = (logits > 0).float().cpu().numpy()
            n_wrong += (pred != chunk_target).sum()
            n_total += chunk_target.size
            del chunk_t, variant, logits
        torch.cuda.empty_cache()
        return float(n_wrong / n_total)

    def ber_mp3(bitrate_kbps):
        n_wrong, n_total = 0, 0
        for start in range(0, len(x_wm_np), eval_batch_size):
            chunk_np = x_wm_np[start:start + eval_batch_size]
            chunk_target = target_np[start:start + eval_batch_size]
            variants = [real_mp3_roundtrip(chunk_np[i], bitrate_kbps) for i in range(len(chunk_np))]
            n = min(v.shape[-1] for v in variants)
            variant_t = torch.from_numpy(np.stack([v[:n] for v in variants])).float().to(DEVICE)
            with torch.no_grad():
                logits = extractor(match_length(variant_t, x_wm_np.shape[-1]))
            pred = (logits > 0).float().cpu().numpy()
            n_wrong += (pred != chunk_target).sum()
            n_total += chunk_target.size
            del variant_t, logits
        torch.cuda.empty_cache()
        return float(n_wrong / n_total)

    ber_benign = {"compression_kbps": {}, "resample_hz": {}, "noise_snr_db": {}, "gain": {}, "eq": {}}
    for kbps in (32, 64, 128, 192, 320):
        print(f"[{now()}]   compression {kbps}kbps...")
        ber_benign["compression_kbps"][kbps] = ber_mp3(kbps)
    for rate in (16000, 22050, 32000, 44100, 48000):
        print(f"[{now()}]   resample {rate}Hz...")
        ber_benign["resample_hz"][rate] = ber_for_transform(
            lambda t, r=rate: torchaudio.functional.resample(
                torchaudio.functional.resample(t, SR, r), r, SR))
    for snr in (10, 20, 30, 40):
        print(f"[{now()}]   noise {snr}dB SNR...")
        def add_noise(t, snr_db=snr):
            sig_power = t.pow(2).mean(dim=-1, keepdim=True)
            noise_power = sig_power / (10 ** (snr_db / 10))
            return t + torch.randn_like(t) * noise_power.sqrt()
        ber_benign["noise_snr_db"][snr] = ber_for_transform(add_noise)
    for g in (0.7, 1.0, 1.3):
        print(f"[{now()}]   gain {g}...")
        ber_benign["gain"][g] = ber_for_transform(lambda t, gg=g: t * gg)
    print(f"[{now()}]   EQ (+-6dB)...")
    ber_benign["eq"]["max_6db"] = ber_for_transform(differentiable_eq)

    film_module.train()
    extractor.train()
    return ber_benign


# --- Main ---------------------------------------------------------------
def main():
    wavs, train_idx, val_idx = load_corpus()
    model, d_start, d_end, frame_rate, bandwidth = load_codec_and_selftest(wavs[train_idx[:BATCH_SIZE]])

    film_module = FixedFiLM(KEY_BITS, d_end - d_start).to(DEVICE)
    extractor = Extractor(KEY_BITS).to(DEVICE)
    n_params_film = sum(p.numel() for p in film_module.parameters())
    n_params_extractor = sum(p.numel() for p in extractor.parameters())
    print(f"[{now()}] FixedFiLM: {n_params_film:,} parameters "
          f"(E1's HyperNetwork had ~{4 * 128 + 128 * 128 + 128 * 2 * 32:,} — this ablation "
          f"is a deliberately much smaller conditioning module).")
    print(f"[{now()}] Extractor: {n_params_extractor:,} parameters (identical to E1's).")

    final_loss, final_bit_acc, epoch_history = train(
        model, film_module, extractor, wavs, train_idx, d_start, d_end, frame_rate, bandwidth)
    torch.cuda.empty_cache()

    ber_benign = evaluate(model, film_module, extractor, wavs, val_idx, d_start, d_end, frame_rate, bandwidth)

    results = {
        "training": {"final_loss": final_loss, "final_bit_acc": final_bit_acc,
                     "epoch_history": epoch_history},
        "ber_benign": ber_benign,
        "ber_benign_e1_hypernet_comparison": E1_BER_BENIGN,
        "target_bandwidth_kbps": TARGET_BANDWIDTH_KBPS, "key_bits": KEY_BITS,
    }
    with open(LOCAL_RESULTS, "w") as f:
        json.dump(results, f, indent=2)
    copy_to_project(LOCAL_RESULTS, "exp6_ablation_fixedfilm_results.json")

    print(f"[{now()}] FixedFiLM BER benign (compare to E1's HyperNetwork BER=0 everywhere):")
    for group, vals in ber_benign.items():
        print(f"    {group}: {vals}")

    fig, ax = plt.subplots(figsize=(9, 5))
    fixedfilm_flat = [v for g in ber_benign.values() for v in g.values()]
    e1_flat = [v for g in E1_BER_BENIGN.values() for v in g.values()]
    x = np.arange(len(fixedfilm_flat))
    ax.plot(x, e1_flat, "o-", color="#4C72B0", label="E1 (HyperNetwork)", alpha=0.7)
    ax.plot(x, fixedfilm_flat, "s-", color="#C44E52", label="E6 (FixedFiLM, this run)", alpha=0.7)
    ax.set_xlabel("benign transform setting (all 18 tested, same order both runs)")
    ax.set_ylabel("BER")
    ax.set_title(f"Does