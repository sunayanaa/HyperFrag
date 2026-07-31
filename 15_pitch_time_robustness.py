# =============================================================================
# Program      : 15_pitch_time_robustness.py
# Version      : 1.0
# Description  : Pitch-Shift / Time-Stretch Robustness Test.
#
#                Motivated directly by DeepMark Benchmark (Kovacevic et
#                al., IEEE Access 2026), which reports that pitch shift
#                and time stretch are unusually effective at breaking
#                watermarks across six evaluated systems, even at subtle
#                magnitudes (their reported sharp sensitivity transition:
#                ~5 cents for pitch shift). Both are plausibly LEGITIMATE
#                audio processing — pitch-correcting a cover, speeding up
#                a podcast, DJ mixing — not tampering. This is a genuine
#                gap: neither was in this project's benign-transform
#                battery (compression, resample, noise, gain, EQ) at any
#                point before now, and if either breaks Path A/B's
#                benign-survival property, that is a real false-positive
#                risk worth disclosing, not a result to bury.
#
#                Eval-only, no training: loads the real, already-trained
#                E3 joint checkpoint (both paths), embeds jointly (matching
#                14's proven embed_joint pattern exactly), applies each
#                severity, scores Path A and Path B BER. Pitch shift via
#                librosa (cents converted to semitones); time stretch via
#                librosa (output length changes naturally with rate --
#                not padded/trimmed back to 10s, since the extractor's
#                global-average-pooling head handles variable length
#                correctly and a real time-stretched clip's length would
#                also change).
#
#                SEVERITIES CHOSEN FOR DIRECT COMPARABILITY TO DEEPMARK:
#                  Pitch shift (cents): 5 (their exact reported sharp-
#                    transition point), 25, 50, 100
#                  Time stretch (rate): 0.9, 1.1, 1.4 (their exact default)
#
# PRE-FLIGHT SELF-TEST: confirms the E3 checkpoint reproduces its own
# known single-hop benign behavior (BER~0 both paths) before running any
# pitch/time condition -- identical pattern to 14's self-test.
#
# INPUT:
#                  PROJECT_DIR (/content/drive/MyDrive/paper/HyperFrag/):
#                    dataset_e0.h5              — from 01_e0_dataextract.py
#                    e3_joint_checkpoint.pth    — from 05_e3_joint_path.py
#                                                 (REQUIRED)
#
# OUTPUT FILES : (stored at /content/drive/MyDrive/paper/HyperFrag/)
#                  exp15_pitch_time_results.json
#                      {pitch_shift: {cents: {path_a_ber, path_b_ber}},
#                       time_stretch: {rate: {path_a_ber, path_b_ber}}}
#                  fig_15_01_pitch_time_robustness.png
#
# GPU Required : YES
# Dependencies : torch, torchaudio, encodec, descript-audio-codec (unused
#                here but kept for consistency), librosa, h5py, matplotlib, tqdm
#
# Change Log   :
#   v1.0  2026-07-30  Initial version
#
# !pip install torch torchaudio encodec librosa h5py matplotlib tqdm
# =============================================================================

!pip install -q encodec librosa h5py matplotlib tqdm

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not torch.cuda.is_available():
    print("\n[ERROR] GPU not detected!")
    import sys
    sys.exit(1)
print("CUDA available: True. Proceeding (eval-only, no training)...")

torch.backends.cudnn.enabled = False  # consistent with every other script here

import os
import json
import shutil
import time
import datetime

import numpy as np
import librosa
import h5py
import torch.nn as nn
import torch.nn.functional as F
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
N_EVAL_CLIPS = 105

# Comparable directly to DeepMark's reported values (see header)
PITCH_SHIFT_CENTS = [5, 25, 50, 100]
TIME_STRETCH_RATES = [0.9, 1.1, 1.4]

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
LOCAL_RESULTS = f"{LOCAL_SCRATCH}/exp15_pitch_time_results.json"
LOCAL_FIG = f"{LOCAL_SCRATCH}/fig_15_01_pitch_time_robustness.png"


# --- Corpus ---------------------------------------------------------------
def load_val_clips():
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
    val_idx = idx[:n_val]
    print(f"[{now()}] {len(val_idx)} held-out clips (identical split to every other experiment).")
    return wavs[val_idx]


# --- Frozen EnCodec ----------------------------------------------------------
def load_codec():
    print(f"[{now()}] Loading frozen EnCodec (24kHz)...")
    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(TARGET_BANDWIDTH_KBPS)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model = model.to(DEVICE)
    frame_rate = getattr(model, "frame_rate", 75)
    bandwidth = getattr(model, "bandwidth", TARGET_BANDWIDTH_KBPS)
    with torch.no_grad():
        d_total = model.encoder(torch.zeros(1, 1, SR, device=DEVICE)).shape[1]
    d_a_start, d_a_end = 0, d_total // 4
    d_b_start, d_b_end = d_total // 4, 2 * (d_total // 4)
    print(f"[{now()}] D_total={d_total}. Path A [{d_a_start}:{d_a_end}], Path B [{d_b_start}:{d_b_end}].")
    return model, d_a_start, d_a_end, d_b_start, d_b_end, frame_rate, bandwidth


# --- Trainable modules (identical architecture to E3) ------------------------
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
            block(1, channels, 4), block(channels, channels * 2, 4),
            block(channels * 2, channels * 4, 4), block(channels * 4, channels * 4, 4),
            block(channels * 4, channels * 4, 4), nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(channels * 4, key_bits)

    def forward(self, wav):
        # Global average pool means this works correctly on variable-length
        # input, which time-stretch naturally produces (see header).
        h = self.net(wav.unsqueeze(1)).squeeze(-1)
        return self.head(h)


def embed_joint(model, wav_1ch, hypernet_a, hypernet_b, key_a, key_b,
                 d_a_start, d_a_end, d_b_start, d_b_end, frame_rate, bandwidth):
    with torch.no_grad():
        raw_emb = model.encoder(wav_1ch)
        gamma_a, beta_a = hypernet_a(key_a)
        gamma_b, beta_b = hypernet_b(key_b)
        gamma_a, beta_a = gamma_a.unsqueeze(-1), beta_a.unsqueeze(-1)
        gamma_b, beta_b = gamma_b.unsqueeze(-1), beta_b.unsqueeze(-1)
        emb_a = raw_emb[:, d_a_start:d_a_end, :] * gamma_a + beta_a
        emb_b = raw_emb[:, d_b_start:d_b_end, :] * gamma_b + beta_b
        emb_mod = torch.cat([emb_a, emb_b, raw_emb[:, d_b_end:, :]], dim=1)
        qres = model.quantizer(emb_mod, frame_rate, bandwidth)
        return model.decoder(qres.quantized)


# --- Pitch shift / time stretch (librosa, numpy in/out) ---------------------
def pitch_shift_np(wav_np, cents):
    semitones = cents / 100.0
    return librosa.effects.pitch_shift(wav_np, sr=SR, n_steps=semitones)


def time_stretch_np(wav_np, rate):
    # Output length changes naturally with rate -- deliberately not padded
    # or trimmed back to the original length; see header for why.
    return librosa.effects.time_stretch(wav_np, rate=rate)


# --- Self-test: E3 checkpoint reproduces known single-hop behavior ---------
def selftest_checkpoint(model, hypernet_a, extractor_a, hypernet_b, extractor_b,
                         d_a_start, d_a_end, d_b_start, d_b_end, frame_rate, bandwidth, sample_wav_np):
    print(f"[{now()}] Self-test: E3 checkpoint reproduces known single-hop clean-signal behavior...")
    wav_t = torch.from_numpy(sample_wav_np).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
    key_a = (torch.randint(0, 2, (1, KEY_BITS), device=DEVICE) * 2 - 1).float()
    key_b = (torch.randint(0, 2, (1, KEY_BITS), device=DEVICE) * 2 - 1).float()
    target_a = (key_a > 0).float()
    target_b = (key_b > 0).float()
    x_wm = embed_joint(model, wav_t, hypernet_a, hypernet_b, key_a, key_b,
                        d_a_start, d_a_end, d_b_start, d_b_end, frame_rate, bandwidth)
    with torch.no_grad():
        pred_a = (extractor_a(x_wm.squeeze(1)) > 0).float()
        pred_b = (extractor_b(x_wm.squeeze(1)) > 0).float()
    n_wrong_a = int((pred_a != target_a).sum().item())
    n_wrong_b = int((pred_b != target_b).sum().item())
    print(f"[selftest] clean-signal bits wrong — Path A: {n_wrong_a}/{KEY_BITS}, "
          f"Path B: {n_wrong_b}/{KEY_BITS} (expect close to 0 for both)")
    if n_wrong_a > KEY_BITS // 2 or n_wrong_b > KEY_BITS // 2:
        raise SystemExit("Self-test failed — checkpoint does not reproduce known clean-signal behavior.")
    print(f"[selftest] PASSED.")


# --- Main ---------------------------------------------------------------
def main():
    val_wavs = load_val_clips()
    model, d_a_start, d_a_end, d_b_start, d_b_end, frame_rate, bandwidth = load_codec()

    print(f"[{now()}] Downloading e3_joint_checkpoint.pth from PROJECT_DIR (REQUIRED)...")
    if not copy_from_project("e3_joint_checkpoint.pth", LOCAL_E3_CKPT):
        raise SystemExit("e3_joint_checkpoint.pth not found in PROJECT_DIR — run "
                          "05_e3_joint_path.py to completion first.")
    ckpt = torch.load(LOCAL_E3_CKPT, map_location=DEVICE)
    hypernet_a = HyperNet(KEY_BITS, d_a_end - d_a_start).to(DEVICE)
    extractor_a = Extractor(KEY_BITS).to(DEVICE)
    hypernet_b = HyperNet(KEY_BITS, d_b_end - d_b_start).to(DEVICE)
    extractor_b = Extractor(KEY_BITS).to(DEVICE)
    hypernet_a.load_state_dict(ckpt["hypernet_a_state"])
    extractor_a.load_state_dict(ckpt["extractor_a_state"])
    hypernet_b.load_state_dict(ckpt["hypernet_b_state"])
    extractor_b.load_state_dict(ckpt["extractor_b_state"])
    for m in (hypernet_a, extractor_a, hypernet_b, extractor_b):
        m.eval()
    print(f"[{now()}] Loaded E3 checkpoint from epoch {ckpt.get('epoch')}.")

    selftest_checkpoint(model, hypernet_a, extractor_a, hypernet_b, extractor_b,
                         d_a_start, d_a_end, d_b_start, d_b_end, frame_rate, bandwidth, val_wavs[0])

    eval_wavs = val_wavs[:N_EVAL_CLIPS]
    print(f"[{now()}] Embedding all {len(eval_wavs)} clips jointly (fresh keys per clip)...")
    embedded_list, keys_a, keys_b = [], [], []
    with torch.no_grad():
        for i in tqdm(range(len(eval_wavs)), desc="joint embed"):
            wav_t = torch.from_numpy(eval_wavs[i]).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
            key_a = (torch.randint(0, 2, (1, KEY_BITS), device=DEVICE) * 2 - 1).float()
            key_b = (torch.randint(0, 2, (1, KEY_BITS), device=DEVICE) * 2 - 1).float()
            x_wm = embed_joint(model, wav_t, hypernet_a, hypernet_b, key_a, key_b,
                                d_a_start, d_a_end, d_b_start, d_b_end, frame_rate, bandwidth)
            embedded_list.append(x_wm.squeeze().cpu().numpy())
            keys_a.append((key_a > 0).float().squeeze().cpu().numpy())
            keys_b.append((key_b > 0).float().squeeze().cpu().numpy())
    keys_a = np.stack(keys_a)
    keys_b = np.stack(keys_b)

    def score_condition(transform_fn):
        n_wrong_a, n_wrong_b, n_total = 0, 0, 0
        for i in range(len(embedded_list)):
            variant_np = transform_fn(embedded_list[i])
            variant_t = torch.from_numpy(variant_np.astype(np.float32)).float().to(DEVICE).unsqueeze(0)
            with torch.no_grad():
                pred_a = (extractor_a(variant_t) > 0).float().squeeze().cpu().numpy()
                pred_b = (extractor_b(variant_t) > 0).float().squeeze().cpu().numpy()
            n_wrong_a += (pred_a != keys_a[i]).sum()
            n_wrong_b += (pred_b != keys_b[i]).sum()
            n_total += KEY_BITS
        return float(n_wrong_a / n_total), float(n_wrong_b / n_total)

    results = {"pitch_shift": {}, "time_stretch": {}}

    for cents in PITCH_SHIFT_CENTS:
        print(f"[{now()}] Pitch shift {cents} cents...")
        ber_a, ber_b = score_condition(lambda w, c=cents: pitch_shift_np(w, c))
        results["pitch_shift"][str(cents)] = {"path_a_ber": ber_a, "path_b_ber": ber_b}
        print(f"[{now()}]   Path A BER={ber_a:.4f}, Path B BER={ber_b:.4f}")

    for rate in TIME_STRETCH_RATES:
        print(f"[{now()}] Time stretch rate={rate}...")
        ber_a, ber_b = score_condition(lambda w, r=rate: time_stretch_np(w, r))
        results["time_stretch"][str(rate)] = {"path_a_ber": ber_a, "path_b_ber": ber_b}
        print(f"[{now()}]   Path A BER={ber_a:.4f}, Path B BER={ber_b:.4f}")

    results["n_clips"] = len(eval_wavs)
    results["key_bits"] = KEY_BITS
    results["note"] = ("Motivated by DeepMark Benchmark (Kovacevic et al., IEEE Access 2026), "
                        "which reports pitch shift and time stretch as unusually effective at "
                        "breaking watermarks even at subtle magnitudes. Severities chosen for "
                        "direct comparability: 5 cents matches their reported sharp sensitivity "
                        "transition; rate=1.4 matches their default time-stretch parameter.")
    with open(LOCAL_RESULTS, "w") as f:
        json.dump(results, f, indent=2)
    copy_to_project(LOCAL_RESULTS, "exp15_pitch_time_results.json")

    print(f"[{now()}] Pitch shift results: {results['pitch_shift']}")
    print(f"[{now()}] Time stretch results: {results['time_stretch']}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    pitch_x = PITCH_SHIFT_CENTS
    axes[0].plot(pitch_x, [results["pitch_shift"][str(c)]["path_a_ber"] for c in pitch_x], "o-", label="Path A")
    axes[0].plot(pitch_x, [results["pitch_shift"][str(c)]["path_b_ber"] for c in pitch_x], "s-", label="Path B")
    axes[0].axhline(0.5, color="gray", linestyle="--", alpha=0.4, label="chance")
    axes[0].set_xlabel("Pitch shift (cents)")
    axes[0].set_ylabel("BER")
    axes[0].set_title("Pitch shift robustness")
    axes[0].legend()

    stretch_x = TIME_STRETCH_RATES
    axes[1].plot(stretch_x, [results["time_stretch"][str(r)]["path_a_ber"] for r in stretch_x], "o-", label="Path A")
    axes[1].plot(stretch_x, [results["time_stretch"][str(r)]["path_b_ber"] for r in stretch_x], "s-", label="Path B")
    axes[1].axhline(0.5, color="gray", linestyle="--", alpha=0.4, label="chance")
    axes[1].set_xlabel("Time stretch rate")
    axes[1].set_ylabel("BER")
    axes[1].set_title("Time stretch robustness")
    axes[1].legend()

    fig.suptitle(f"Pitch-shift / time-stretch robustness, n={len(eval_wavs)} held-out clips "
                 f"(cf. DeepMark Benchmark)")
    fig.tight_layout()
    fig.savefig(LOCAL_FIG, dpi=300)
    plt.close(fig)
    copy_to_project(LOCAL_FIG, "fig_15_01_pitch_time_robustness.png")

    print(f"[{now()}] DONE. exp15_pitch_time_results.json, "
          f"fig_15_01_pitch_time_robustness.png all saved to PROJECT_DIR.")


if __name__ == "__main__":
    main()