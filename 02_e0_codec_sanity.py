# =============================================================================
# Program      : 02_e0_codec_sanity.py
# Version      : 1.0
# Description  : Experiment 0 — Codec Sanity Check.
#
#                Establishes the reconstruction-quality CEILING that every
#                later watermarked variant (E1-E6) gets compared against:
#                runs the frozen, pretrained EnCodec (24kHz, no fine-tuning,
#                no watermark) over all 1050 clips from dataset_e0.h5 and
#                measures how much quality the codec itself gives up before
#                any HyperNetwork modulation is added. If this ceiling is
#                already mediocre, nothing built on top of it in E1-E6 can
#                look good either — this is the first thing worth trusting
#                before spending T4 hours on anything else.
#
#                Operates entirely off PROJECT_DIR in Google Drive.
#                No Google Drive mount at all — dataset_e0.h5 already has
#                everything needed in PROJECT_DIR.
#
#                TARGET_BANDWIDTH_KBPS = 6.0 is treated as this project's
#                standard EnCodec operating point from here on; 02-06 should
#                all use the same value unless a program explicitly sweeps it
#                (E6's ablations may want to vary it — see blueprint §6).
#
# INPUT:
#                  PROJECT_DIR (/content/drive/MyDrive/paper/HyperFrag/):
#                    dataset_e0.h5       — from 01_e0_dataextract.py
#                    manifest_e0.json    — from 01_e0_dataextract.py (used
#                                          only to label jamendo vs musdb18
#                                          in the summary breakdown)
#
# STEPS:
#                  Step 1  Download dataset_e0.h5 + manifest_e0.json from PROJECT_DIR
#                  Step 2  Load frozen pretrained EnCodec (24kHz), freeze all
#                          parameters, set target bandwidth, move to GPU
#                  Step 3  For every clip: encode -> decode (no watermark),
#                          compute multi-resolution STFT loss (native 24kHz),
#                          PESQ wideband (resampled to 16kHz), ViSQOL audio
#                          mode (resampled to 48kHz)
#                  Step 4  Checkpoint per-clip results to PROJECT_DIR every
#                          CHECKPOINT_EVERY clips (resumable — skips clips
#                          already scored on a re-run)
#                  Step 5  Aggregate summary stats (overall + per-source),
#                          generate diagnostic figures, save everything to
#                          PROJECT_DIR
#
# OUTPUT FILES : (stored at /content/drive/MyDrive/paper/HyperFrag/)
#                  exp0_codec_sanity.json
#                      {target_bandwidth_kbps, per_clip: [{id, source,
#                       pesq_wb, visqol_moslqo, mrstft_loss}, ...],
#                       summary: {overall, jamendo, musdb18} with
#                       mean/std/n per metric}
#                  fig_02_01_metric_distributions.png
#                      3-panel histogram: PESQ (wb), ViSQOL (MOS-LQO),
#                      multi-res STFT loss, across all 1050 clips
#                  fig_02_02_example_spectrograms.png
#                      Original vs. reconstructed log-magnitude spectrogram
#                      for one example clip, as a visual sanity check —
#                      exploratory/diagnostic, not necessarily a paper figure
#
# GPU Required : YES
# Dependencies : torch, torchaudio, encodec, pesq, visqol-python, h5py,
#                matplotlib, tqdm, numpy
#
# NOTE ON METRIC SAMPLE RATES: PESQ (ITU-T P.862) is only defined at 8kHz
# (narrowband) or 16kHz (wideband) — we use wideband, resampling both
# reference and reconstruction down to 16kHz just for that metric. ViSQOL's
# audio mode requires 48kHz — resampled up separately for that metric only.
# The multi-resolution STFT loss runs at the native 24kHz. Three different
# sample rates for three metrics is intentional, not a bug — each metric's
# resampling is local to its own computation and doesn't touch the others.
#
# NOTE ON ViSQOL PACKAGE: installing the official google/visqol requires a
# bazel build, which isn't practical on Colab. `visqol-python` (PyPI) is an
# independent pure-Python port that installs via plain pip and is validated
# against the C++ conformance suite. Using that instead.
#
# Change Log   :
#   v1.0  2026-07-15  Initial version
#
# !pip install torch torchaudio encodec pesq visqol-python h5py matplotlib tqdm
# =============================================================================

!pip install -q encodec pesq visqol-python h5py matplotlib tqdm

import torch
import sys

if not torch.cuda.is_available():
    print("\n[ERROR] GPU not detected!")
    print("This script requires a GPU")
    print("Please switch your Colab runtime to a T4 GPU and restart.")
    sys.exit(1)
print("CUDA available: True. Proceeding...")

import os
import json
import shutil
import time
import datetime

import numpy as np
import h5py
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


def copy_from_project(remote_filename, local_filepath, verbose=True):
    """Copy a file from PROJECT_DIR (Google Drive) to local path."""
    project_filepath = os.path.join(PROJECT_DIR, remote_filename)
    if not os.path.isfile(project_filepath):
        print(f"[PROJECT_DIR] {remote_filename} not found in PROJECT_DIR")
        return False
    
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            os.makedirs(os.path.dirname(local_filepath), exist_ok=True)
            total_size = os.path.getsize(project_filepath)
            downloaded = [0]
            last_report = [time.time()]
            
            with open(local_filepath, "wb") as f:
                with open(project_filepath, "rb") as src:
                    while True:
                        chunk = src.read(8192)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded[0] += len(chunk)
                        if verbose and time.time() - last_report[0] > 5:
                            mb = downloaded[0] / 1e6
                            pct = 100 * downloaded[0] / total_size
                            print(f"[PROJECT_DIR]   ...{mb:.0f} MB / {total_size / 1e6:.0f} MB ({pct:.0f}%)")
                            last_report[0] = time.time()
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


# --- Config -------------------------------------------------------------
LOCAL_SCRATCH = "/content/scratch"
os.makedirs(LOCAL_SCRATCH, exist_ok=True)
LOCAL_H5 = f"{LOCAL_SCRATCH}/dataset_e0.h5"
LOCAL_MANIFEST = f"{LOCAL_SCRATCH}/manifest_e0.json"
LOCAL_RESULTS = f"{LOCAL_SCRATCH}/exp0_codec_sanity.json"
LOCAL_FIG1 = f"{LOCAL_SCRATCH}/fig_02_01_metric_distributions.png"
LOCAL_FIG2 = f"{LOCAL_SCRATCH}/fig_02_02_example_spectrograms.png"

SR = 24000
TARGET_BANDWIDTH_KBPS = 6.0  # project-wide standard operating point from here on
CHECKPOINT_EVERY = 100


# --- Step 1: pull inputs from PROJECT_DIR ----------------------------------
def fetch_inputs():
    print(f"[{now()}] Downloading dataset_e0.h5 and manifest_e0.json from PROJECT_DIR...")
    if not copy_from_project("dataset_e0.h5", LOCAL_H5):
        raise SystemExit("dataset_e0.h5 not found in PROJECT_DIR — run 01_e0_dataextract.py first.")
    if not copy_from_project("manifest_e0.json", LOCAL_MANIFEST):
        raise SystemExit("manifest_e0.json not found in PROJECT_DIR — run 01_e0_dataextract.py first.")
    with open(LOCAL_MANIFEST) as f:
        manifest = json.load(f)
    source_by_id = {e["id"]: e["source"] for e in manifest}
    return source_by_id


# --- Step 2: load frozen pretrained EnCodec ------------------------------
def load_codec():
    print(f"[{now()}] Loading pretrained EnCodec (24kHz), target bandwidth "
          f"{TARGET_BANDWIDTH_KBPS} kbps...")
    model = EncodecModel.encodec_model_24khz()
    model.set_target_bandwidth(TARGET_BANDWIDTH_KBPS)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model = model.to(DEVICE)
    return model


def load_visqol():
    api = VisqolApi()
    api.create(mode="audio")  # audio mode expects 48kHz
    return api


# --- Step 3: per-clip encode/decode + metrics ----------------------------
def encode_decode(model, wav_np):
    wav_t = torch.from_numpy(wav_np).float().unsqueeze(0).unsqueeze(0).to(DEVICE)  # [1,1,T]
    with torch.no_grad():
        encoded_frames = model.encode(wav_t)
        recon_t = model.decode(encoded_frames)
    recon_np = recon_t.squeeze().cpu().numpy()
    n = min(len(wav_np), len(recon_np))
    return wav_np[:n].astype(np.float32), recon_np[:n].astype(np.float32)


def multi_resolution_stft_loss(ref_np, deg_np,
                                resolutions=((1024, 120, 600), (2048, 240, 1200), (512, 50, 240))):
    ref_t = torch.from_numpy(ref_np).to(DEVICE)
    deg_t = torch.from_numpy(deg_np).to(DEVICE)
    total = 0.0
    for n_fft, hop, win in resolutions:
        window = torch.hann_window(win, device=DEVICE)
        ref_spec = torch.stft(ref_t, n_fft=n_fft, hop_length=hop, win_length=win,
                               window=window, return_complex=True)
        deg_spec = torch.stft(deg_t, n_fft=n_fft, hop_length=hop, win_length=win,
                               window=window, return_complex=True)
        ref_mag = torch.clamp(ref_spec.abs(), min=1e-7)
        deg_mag = torch.clamp(deg_spec.abs(), min=1e-7)
        sc_loss = torch.norm(ref_mag - deg_mag, p="fro") / torch.norm(ref_mag, p="fro")
        mag_loss = torch.nn.functional.l1_loss(torch.log(ref_mag), torch.log(deg_mag))
        total += (sc_loss + mag_loss).item()
    return total / len(resolutions)


def resample_np(wav_np, orig_sr, new_sr):
    wav_t = torch.from_numpy(wav_np).float().unsqueeze(0)
    out = torchaudio.functional.resample(wav_t, orig_sr, new_sr)
    return out.squeeze(0).numpy()


def compute_pesq_wb(ref_np, deg_np):
    ref_16k = resample_np(ref_np, SR, 16000)
    deg_16k = resample_np(deg_np, SR, 16000)
    return float(pesq_fn(16000, ref_16k, deg_16k, "wb"))


def compute_visqol(api, ref_np, deg_np):
    ref_48k = resample_np(ref_np, SR, 48000)
    deg_48k = resample_np(deg_np, SR, 48000)
    result = api.measure_from_arrays(ref_48k, deg_48k, sample_rate=48000)
    return float(result.moslqo)


# --- Step 4/5: main loop, checkpointing, aggregation, figures -----------
def main():
    source_by_id = fetch_inputs()
    model = load_codec()
    visqol_api = load_visqol()

    have_results = copy_from_project("exp0_codec_sanity.json", LOCAL_RESULTS)
    if have_results:
        with open(LOCAL_RESULTS) as f:
            results = json.load(f)
        print(f"[{now()}] Resumed with {len(results['per_clip'])} clips already scored.")
    else:
        results = {"target_bandwidth_kbps": TARGET_BANDWIDTH_KBPS, "per_clip": []}

    done_ids = {r["id"] for r in results["per_clip"]}

    def checkpoint():
        with open(LOCAL_RESULTS, "w") as f:
            json.dump(results, f, indent=2)
        copy_to_project(LOCAL_RESULTS, "exp0_codec_sanity.json")
        print(f"[{now()}] Checkpoint saved to PROJECT_DIR "
              f"({len(results['per_clip'])} clips scored so far).")

    h5f = h5py.File(LOCAL_H5, "r")
    all_items = [(gid, tid) for gid in ("jamendo", "musdb18") for tid in h5f[gid].keys()]
    print(f"[{now()}] {len(all_items)} total clips to score "
          f"({len(all_items) - len(done_ids)} remaining).")

    example_ref, example_deg = None, None  # for the diagnostic spectrogram figure
    n_pesq_fail, n_visqol_fail = 0, 0
    processed_since_checkpoint = 0

    for i, (gid, tid) in enumerate(tqdm(all_items, desc="codec sanity")):
        if tid in done_ids:
            continue
        wav_np = h5f[gid][tid][:]
        ref_np, deg_np = encode_decode(model, wav_np)

        if example_ref is None:
            example_ref, example_deg = ref_np.copy(), deg_np.copy()

        mrstft = multi_resolution_stft_loss(ref_np, deg_np)

        try:
            pesq_wb = compute_pesq_wb(ref_np, deg_np)
        except Exception as e:
            n_pesq_fail += 1
            pesq_wb = None

        try:
            visqol_moslqo = compute_visqol(visqol_api, ref_np, deg_np)
        except Exception as e:
            n_visqol_fail += 1
            visqol_moslqo = None

        results["per_clip"].append({
            "id": tid, "source": source_by_id.get(tid, gid),
            "pesq_wb": pesq_wb, "visqol_moslqo": visqol_moslqo, "mrstft_loss": mrstft,
        })
        processed_since_checkpoint += 1
        if processed_since_checkpoint >= CHECKPOINT_EVERY:
            checkpoint()
            processed_since_checkpoint = 0
        if (i + 1) % 100 == 0:
            print(f"[{now()}] Codec sanity {i + 1}/{len(all_items)} processed")

    checkpoint()
    h5f.close()

    if n_pesq_fail or n_visqol_fail:
        print(f"[{now()}] [WARN] {n_pesq_fail} PESQ failures, {n_visqol_fail} ViSQOL "
              f"failures (stored as null, excluded from summary stats).")

    # --- Aggregate summary stats ---
    def summarize(entries, metric):
        vals = [e[metric] for e in entries if e[metric] is not None]
        if not vals:
            return {"mean": None, "std": None, "n": 0}
        arr = np.array(vals)
        return {"mean": float(arr.mean()), "std": float(arr.std()), "n": len(vals)}

    all_entries = results["per_clip"]
    jam_entries = [e for e in all_entries if e["source"] == "jamendo"]
    mus_entries = [e for e in all_entries if e["source"] == "musdb18"]

    summary = {}
    for label, entries in (("overall", all_entries), ("jamendo", jam_entries), ("musdb18", mus_entries)):
        summary[label] = {
            "pesq_wb": summarize(entries, "pesq_wb"),
            "visqol_moslqo": summarize(entries, "visqol_moslqo"),
            "mrstft_loss": summarize(entries, "mrstft_loss"),
        }
    results["summary"] = summary
    checkpoint()

    print(f"[{now()}] Summary (overall, n={len(all_entries)}):")
    for metric in ("pesq_wb", "visqol_moslqo", "mrstft_loss"):
        s = summary["overall"][metric]
        print(f"    {metric}: mean={s['mean']}, std={s['std']}, n={s['n']}")

    # --- Figures ---
    print(f"[{now()}] Generating diagnostic figures...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    metric_labels = [("pesq_wb", "PESQ (wideband)"),
                      ("visqol_moslqo", "ViSQOL (MOS-LQO)"),
                      ("mrstft_loss", "Multi-res STFT loss")]
    for ax, (metric, label) in zip(axes, metric_labels):
        vals = [e[metric] for e in all_entries if e[metric] is not None]
        ax.hist(vals, bins=30, color="#4C72B0", edgecolor="white")
        ax.set_title(label)
        ax.set_xlabel(label)
        ax.set_ylabel("count")
    fig.suptitle(f"E0 codec sanity — frozen EnCodec @ {TARGET_BANDWIDTH_KBPS} kbps, "
                 f"n={len(all_entries)} clips")
    fig.tight_layout()
    fig.savefig(LOCAL_FIG1, dpi=300)
    plt.close(fig)

    if example_ref is not None:
        fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))
        for ax, sig, title in ((axes2[0], example_ref, "Original"),
                                (axes2[1], example_deg, "EnCodec reconstruction")):
            spec = torch.stft(torch.from_numpy(sig), n_fft=1024, hop_length=256,
                               win_length=1024, window=torch.hann_window(1024),
                               return_complex=True)
            mag_db = 20 * torch.log10(torch.clamp(spec.abs(), min=1e-5))
            im = ax.imshow(mag_db.numpy(), origin="lower", aspect="auto", cmap="magma")
            ax.set_title(title)
            ax.set_xlabel("frame")
            ax.set_ylabel("freq bin")
        fig2.suptitle("Example clip: original vs. reconstructed (log-magnitude STFT)")
        fig2.tight_layout()
        fig2.savefig(LOCAL_FIG2, dpi=300)
        plt.close(fig2)

    copy_to_project(LOCAL_FIG1, "fig_02_01_metric_distributions.png")
    if example_ref is not None:
        copy_to_project(LOCAL_FIG2, "fig_02_02_example_spectrograms.png")

    print(f"[{now()}] DONE. exp0_codec_sanity.json + figures saved to PROJECT_DIR.")


if __name__ == "__main__":
    main()