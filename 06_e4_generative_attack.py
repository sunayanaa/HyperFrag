# =============================================================================
# Program      : 06_e4_generative_attack.py
# Version      : 1.1
# Description  : Experiment 4 — Held-Out Generative Attack.
#
#                EVALUATION ONLY — no training. Loads E3's trained joint
#                checkpoint (Path A + Path B) and tests it against REAL
#                generative regeneration, replacing E2/E3's Griffin-Lim
#                training-time proxy with two models NEVER seen during
#                training: MusicGen-small (continuation) and DAC (a
#                different neural codec architecture entirely).
#
#                Two things this tests, per the blueprint's original design:
#                  1. Does Path B's fragility (BER -> chance under Griffin-
#                     Lim) GENERALIZE to a real generative model, or was it
#                     an artifact of the specific proxy used in training?
#                  2. Does Path A's robustness extend to surviving AI
#                     regeneration too — i.e., can ownership still be
#                     identified even after the track has been AI-remixed,
#                     while Path B simultaneously proves it WAS remixed?
#                     (Path A was only ever trained/tested against BENIGN
#                     transforms before now — this is its first real test
#                     against anything generative.)
#
#                MusicGen continuation: the first PROMPT_SECONDS of each
#                watermarked clip is kept as-is and fed to MusicGen-small
#                as a continuation prompt; MusicGen generates the remaining
#                duration from scratch. The result is a hybrid clip (real
#                watermarked audio + AI-generated continuation) — a
#                realistic simulation of someone extending/remixing a
#                watermarked track with AI, not a full-clip regeneration.
#
#                DAC resynthesis: the ENTIRE clip is encoded and decoded
#                through DAC's 24kHz model — a full-clip test with a
#                genuinely different codec architecture (not EnCodec-based),
#                complementary to MusicGen's partial-clip continuation test.
#
#                N_EVAL_CLIPS=16 for this FIRST run (not the full 105) —
#                MusicGen inference is far slower than anything used so
#                far in this project; confirm both models work and give
#                sensible results on a small subset before committing to
#                the full held-out split.
#
# PRE-FLIGHT SELF-TEST: neither MusicGen's audio-prompted generate() nor
# DAC's encode/decode API has been used anywhere in this project before.
# Both are checked on a single dummy clip — shapes, sample rates, and a
# basic sanity check that output isn't silence/NaN — before running the
# full evaluation loop.
#
# KNOWN INSTALL RISK (RESOLVED IN v1.1): originally used `audiocraft` for
# MusicGen, which pulls in a fragile dependency chain (av==11.0.0, xformers,
# spacy, demucs, all pinned to a 2024-era torch==2.1.0) — a well-documented,
# common installation failure across many GitHub issues, not specific to
# this environment. Switched to `transformers`' MusicgenForConditionalGeneration
# instead: same model weights (facebook/musicgen-small), same underlying
# architecture, but transformers is near-universally already installed and
# has none of audiocraft's fragile compiled dependencies. transformers'
# audio-prompted generation genuinely supports continuation (confirmed:
# the prompt appears verbatim at the start of the output, followed by real
# generated continuation) — not a downgrade in capability, just a much
# more reliable dependency path to the same thing.
#
# INPUT:
#                  PROJECT_DIR (/content/drive/MyDrive/paper/HyperFrag/):
#                    dataset_e0.h5             — from 01_e0_dataextract.py
#                    e3_joint_checkpoint.pth   — from 05_e3_joint_path.py
#                                                (REQUIRED — this script
#                                                does not train anything)
#
# STEPS:
#                  Step 1  Download dataset_e0.h5 + e3_joint_checkpoint.pth
#                          from PROJECT_DIR
#                  Step 2  Load frozen EnCodec, rebuild HyperNet/Extractor
#                          for both paths, load E3's trained weights
#                  Step 3  Load MusicGen-small (transformers) and DAC
#                          (24kHz); self-test both on a dummy clip before
#                          proceeding
#                  Step 4  For N_EVAL_CLIPS held-out clips: embed jointly
#                          (E3's trained model), then attack with (a)
#                          MusicGen continuation, (b) DAC resynthesis;
#                          score Path A and Path B BER against each
#                  Step 5  Compare against E3's known benign/Griffin-Lim
#                          numbers, save results + figure to PROJECT_DIR
#
# OUTPUT FILES : (stored at /content/drive/MyDrive/paper/HyperFrag/)
#                  exp4_generative_attack_results.json
#                      {musicgen: {ber_a, ber_b, n_clips},
#                       dac: {ber_a, ber_b, n_clips},
#                       comparison: {e3_benign_a, e3_benign_b, e3_griffinlim_b}}
#                  fig_06_01_real_generative_attack.png
#                      Bar chart: Path A/B BER under benign (E3), Griffin-
#                      Lim (E3), MusicGen (real, this experiment), DAC
#                      (real, this experiment) — the actual generalization
#                      test the paper's fragility claim rests on
#
# GPU Required : YES
# Dependencies : torch, torchaudio, encodec, transformers, descript-audio-codec,
#                h5py, matplotlib, tqdm
#
# Change Log   :
#   v1.0  2026-07-16  Initial version (audiocraft-based)
#   v1.1  2026-07-16  audiocraft's install failed on av==11.0.0 (documented,
#                      common issue, not environment-specific — GitHub
#                      issues #476, #463 on facebookresearch/audiocraft
#                      show the same failure independent of platform).
#                      Switched MusicGen loading/inference to transformers'
#                      MusicgenForConditionalGeneration — same model, no
#                      fragile compiled-dependency chain. 
#
# !pip install torch torchaudio encodec transformers descript-audio-codec h5py matplotlib tqdm
# =============================================================================

!pip install -q encodec transformers descript-audio-codec h5py matplotlib tqdm

import torch
import sys

if not torch.cuda.is_available():
    print("\n[ERROR] GPU not detected!")
    print("This script requires a GPU")
    print("Please switch your Colab runtime to a T4 GPU and restart.")
    sys.exit(1)
print("CUDA available: True. Proceeding...")

torch.backends.cudnn.enabled = False  # same LSTM/eval-mode fix as 03/04/05

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
LOCAL_E3_CKPT = f"{LOCAL_SCRATCH}/e3_joint_checkpoint.pth"
LOCAL_RESULTS = f"{LOCAL_SCRATCH}/exp4_generative_attack_results.json"
LOCAL_FIG = f"{LOCAL_SCRATCH}/fig_06_01_real_generative_attack.png"

SR = 24000
TARGET_BANDWIDTH_KBPS = 6.0
KEY_BITS_A = 4
KEY_BITS_B = 4
VAL_FRACTION = 0.10
SEED = 20260716
N_EVAL_CLIPS = 16   # FIRST run — MusicGen is much slower than anything used before
MUSICGEN_SR = 32000
MUSICGEN_FRAME_RATE = 50  # tokens/sec, from MusicGen's architecture — used to convert
# additional-seconds-to-generate into max_new_tokens for transformers' .generate()
PROMPT_SECONDS = 3.0
TOTAL_GEN_SECONDS = 10.0  # matches our clip length; PROMPT_SECONDS of it is real, the rest is generated

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# --- Corpus (val split only — this script never touches train_idx) ---------
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
    val_idx = idx[:n_val]  # SAME split as E1/E2/E3 (same SEED, same shuffle) — these are the
    # exact clips E1/E2/E3 held out and reported benign/Griffin-Lim BER for.
    print(f"[{now()}] {len(val_idx)} held-out val clips available (same split as E1/E2/E3). "
          f"Using first {N_EVAL_CLIPS} of them for this generative-attack check.")
    return wavs[val_idx[:N_EVAL_CLIPS]]


# --- EnCodec + trained E3 model -----------------------------------------
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


def embed_joint(model, wav_batch_1ch, hypernet_a, hypernet_b, key_bits_a, key_bits_b,
                 d_a_start, d_a_end, d_b_start, d_b_end, frame_rate, bandwidth):
    with torch.no_grad():
        raw_emb = model.encoder(wav_batch_1ch)
        gamma_a, beta_a = hypernet_a(key_bits_a)
        gamma_b, beta_b = hypernet_b(key_bits_b)
        gamma_a, beta_a = gamma_a.unsqueeze(-1), beta_a.unsqueeze(-1)
        gamma_b, beta_b = gamma_b.unsqueeze(-1), beta_b.unsqueeze(-1)
        emb_a = raw_emb[:, d_a_start:d_a_end, :] * gamma_a + beta_a
        emb_b = raw_emb[:, d_b_start:d_b_end, :] * gamma_b + beta_b
        emb_mod = torch.cat([emb_a, emb_b, raw_emb[:, d_b_end:, :]], dim=1)
        qres = model.quantizer(emb_mod, frame_rate, bandwidth)
        wav_wm = model.decoder(qres.quantized)
    return wav_wm


def load_e3_model():
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

    print(f"[{now()}] Downloading e3_joint_checkpoint.pth from PROJECT_DIR (REQUIRED)...")
    if not copy_from_project("e3_joint_checkpoint.pth", LOCAL_E3_CKPT):
        raise SystemExit("e3_joint_checkpoint.pth not found in PROJECT_DIR — run "
                          "05_e3_joint_path.py to completion first. This script "
                          "evaluates the trained E3 model, it does not train one.")
    ckpt = torch.load(LOCAL_E3_CKPT, map_location=DEVICE)

    hypernet_a = HyperNet(KEY_BITS_A, d_a_end - d_a_start).to(DEVICE)
    extractor_a = Extractor(KEY_BITS_A).to(DEVICE)
    hypernet_b = HyperNet(KEY_BITS_B, d_b_end - d_b_start).to(DEVICE)
    extractor_b = Extractor(KEY_BITS_B).to(DEVICE)
    hypernet_a.load_state_dict(ckpt["hypernet_a_state"])
    extractor_a.load_state_dict(ckpt["extractor_a_state"])
    hypernet_b.load_state_dict(ckpt["hypernet_b_state"])
    extractor_b.load_state_dict(ckpt["extractor_b_state"])
    for m in (hypernet_a, extractor_a, hypernet_b, extractor_b):
        m.eval()
    print(f"[{now()}] Loaded E3 checkpoint from epoch {ckpt.get('epoch')}.")

    return (model, hypernet_a, extractor_a, hypernet_b, extractor_b,
            d_a_start, d_a_end, d_b_start, d_b_end, frame_rate, bandwidth)


# --- MusicGen + DAC, loaded and self-tested before the real evaluation -----
def load_and_selftest_attackers():
    print(f"[{now()}] Loading MusicGen-small (transformers)...")
    from transformers import AutoProcessor, MusicgenForConditionalGeneration
    mg_processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
    mg_model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small").to(DEVICE)
    mg_model.eval()
    mg_sr = mg_model.config.audio_encoder.sampling_rate
    print(f"[{now()}] MusicGen internal sampling rate: {mg_sr} Hz "
          f"(expected {MUSICGEN_SR} — mismatch here would mean the resample "
          f"step below is using the wrong rate).")

    print(f"[{now()}] Loading DAC (24kHz, descript-audio-codec)...")
    import dac
    dac_model_path = dac.utils.download(model_type="24khz")
    dac_model = dac.DAC.load(dac_model_path)
    dac_model.to(DEVICE)
    dac_model.eval()

    print(f"[{now()}] Self-testing both attackers on a dummy clip before the real run...")
    dummy = (0.1 * torch.randn(1, SR)).numpy().astype(np.float32)[0]

    # MusicGen continuation self-test
    try:
        prompt_np = dummy[: int(PROMPT_SECONDS * SR)]
        prompt_t = torch.from_numpy(prompt_np).float()
        prompt_resampled = torchaudio.functional.resample(prompt_t.unsqueeze(0), SR, mg_sr).squeeze(0).numpy()
        additional_seconds = TOTAL_GEN_SECONDS - PROMPT_SECONDS
        max_new_tokens = int(additional_seconds * MUSICGEN_FRAME_RATE)
        inputs = mg_processor(
            audio=prompt_resampled, sampling_rate=mg_sr, text=["music"],
            padding=True, return_tensors="pt",
        ).to(DEVICE)
        with torch.no_grad():
            mg_out = mg_model.generate(**inputs, do_sample=True, guidance_scale=3,
                                        max_new_tokens=max_new_tokens)
        print(f"[selftest] MusicGen output shape: {tuple(mg_out.shape)} "
              f"(expect (1, 1_or_2_channels, samples))")
        mg_wav = mg_out[0, 0].cpu().numpy()
        print(f"[selftest] MusicGen output range: [{mg_wav.min():.3f}, {mg_wav.max():.3f}]")
        if np.isnan(mg_wav).any() or np.abs(mg_wav).max() < 1e-6:
            print(f"[selftest] [FATAL] MusicGen output is NaN or silent — something is "
                  f"wrong with the continuation call, not just this dummy input.")
            raise SystemExit("Self-test failed — MusicGen produced NaN/silent output.")
        print(f"[selftest] MusicGen PASSED.")
    except SystemExit:
        raise
    except Exception as e:
        print(f"[selftest] [FATAL] MusicGen generate() failed: {e}")
        raise SystemExit("Self-test failed — MusicGen call errored, see diagnostics above.")

    # DAC self-test
    try:
        dummy_t = torch.from_numpy(dummy).float().unsqueeze(0).unsqueeze(0).to(DEVICE)  # [1,1,T]
        with torch.no_grad():
            x = dac_model.preprocess(dummy_t, SR)
            z, codes, latents, _, _ = dac_model.encode(x)
            y = dac_model.decode(z)
        y_np = y.squeeze().cpu().numpy()
        print(f"[selftest] DAC output shape: {tuple(y.shape)}, "
              f"range [{y_np.min():.3f}, {y_np.max():.3f}]")
        if np.isnan(y_np).any() or np.abs(y_np).max() < 1e-6:
            print(f"[selftest] [FATAL] DAC output is NaN or silent.")
            raise SystemExit("Self-test failed — DAC produced NaN/silent output.")
        print(f"[selftest] DAC PASSED.")
    except SystemExit:
        raise
    except Exception as e:
        print(f"[selftest] [FATAL] DAC encode/decode failed: {e}")
        print(f"[selftest]   dir(dac_model) relevant methods: "
              f"{[m for m in dir(dac_model) if not m.startswith('_')]}")
        raise SystemExit("Self-test failed — DAC call errored, see diagnostics above.")

    return mg_processor, mg_model, mg_sr, dac_model


def match_length(wav_np, target_len):
    cur = len(wav_np)
    if cur == target_len:
        return wav_np
    if cur > target_len:
        return wav_np[:target_len]
    return np.pad(wav_np, (0, target_len - cur))


def musicgen_attack(mg_processor, mg_model, mg_sr, wav_wm_np):
    """First PROMPT_SECONDS kept as-is, remaining duration AI-generated as a
    continuation. Returns a 24kHz numpy array matching the original length."""
    prompt_samples = int(PROMPT_SECONDS * SR)
    prompt_np = wav_wm_np[:prompt_samples]
    prompt_t = torch.from_numpy(prompt_np).float().unsqueeze(0)
    prompt_resampled = torchaudio.functional.resample(prompt_t, SR, mg_sr).squeeze(0).numpy()
    additional_seconds = TOTAL_GEN_SECONDS - PROMPT_SECONDS
    max_new_tokens = int(additional_seconds * MUSICGEN_FRAME_RATE)
    inputs = mg_processor(
        audio=prompt_resampled, sampling_rate=mg_sr, text=["music"],
        padding=True, return_tensors="pt",
    ).to(DEVICE)
    with torch.no_grad():
        out = mg_model.generate(**inputs, do_sample=True, guidance_scale=3,
                                 max_new_tokens=max_new_tokens)
    out_wav = out[0, 0].cpu()  # (batch, channels, samples) -> first batch, first channel
    out_24k = torchaudio.functional.resample(out_wav.unsqueeze(0), mg_sr, SR).squeeze(0).numpy()
    return match_length(out_24k, len(wav_wm_np))


def dac_attack(dac_model, wav_wm_np):
    wav_t = torch.from_numpy(wav_wm_np).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        x = dac_model.preprocess(wav_t, SR)
        z, codes, latents, _, _ = dac_model.encode(x)
        y = dac_model.decode(z)
    y_np = y.squeeze().cpu().numpy()
    return match_length(y_np, len(wav_wm_np))


# --- Main ---------------------------------------------------------------
def main():
    val_wavs = load_val_clips()
    (model, hypernet_a, extractor_a, hypernet_b, extractor_b,
     d_a_start, d_a_end, d_b_start, d_b_end, frame_rate, bandwidth) = load_e3_model()
    mg_processor, mg_model, mg_sr, dac_model = load_and_selftest_attackers()

    print(f"[{now()}] Embedding {len(val_wavs)} clips with E3's trained joint model...")
    key_a = (torch.randint(0, 2, (len(val_wavs), KEY_BITS_A), device=DEVICE) * 2 - 1).float()
    key_b = (torch.randint(0, 2, (len(val_wavs), KEY_BITS_B), device=DEVICE) * 2 - 1).float()
    target_a_np = (key_a > 0).float().cpu().numpy()
    target_b_np = (key_b > 0).float().cpu().numpy()
    wav_batch = torch.from_numpy(val_wavs).float().to(DEVICE)
    x_wm = embed_joint(model, wav_batch.unsqueeze(1), hypernet_a, hypernet_b, key_a, key_b,
                        d_a_start, d_a_end, d_b_start, d_b_end, frame_rate, bandwidth)
    x_wm_np = x_wm.squeeze(1).cpu().numpy()

    def score(variant_np_list, target_np):
        variant_t = torch.from_numpy(np.stack(variant_np_list)).float().to(DEVICE)
        with torch.no_grad():
            logits_a = extractor_a(variant_t)
            logits_b = extractor_b(variant_t)
        pred_a = (logits_a > 0).float().cpu().numpy()
        pred_b = (logits_b > 0).float().cpu().numpy()
        ber_a = float((pred_a != target_np).mean())
        return ber_a, pred_b

    print(f"[{now()}] Running MusicGen continuation attack on {len(x_wm_np)} clips "
          f"(this is the slow part)...")
    mg_variants = []
    for i in tqdm(range(len(x_wm_np)), desc="musicgen continuation"):
        mg_variants.append(musicgen_attack(mg_processor, mg_model, mg_sr, x_wm_np[i]))
    mg_t = torch.from_numpy(np.stack(mg_variants)).float().to(DEVICE)
    with torch.no_grad():
        mg_pred_a = (extractor_a(mg_t) > 0).float().cpu().numpy()
        mg_pred_b = (extractor_b(mg_t) > 0).float().cpu().numpy()
    mg_ber_a = float((mg_pred_a != target_a_np).mean())
    mg_ber_b = float((mg_pred_b != target_b_np).mean())

    print(f"[{now()}] Running DAC resynthesis attack on {len(x_wm_np)} clips...")
    dac_variants = [dac_attack(dac_model, x_wm_np[i]) for i in tqdm(range(len(x_wm_np)), desc="dac resynthesis")]
    dac_t = torch.from_numpy(np.stack(dac_variants)).float().to(DEVICE)
    with torch.no_grad():
        dac_pred_a = (extractor_a(dac_t) > 0).float().cpu().numpy()
        dac_pred_b = (extractor_b(dac_t) > 0).float().cpu().numpy()
    dac_ber_a = float((dac_pred_a != target_a_np).mean())
    dac_ber_b = float((dac_pred_b != target_b_np).mean())

    results = {
        "musicgen": {"ber_a": mg_ber_a, "ber_b": mg_ber_b, "n_clips": len(x_wm_np),
                     "prompt_seconds": PROMPT_SECONDS, "total_seconds": TOTAL_GEN_SECONDS},
        "dac": {"ber_a": dac_ber_a, "ber_b": dac_ber_b, "n_clips": len(x_wm_np),
                "model_type": "24khz"},
        "comparison_from_e3": {
            "benign_a_range": [0.012, 0.029], "benign_b_range": [0.093, 0.107],
            "griffinlim_b_range": [0.436, 0.481],
        },
        "key_bits_a": KEY_BITS_A, "key_bits_b": KEY_BITS_B,
    }
    with open(LOCAL_RESULTS, "w") as f:
        json.dump(results, f, indent=2)
    copy_to_project(LOCAL_RESULTS, "exp4_generative_attack_results.json")

    print(f"[{now()}] MusicGen continuation — Path A BER: {mg_ber_a:.4f}, Path B BER: {mg_ber_b:.4f}")
    print(f"[{now()}] DAC resynthesis     — Path A BER: {dac_ber_a:.4f}, Path B BER: {dac_ber_b:.4f}")
    print(f"[{now()}] For comparison, E3 (joint, n=105): benign A~0.012-0.029, "
          f"benign B~0.093-0.107, Griffin-Lim B~0.436-0.481")

    fig, ax = plt.subplots(figsize=(9, 5))
    labels = ["Benign\n(E3, n=105)", "Griffin-Lim\n(E3, n=105)", "MusicGen\n(real, n={})".format(len(x_wm_np)),
              "DAC\n(real, n={})".format(len(x_wm_np))]
    path_a_vals = [np.mean([0.012, 0.029]), None, mg_ber_a, dac_ber_a]
    path_b_vals = [np.mean([0.093, 0.107]), np.mean([0.436, 0.481]), mg_ber_b, dac_ber_b]
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width / 2, [v if v is not None else 0 for v in path_a_vals], width, label="Path A (robust)")
    ax.bar(x + width / 2, path_b_vals, width, label="Path B (fragile)")
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.6, label="chance")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("BER")
    ax.set_title(f"E4 — real generative attack vs. training-time proxies "
                 f"(MusicGen n={len(x_wm_np)}, DAC n={len(x_wm_np)})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(LOCAL_FIG, dpi=300)
    plt.close(fig)
    copy_to_project(LOCAL_FIG, "fig_06_01_real_generative_attack.png")

    print(f"[{now()}] DONE. exp4_generative_attack_results.json, "
          f"fig_06_01_real_generative_attack.png saved to PROJECT_DIR.")


if __name__ == "__main__":
    main()