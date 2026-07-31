# =============================================================================
# Program      : 16_baseline_aware.py
# Version      : 1.4
# Description  : Quantitative baseline comparison — AWARE (Audio
#                Watermarking via Adversarial Resistance to Edits;
#                Pavlovic et al., 2025, arXiv:2510.17512).
#
#                Fourth baseline added to Section VI's comparison, chosen
#                specifically because AWARE's own paper reports strong
#                robustness to pitch shift (BER=0.014 at 10 cents) and
#                time-scale modification (BER=0.016 at +-35%) — exactly
#                the desynchronizing transforms that broke OUR system in
#                15_pitch_time_robustness.py (BER 0.405-0.469 for pitch
#                shift, indistinguishable from regeneration). This script
#                tests AWARE against the identical benign/regen battery
#                used for every other baseline, AND against 15's exact
#                pitch-shift/time-stretch severities (5/25/50/100 cents;
#                0.9/1.1/1.4 rates) — not just AWARE's own single reported
#                points — for a direct, apples-to-apples comparison to our
#                own collapse.
#
#                One specific, genuine weakness AWARE's own paper reports:
#                on EnCodec-based neural compression, AWARE UNDERPERFORMS
#                AudioSeal (0.179 vs 0.091), attributed to AudioSeal's
#                explicit training exposure to that distortion. Worth
#                watching specifically in the DAC-regen results below,
#                since DAC is a different but related neural codec.
#
#                TWO DEPARTURES FROM 09-12'S ESTABLISHED PATTERN, both
#                deliberate: (1) embedding IS checkpointed here, unlike
#                every other baseline, because AWARE's embedding is a
#                500-iteration per-clip adversarial optimization
#                (Algorithm 1 in the paper), not a single forward pass —
#                the "re-embed fresh each run, it's cheap" justification
#                used elsewhere does not hold. (2) detection-rate
#                diagnostics are built in from the start, not patched on
#                after the fact, since detect_watermark natively returns
#                a confidence score with a documented tau=0.5 threshold —
#                AudioSeal/WavMark/SilentCipher all required a reactive
#                v1.2+ fix to add this; no reason to repeat that here.
#
# PRE-FLIGHT SELF-TEST: embed + detect on one clip, no attack. Also
# confirms embed_watermark preserves input length (expected given
# iSTFT reuses the original phase per Algorithm 1), since nothing
# downstream assumes this without checking.
#
# INPUT:
#                  PROJECT_DIR (/content/drive/MyDrive/paper/HyperFrag/):
#                    dataset_e0.h5              — from 01_e0_dataextract.py
#
# CAPACITY NOTE (CORRECTED, v1.4): AWARE's paper (Table 1/2) states "16
# bps" for its comparison against AudioSeal/WavMark, but the actual
# distributed load(name="AWARE") checkpoint requires 20 bits
# (model.detection_net.output_length) -- confirmed by a runtime
# ValueError, not assumed. Used here at 20 bits accordingly: still
# harder than our own K=4, not an advantage chosen either way, just not
# the exact number the paper's prose described for its own internal
# comparison run. Uses load(name="AWARE"), the default full-sequence
# profile — not "AWARE(20bps)", the higher-capacity segment-based
# variant (a different, unrelated 20 that happens to coincide numerically).
#
# OUTPUT FILES : (stored at /content/drive/MyDrive/paper/HyperFrag/)
#                  exp_baseline_aware_results.json
#                  fig_16_01_aware_vs_ours.png
#
# GPU Required : YES
# Dependencies : torch, torchaudio, aware (pip install -e from the repo,
#                see below), pydub, descript-audio-codec, transformers,
#                librosa, h5py, matplotlib, tqdm
#
# Change Log   :
#   v1.0  2026-07-30  Initial version
#   v1.1  2026-07-30  Fixed a numpy binary-compatibility crash on first
#                      import (ValueError: numpy.dtype size changed...,
#                      triggered inside numpy.random._pickle when
#                      importing librosa). Same class of bug as
#                      11_baseline_silentcipher.py's v1.2 fix: the aware
#                      repo's `pip install -e .` ran first and likely
#                      pulled a different numpy version than what was
#                      already active, leaving an inconsistent binary
#                      state. Fixed by reordering (established packages
#                      install first) and isolating the aware repo
#                      install with --no-deps. IMPORTANT: as with 11's
#                      fix, this code change alone does not repair an
#                      already-broken session — the mismatched numpy is
#                      already loaded into memory before the error
#                      surfaces, so a genuine Colab runtime restart is
#                      required before re-running, not just re-executing
#                      the cell.
#   v1.2  2026-07-31  Fixed ModuleNotFoundError: No module named 'aware'
#                      despite pip confirming a successful install
#                      (verified via pip show — real editable install,
#                      Location: /content/aware_repo/src, no errors).
#                      Root cause, confirmed by diagnostic output, NOT
#                      guessed: the repo is a src/-layout project, and
#                      its editable install registers the path via a
#                      .pth file that Python's site module normally only
#                      processes at interpreter startup — since the
#                      install ran mid-session via ! shell magic, the
#                      already-running kernel's sys.path never picked it
#                      up. Unlike v1.1's numpy fix, this is a pure Python
#                      path issue, not a binary/ABI mismatch — fixed by
#                      explicitly inserting the src/ path into sys.path
#                      before importing aware. NO RUNTIME RESTART NEEDED
#                      for this specific fix.
#   v1.3  2026-07-31  Fixed ModuleNotFoundError: No module named
#                      'webrtcvad', hit on the next run. This is exactly
#                      the risk flagged (not just guessed after the
#                      fact) when v1.1 isolated the aware repo install
#                      with --no-deps: pip show's Requires: line already
#                      listed pesq, pydantic, pystoi, and webrtcvad as
#                      AWARE's own real dependencies, none obviously
#                      covered by librosa/pydub/etc's transitive
#                      installs. Added all four now rather than fixing
#                      them one at a time across four more reruns. No
#                      restart needed — same as v1.2, a missing-package
#                      issue, not a binary/ABI mismatch.
#   v1.4  2026-07-31  Fixed two issues from the next run. (1) Missing
#                      'resampy' dependency (added to install line,
#                      same class as v1.3's fix). (2) ValueError:
#                      Invalid watermark length. Expected 20, got 16 —
#                      confirmed by the error itself that AWARE's paper
#                      ("16 bps" for its own comparison table) and the
#                      actual distributed load(name="AWARE") checkpoint
#                      (model.detection_net.output_length=20) disagree.
#                      The repo's own README example embeds 20 bits even
#                      for this same default profile — that was the
#                      correct signal and should have been trusted over
#                      the paper's prose when the two conflicted.
#                      N_MSG_BITS corrected to 20 throughout, including
#                      all previously-hardcoded "16-bit" references in
#                      logging, the results note, and the figure label.
#
# !pip install torch torchaudio pydub descript-audio-codec transformers librosa h5py matplotlib tqdm pesq pydantic pystoi webrtcvad resampy
# !git clone https://github.com/deepmarkpy/aware.git && cd aware && pip install --no-deps -e .
# =============================================================================

!pip install -q pydub descript-audio-codec transformers librosa h5py matplotlib tqdm pesq pydantic pystoi webrtcvad resampy
!git clone -q https://github.com/deepmarkpy/aware.git /content/aware_repo 2>/dev/null || echo "repo already cloned"
!cd /content/aware_repo && pip install -q --no-deps -e .

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not torch.cuda.is_available():
    print("\n[ERROR] GPU not detected!")
    import sys
    sys.exit(1)
print(f"Using device: {DEVICE} (embedding is a 500-iteration per-clip optimization — "
      f"expect this to be the slow part, unlike every other baseline here).")

import os
import json
import random
import shutil
import time
import datetime
import tempfile

import numpy as np
import h5py
import torch.nn.functional as F
import torchaudio
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

SR = 24000            # our corpus's native rate
AWARE_SR = 16000       # AWARE's REQUIRED operating rate — not a choice
N_MSG_BITS = 20        # CORRECTED (v1.4): the actual distributed load(name="AWARE")
                        # checkpoint requires 20 bits (model.detection_net.output_length),
                        # confirmed by "Invalid watermark length. Expected 20, got 16."
                        # The paper's Table 1/2 states "16 bps" for its own comparison
                        # run, but that evidently used a different checkpoint/config than
                        # what's currently distributed in the public repo -- the repo's
                        # own README example (which embeds 20 bits even for this same
                        # default profile) was the correct signal, not the paper's prose.
VAL_FRACTION = 0.10
SEED = 20260716        # SAME seed — identical 105-clip split as every other experiment
CONF_THRESHOLD = 0.5   # AWARE's own documented default detection threshold

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
LOCAL_EMBEDDED = f"{LOCAL_SCRATCH}/aware_embedded_cache.npz"
LOCAL_RESULTS = f"{LOCAL_SCRATCH}/exp_baseline_aware_results.json"
LOCAL_FIG = f"{LOCAL_SCRATCH}/fig_16_01_aware_vs_ours.png"


# --- Corpus (resampled to AWARE's required 16kHz, matching 10's pattern) ----
def load_val_clips_16k():
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
    val_wavs_24k = wavs[val_idx]
    print(f"[{now()}] Resampling {len(val_idx)} held-out clips from {SR}Hz to "
          f"AWARE's required {AWARE_SR}Hz...")
    val_wavs_16k = []
    for w in val_wavs_24k:
        t = torch.from_numpy(w).float().unsqueeze(0)
        t16 = torchaudio.functional.resample(t, SR, AWARE_SR)
        val_wavs_16k.append(t16.squeeze(0).numpy())
    return np.stack(val_wavs_16k)


# --- AWARE model + self-test -------------------------------------------------
def load_aware_and_selftest(sample_wav_np):
    print(f"[{now()}] Loading AWARE (name='AWARE', default full-sequence profile, "
          f"{N_MSG_BITS}-bit payload)...")
    import sys
    aware_src_path = "/content/aware_repo/src"
    if aware_src_path not in sys.path:
        sys.path.insert(0, aware_src_path)
    from aware.utils.models import load
    from aware.service import embed_watermark, detect_watermark
    embedder, detector = load(name="AWARE")

    print(f"[{now()}] Self-test: embed + detect on one clip, NO attack applied...")
    msg = np.random.randint(0, 2, size=N_MSG_BITS, dtype=np.int32)
    watermarked = embed_watermark(sample_wav_np, AWARE_SR, msg, embedder)
    if len(watermarked) != len(sample_wav_np):
        print(f"[selftest] [WARNING] embed_watermark changed length: "
              f"{len(sample_wav_np)} -> {len(watermarked)}. Downstream code assumes "
              f"length is preserved (per Algorithm 1's phase-reuse iSTFT) — this may "
              f"need match_length() calls added if lengths genuinely differ.")
    detected_pattern, confidence = detect_watermark(watermarked, AWARE_SR, detector)
    n_wrong = int((np.asarray(detected_pattern).astype(int) != msg).sum())
    print(f"[selftest] bits wrong (no attack): {n_wrong}/{N_MSG_BITS}, "
          f"confidence={float(confidence):.4f} (expect close to 0 wrong, "
          f"confidence well above tau={CONF_THRESHOLD})")
    if n_wrong > N_MSG_BITS // 4:
        raise SystemExit("Self-test failed — AWARE round-trip did not recover the message.")
    print(f"[selftest] PASSED.")
    return embedder, detector, embed_watermark, detect_watermark


def match_length(arr, target_len):
    cur = len(arr)
    if cur == target_len:
        return arr
    if cur > target_len:
        return arr[:target_len]
    return np.pad(arr, (0, target_len - cur))


# --- Benign + regen battery (identical pattern to 09-12, at 16kHz) ---------
def real_mp3_roundtrip(wav_np, bitrate_kbps, sr):
    from pydub import AudioSegment
    import soundfile as sf
    with tempfile.NamedTemporaryFile(suffix=".wav") as wav_f, \
         tempfile.NamedTemporaryFile(suffix=".mp3") as mp3_f:
        sf.write(wav_f.name, wav_np, sr)
        AudioSegment.from_wav(wav_f.name).export(mp3_f.name, format="mp3", bitrate=f"{bitrate_kbps}k")
        seg = AudioSegment.from_mp3(mp3_f.name)
        out = np.array(seg.get_array_of_samples()).astype(np.float32)
        if seg.channels > 1:
            out = out.reshape(-1, seg.channels).mean(axis=1)
        out = out / (2 ** (8 * seg.sample_width - 1))
        if seg.frame_rate != sr:
            out = torchaudio.functional.resample(
                torch.from_numpy(out).unsqueeze(0), seg.frame_rate, sr).squeeze(0).numpy()
        return out


def resample_roundtrip(wav_np, sr, target_rate):
    t = torch.from_numpy(wav_np).float().unsqueeze(0)
    down = torchaudio.functional.resample(t, sr, target_rate)
    return torchaudio.functional.resample(down, target_rate, sr).squeeze(0).numpy()


def add_noise_np(wav_np, snr_db):
    sig_power = np.mean(wav_np ** 2)
    noise_power = sig_power / (10 ** (snr_db / 10))
    return wav_np + np.random.randn(*wav_np.shape).astype(np.float32) * np.sqrt(noise_power)


def apply_gain_np(wav_np, g):
    return wav_np * g


def eq_6db_np(wav_np, sr, n_fft=1024, hop=256, max_db=6.0, n_bands=6):
    t = torch.from_numpy(wav_np).float().unsqueeze(0)
    window = torch.hann_window(n_fft)
    spec = torch.stft(t, n_fft=n_fft, hop_length=hop, window=window, return_complex=True)
    n_bins = spec.shape[-2]
    band_gains_db = (torch.rand(1, 1, n_bands) * 2 - 1) * max_db
    gain_curve_db = F.interpolate(band_gains_db, size=n_bins, mode="linear", align_corners=True)
    gain_curve_db = gain_curve_db.squeeze(1).unsqueeze(-1)
    gain_lin = 10 ** (gain_curve_db / 20)
    out = torch.istft(spec * gain_lin, n_fft=n_fft, hop_length=hop, window=window, length=t.shape[-1])
    return out.squeeze(0).numpy()


def griffinlim_regen_np(wav_np, sr, n_fft=1024, hop=256, n_iter=32):
    t = torch.from_numpy(wav_np).float().unsqueeze(0)
    window = torch.hann_window(n_fft)
    spec = torch.stft(t, n_fft=n_fft, hop_length=hop, window=window, return_complex=True)
    mag = spec.abs()
    out = torchaudio.functional.griffinlim(
        mag, window=window, n_fft=n_fft, hop_length=hop, win_length=n_fft,
        power=1.0, n_iter=n_iter, momentum=0.99, length=t.shape[-1], rand_init=True,
    )
    return out.squeeze(0).numpy()


def load_dac():
    import dac
    dac_model_path = dac.utils.download(model_type="16khz")
    dac_model = dac.DAC.load(dac_model_path).to(DEVICE)
    dac_model.eval()
    return dac_model


def dac_regen_np(dac_model, wav_np, sr):
    wav_t = torch.from_numpy(wav_np).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        x = dac_model.preprocess(wav_t, sr)
        z, codes, latents, _, _ = dac_model.encode(x)
        y = dac_model.decode(z)
    return match_length(y.squeeze().cpu().numpy(), len(wav_np))


PROMPT_SECONDS = 3.0
TOTAL_GEN_SECONDS = 10.0
MUSICGEN_FRAME_RATE = 50


def load_musicgen():
    from transformers import AutoProcessor, MusicgenForConditionalGeneration
    processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
    model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small").to(DEVICE)
    model.eval()
    mg_sr = model.config.audio_encoder.sampling_rate
    return processor, model, mg_sr


def musicgen_regen_np(processor, model, mg_sr, wav_np, sr):
    prompt_samples = int(PROMPT_SECONDS * sr)
    prompt_np = wav_np[:prompt_samples]
    prompt_t = torch.from_numpy(prompt_np).float().unsqueeze(0)
    prompt_resampled = torchaudio.functional.resample(prompt_t, sr, mg_sr).squeeze(0).numpy()
    additional_seconds = TOTAL_GEN_SECONDS - PROMPT_SECONDS
    max_new_tokens = int(additional_seconds * MUSICGEN_FRAME_RATE)
    inputs = processor(audio=prompt_resampled, sampling_rate=mg_sr, text=["music"],
                        padding=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(**inputs, do_sample=True, guidance_scale=3, max_new_tokens=max_new_tokens)
    out_wav = out[0, 0].cpu()
    out_target_sr = torchaudio.functional.resample(out_wav.unsqueeze(0), mg_sr, sr).squeeze(0).numpy()
    return match_length(out_target_sr, len(wav_np))


# --- Pitch shift / time stretch (identical severities to 15) ---------------
PITCH_SHIFT_CENTS = [5, 25, 50, 100]
TIME_STRETCH_RATES = [0.9, 1.1, 1.4]


def pitch_shift_np(wav_np, sr, cents):
    return librosa.effects.pitch_shift(wav_np, sr=sr, n_steps=cents / 100.0)


def time_stretch_np(wav_np, rate):
    return librosa.effects.time_stretch(wav_np, rate=rate)


# --- Main ---------------------------------------------------------------
RESULTS_REMOTE = "exp_baseline_aware_results.json"


def load_checkpoint():
    if copy_from_project(RESULTS_REMOTE, LOCAL_RESULTS, skip_if_exists=False):
        with open(LOCAL_RESULTS) as f:
            results = json.load(f)
        print(f"[{now()}] Found existing results checkpoint in PROJECT_DIR — resuming.")
        return results
    print(f"[{now()}] No existing results checkpoint found — starting fresh.")
    return {
        "ber_benign": {"compression_kbps": {}, "resample_hz": {}, "noise_snr_db": {}, "gain": {}, "eq": {}},
        "ber_regen": {"griffinlim": {}, "dac": None, "musicgen": None},
        "pitch_shift": {}, "time_stretch": {},
        "detection_diagnostics": {},
    }


def save_checkpoint(results):
    with open(LOCAL_RESULTS, "w") as f:
        json.dump(results, f, indent=2)
    copy_to_project(LOCAL_RESULTS, RESULTS_REMOTE)


def main():
    val_wavs = load_val_clips_16k()
    embedder, detector, embed_watermark, detect_watermark = load_aware_and_selftest(val_wavs[0])

    # --- Embedding: CHECKPOINTED, unlike every other baseline (see header) ---
    if copy_from_project("aware_embedded_cache.npz", LOCAL_EMBEDDED):
        print(f"[{now()}] Loaded cached embedded audio from PROJECT_DIR — skipping the "
              f"500-iteration-per-clip embedding step entirely.")
        cache = np.load(LOCAL_EMBEDDED, allow_pickle=True)
        watermarked_list = list(cache["watermarked"])
        true_msgs = cache["true_msgs"]
    else:
        print(f"[{now()}] Embedding all {len(val_wavs)} clips with AWARE "
              f"(500 iterations each — this is the slow part)...")
        watermarked_list, true_msgs = [], []
        for i in tqdm(range(len(val_wavs)), desc="AWARE embed"):
            msg = np.random.randint(0, 2, size=N_MSG_BITS, dtype=np.int32)
            wm = embed_watermark(val_wavs[i], AWARE_SR, msg, embedder)
            watermarked_list.append(wm.astype(np.float32))
            true_msgs.append(msg)
        true_msgs = np.stack(true_msgs)
        np.savez(LOCAL_EMBEDDED, watermarked=np.array(watermarked_list, dtype=object), true_msgs=true_msgs)
        copy_to_project(LOCAL_EMBEDDED, "aware_embedded_cache.npz")
        print(f"[{now()}] Embedding complete and cached to PROJECT_DIR.")

    results = load_checkpoint()
    ber_benign, ber_regen = results["ber_benign"], results["ber_regen"]
    diag = results["detection_diagnostics"]

    def ber_and_detection_for_variant(variant_fn):
        n_wrong, n_total, n_undetected = 0, 0, 0
        for i in range(len(watermarked_list)):
            variant = variant_fn(watermarked_list[i])
            variant = match_length(variant, len(watermarked_list[i]))
            detected_pattern, confidence = detect_watermark(variant.astype(np.float32), AWARE_SR, detector)
            detected = np.asarray(detected_pattern).astype(int)
            n_wrong += int((detected != true_msgs[i]).sum())
            if float(confidence) < CONF_THRESHOLD:
                n_undetected += 1
            n_total += N_MSG_BITS
        ber = float(n_wrong / n_total)
        detection_rate = float((len(watermarked_list) - n_undetected) / len(watermarked_list))
        return ber, detection_rate, n_undetected

    def run_condition(group_dict, key, label, compute_fn):
        key = str(key)
        if key in group_dict and group_dict[key] is not None:
            print(f"[{now()}]   [skip, already checkpointed] {label}: {group_dict[key]}")
            return
        print(f"[{now()}]   {label}...")
        ber, det_rate, n_undet = compute_fn()
        group_dict[key] = {"ber": ber, "detection_rate": det_rate, "n_undetected": n_undet,
                            "n_clips": len(watermarked_list)}
        print(f"[{now()}]   {label}: BER={ber:.4f}, detection_rate={det_rate:.4f}")
        save_checkpoint(results)

    for kbps in (32, 64, 128, 192, 320):
        run_condition(ber_benign["compression_kbps"], kbps, f"[benign] compression {kbps}kbps",
                      lambda k=kbps: ber_and_detection_for_variant(lambda w, kk=k: real_mp3_roundtrip(w, kk, AWARE_SR)))
    for rate in (8000, 11025, 16000, 22050, 24000):
        run_condition(ber_benign["resample_hz"], rate, f"[benign] resample {rate}Hz",
                      lambda r=rate: ber_and_detection_for_variant(lambda w, rr=r: resample_roundtrip(w, AWARE_SR, rr)))
    for snr in (10, 20, 30, 40):
        run_condition(ber_benign["noise_snr_db"], snr, f"[benign] noise {snr}dB SNR",
                      lambda s=snr: ber_and_detection_for_variant(lambda w, ss=s: add_noise_np(w, ss)))
    for g in (0.7, 1.0, 1.3):
        run_condition(ber_benign["gain"], g, f"[benign] gain {g}",
                      lambda gg=g: ber_and_detection_for_variant(lambda w, ggg=gg: apply_gain_np(w, ggg)))
    run_condition(ber_benign["eq"], "max_6db", "[benign] EQ (+-6dB)",
                  lambda: ber_and_detection_for_variant(lambda w: eq_6db_np(w, AWARE_SR)))

    for n_iter in (0, 4, 16, 64):
        run_condition(ber_regen["griffinlim"], n_iter, f"[regen] Griffin-Lim n_iter={n_iter}",
                      lambda ni=n_iter: ber_and_detection_for_variant(lambda w, nn=ni: griffinlim_regen_np(w, AWARE_SR, n_iter=nn)))

    if ber_regen.get("dac") is None:
        print(f"[{now()}]   [regen] loading DAC (16kHz variant)...")
        dac_model = load_dac()
        print(f"[{now()}]   [regen] DAC full-clip resynthesis...")
        ber, det_rate, n_undet = ber_and_detection_for_variant(lambda w: dac_regen_np(dac_model, w, AWARE_SR))
        ber_regen["dac"] = {"ber": ber, "detection_rate": det_rate, "n_undetected": n_undet, "n_clips": len(watermarked_list)}
        save_checkpoint(results)
        del dac_model
        torch.cuda.empty_cache()
    else:
        print(f"[{now()}]   [skip, already checkpointed] [regen] DAC: {ber_regen['dac']}")

    if ber_regen.get("musicgen") is None:
        print(f"[{now()}]   [regen] loading MusicGen-small...")
        mg_processor, mg_model, mg_sr = load_musicgen()
        print(f"[{now()}]   [regen] MusicGen partial continuation (slow)...")
        ber, det_rate, n_undet = ber_and_detection_for_variant(
            lambda w: musicgen_regen_np(mg_processor, mg_model, mg_sr, w, AWARE_SR))
        ber_regen["musicgen"] = {"ber": ber, "detection_rate": det_rate, "n_undetected": n_undet, "n_clips": len(watermarked_list)}
        save_checkpoint(results)
    else:
        print(f"[{now()}]   [skip, already checkpointed] [regen] MusicGen: {ber_regen['musicgen']}")

    # --- Pitch shift / time stretch, matching 15's exact severities ---------
    for cents in PITCH_SHIFT_CENTS:
        run_condition(results["pitch_shift"], cents, f"[pitch] {cents} cents",
                      lambda c=cents: ber_and_detection_for_variant(lambda w, cc=c: pitch_shift_np(w, AWARE_SR, cc)))
    for rate in TIME_STRETCH_RATES:
        run_condition(results["time_stretch"], rate, f"[time-stretch] rate={rate}",
                      lambda r=rate: ber_and_detection_for_variant(lambda w, rr=r: time_stretch_np(w, rr)))

    results["n_clips"] = len(val_wavs)
    results["n_msg_bits"] = N_MSG_BITS
    results["operating_sample_rate_hz"] = AWARE_SR
    results["note"] = (f"AWARE evaluated at N={N_MSG_BITS}-bit payload (load(name='AWARE'), default "
                        "full-sequence profile) -- the actual capacity required by the "
                        "distributed checkpoint, confirmed by a runtime ValueError, not the "
                        "16 bits AWARE's own paper states for its internal comparison table. "
                        "Still harder than our own K=4, not an advantage chosen either way, "
                        "and NOT the higher-capacity 'AWARE(20bps)' segment-based variant "
                        "(a different, unrelated 20 that happens to coincide numerically). "
                        "Embedding is a 500-iteration per-clip adversarial optimization "
                        "(Algorithm 1), checkpointed separately from per-condition results "
                        "given its cost.")
    save_checkpoint(results)

    print(f"[{now()}] ber_benign: {ber_benign}")
    print(f"[{now()}] ber_regen: {ber_regen}")
    print(f"[{now()}] pitch_shift: {results['pitch_shift']}")
    print(f"[{now()}] time_stretch: {results['time_stretch']}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    benign_flat = [v["ber"] for g in ber_benign.values() for v in g.values()]
    regen_flat = [v["ber"] for v in ber_regen["griffinlim"].values()] + [ber_regen["dac"]["ber"], ber_regen["musicgen"]["ber"]]
    axes[0].boxplot([benign_flat, regen_flat], tick_labels=["Benign (18 settings)", "Regen (Griffin-Lim/DAC/MusicGen)"])
    axes[0].axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="chance")
    axes[0].set_ylabel(f"BER ({N_MSG_BITS}-bit message)")
    axes[0].set_title(f"AWARE baseline — n={len(val_wavs)} held-out clips")
    axes[0].legend()

    pitch_x = PITCH_SHIFT_CENTS
    stretch_x = TIME_STRETCH_RATES
    axes[1].plot(pitch_x, [results["pitch_shift"][str(c)]["ber"] for c in pitch_x], "o-", label="Pitch shift (cents)")
    ax2 = axes[1].twiny()
    ax2.plot(stretch_x, [results["time_stretch"][str(r)]["ber"] for r in stretch_x], "s--", color="orange", label="Time stretch (rate)")
    axes[1].axhline(0.5, color="gray", linestyle="--", alpha=0.3)
    axes[1].set_xlabel("Pitch shift (cents)")
    ax2.set_xlabel("Time stretch (rate)")
    axes[1].set_ylabel("BER")
    axes[1].set_title("AWARE: pitch-shift / time-stretch\n(cf. our own collapse in 15)")
    lines1, labels1 = axes[1].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axes[1].legend(lines1 + lines2, labels1 + labels2, fontsize=8)

    fig.tight_layout()
    fig.savefig(LOCAL_FIG, dpi=300)
    plt.close(fig)
    copy_to_project(LOCAL_FIG, "fig_16_01_aware_vs_ours.png")

    print(f"[{now()}] DONE. exp_baseline_aware_results.json, fig_16_01_aware_vs_ours.png saved to PROJECT_DIR.")


if __name__ == "__main__":
    main()