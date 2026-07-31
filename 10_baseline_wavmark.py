# =============================================================================
# Program      : 10_baseline_wavmark.py
# Version      : 1.2
# Description  : Quantitative baseline comparison — WavMark.
#
#                Second baseline added after the SPL prescreening rejection
#                for missing baseline comparison (see 09_baseline_audioseal.py
#                header for full context). WavMark (Chen et al., 2023) is
#                confirmed to have a public, pretrained, pip-installable
#                checkpoint (`pip install wavmark`) — verified against the
#                official README before writing this, not assumed.
#
#                WavMark's high-level API (`wavmark.encode_watermark` /
#                `wavmark.decode_watermark`) embeds/recovers a 16-bit
#                payload. Internally the model actually encodes 32 bits —
#                a fixed 16-bit identification pattern plus the 16-bit
#                custom payload — but the high-level functions used here
#                abstract that away; only the 16-bit payload is what we
#                embed and score.
#
#                REQUIRED SAMPLE RATE: WavMark operates on single-channel
#                16 kHz audio ONLY (unlike AudioSeal, which handles 24 kHz
#                natively). Our clips are downsampled to 16 kHz once before
#                embedding, and the ENTIRE benign+regen battery runs at
#                16 kHz throughout — this mirrors how any real deployment
#                of WavMark would have to use it, not an artificial
#                handicap chosen here.
#
#                ENTIRELY INFERENCE-ONLY, same as 09 — no training, no
#                backward passes anywhere in this script.
#
# PRE-FLIGHT SELF-TEST: embed+decode round-trip on one clip with NO attack
# applied, confirms near-zero BER before running the full battery.
#
# INPUT:
#                  PROJECT_DIR (/content/drive/MyDrive/paper/HyperFrag/):
#                    dataset_e0.h5   — from 01_e0_dataextract.py
#
# STEPS:
#                  Step 1  Download dataset_e0.h5, extract the SAME 105
#                          held-out clips used by every other experiment,
#                          resample to 16kHz once
#                  Step 2  Load WavMark (pretrained), self-test round-trip
#                  Step 3  Embed a random 16-bit payload in all 105 clips
#                  Step 4  Score BER under the identical benign+regen
#                          battery used for AudioSeal (09) and our own
#                          system, at 16kHz throughout
#                  Step 5  Save results in the same JSON structure as 09
#                          for direct table merging
#
# OUTPUT FILES : (stored at /content/drive/MyDrive/paper/HyperFrag/)
#                  exp_baseline_wavmark_results.json
#                  fig_10_01_wavmark_vs_ours.png
#
# GPU Required : Optional (inference-only)
# Dependencies : torch, torchaudio, wavmark, pydub, descript-audio-codec,
#                transformers, h5py, matplotlib, tqdm
#
# Change Log   :
#   v1.0  2026-07-27  Initial version
#   v1.1  2026-07-27  Added PROJECT_DIR checkpoint-and-resume: results are saved to
#                      PROJECT_DIR after every individual condition, not just once
#                      at the end. A run observed in production took ~17min
#                      per benign condition (WavMark's decoder appears to do
#                      an internal sync-position search, unlike AudioSeal's
#                      single forward pass) — an uncheckpointed multi-hour
#                      Colab run is a real disconnection risk. Rerunning
#                      this script after any interruption now resumes from
#                      the last completed condition instead of restarting.
#   v1.2  2026-07-28  Added a targeted detection-rate diagnostic pass.
#                      Production results showed several conditions at
#                      BER=1.0 (10dB noise, 20dB noise, Griffin-Lim n_iter=0,
#                      DAC, MusicGen) — a BER of 1.0 is ambiguous between
#                      "WavMark failed to detect the watermark at all"
#                      (decode() returns None, scored as maximally wrong)
#                      and "WavMark detected it but decoded confidently
#                      wrong." DIAGNOSE_CONDITIONS controls which flagged
#                      conditions get re-run with detection-rate tracking;
#                      this does NOT touch or re-run any of the ~19 already-
#                      good conditions from the main battery above, and is
#                      checkpointed separately under
#                      results["detection_diagnostics"] so it can never
#                      silently overwrite a BER already on record.
#                      INCLUDE_MUSICGEN_DIAGNOSIS toggles the one genuinely
#                      slow (~1hr+) condition in this pass.
#
# !pip install torch torchaudio wavmark pydub descript-audio-codec transformers h5py matplotlib tqdm
# =============================================================================

!pip install -q wavmark pydub descript-audio-codec transformers h5py matplotlib tqdm
!pip install "protobuf>=5.28.0"

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE} (inference-only, same as 09_baseline_audioseal.py).")

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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

SR = 24000            # our corpus's native rate
WM_SR = 16000          # WavMark's REQUIRED operating rate — not a choice
VAL_FRACTION = 0.10
SEED = 20260716
N_MSG_BITS = 16

# Targeted diagnostic pass (added after production results showed several
# conditions at or near BER=1.0): re-runs ONLY these specific conditions
# with detection-rate tracking, to distinguish "WavMark failed to detect
# the watermark at all" from "WavMark detected it but decoded it wrong" —
# a distinction the main battery above doesn't capture. Does NOT touch or
# re-run the ~19 already-good conditions already checkpointed. Set
# INCLUDE_MUSICGEN_DIAGNOSIS = False to skip the slow (~1hr+) MusicGen
# regeneration step if you only need the fast conditions diagnosed.
INCLUDE_MUSICGEN_DIAGNOSIS = True
DIAGNOSE_CONDITIONS = [
    ("noise_snr_db", 10, lambda w: add_noise_np(w, 10)),
    ("noise_snr_db", 20, lambda w: add_noise_np(w, 20)),
    ("griffinlim", 0, lambda w: griffinlim_regen_np(w, WM_SR, n_iter=0)),
    ("dac", None, None),        # handled specially below — needs the DAC model loaded
    ("musicgen", None, None),   # handled specially below — needs MusicGen loaded, slow
]

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
LOCAL_RESULTS = f"{LOCAL_SCRATCH}/exp_baseline_wavmark_results.json"
LOCAL_FIG = f"{LOCAL_SCRATCH}/fig_10_01_wavmark_vs_ours.png"


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
    print(f"[{now()}] {len(val_idx)} held-out clips (identical split to E1-E6, "
          f"09_baseline_audioseal.py). Resampling to {WM_SR}Hz for WavMark...")
    val_wavs_16k = []
    for w in val_wavs_24k:
        t = torch.from_numpy(w).float().unsqueeze(0)
        t16 = torchaudio.functional.resample(t, SR, WM_SR)
        val_wavs_16k.append(t16.squeeze(0).numpy())
    return np.stack(val_wavs_16k)


def load_wavmark_and_selftest(sample_wav_np):
    print(f"[{now()}] Loading WavMark (pretrained)...")
    import wavmark
    model = wavmark.load_model().to(DEVICE)

    print(f"[{now()}] Self-test: embed + decode on one clip, NO attack applied...")
    payload = np.random.choice([0, 1], size=N_MSG_BITS)
    watermarked, _ = wavmark.encode_watermark(model, sample_wav_np, payload, show_progress=False)
    decoded, _ = wavmark.decode_watermark(model, watermarked, show_progress=False)
    n_wrong = int((payload != decoded).sum())
    print(f"[selftest] payload bits wrong (no attack): {n_wrong}/{N_MSG_BITS} "
          f"(expect 0 or very close to it)")
    if n_wrong > N_MSG_BITS // 4:
        print(f"[selftest] [FATAL] more than a quarter of bits wrong under zero attack — "
              f"the embed/decode call is not behaving as documented.")
        raise SystemExit("Self-test failed — WavMark round-trip did not recover the payload.")
    print(f"[selftest] PASSED. Proceeding to the full evaluation battery.")
    return model


def match_length(arr, target_len):
    is_tensor = torch.is_tensor(arr)
    cur = arr.shape[-1]
    if cur == target_len:
        return arr
    if cur > target_len:
        return arr[..., :target_len]
    pad = target_len - cur
    return F.pad(arr, (0, pad)) if is_tensor else np.pad(arr, (0, pad))


# --- Benign transforms, all operating at WM_SR (16kHz) ----------------------
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
    dac_model_path = dac.utils.download(model_type="16khz")  # matches WavMark's operating rate
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
    out_sr = torchaudio.functional.resample(out_wav.unsqueeze(0), mg_sr, sr).squeeze(0).numpy()
    return match_length(out_sr, len(wav_np))


# --- Main ---------------------------------------------------------------
RESULTS_REMOTE = "exp_baseline_wavmark_results.json"


def load_checkpoint():
    """Returns the partial/complete results dict already in PROJECT_DIR, or a fresh
    skeleton if nothing has been saved yet. This is what makes the script
    resumable — rerunning it after any interruption picks up from here."""
    if copy_from_project(RESULTS_REMOTE, LOCAL_RESULTS, skip_if_exists=False):
        with open(LOCAL_RESULTS) as f:
            results = json.load(f)
        n_done = sum(len(v) if isinstance(v, dict) else (1 if v is not None else 0)
                     for v in list(results.get("ber_benign", {}).values())
                     + list(results.get("ber_regen", {}).values()))
        print(f"[{now()}] Found existing checkpoint in PROJECT_DIR with {n_done} conditions "
              f"already scored — resuming, not restarting from scratch.")
        return results
    print(f"[{now()}] No existing checkpoint found — starting fresh.")
    return {"ber_benign": {"compression_kbps": {}, "resample_hz": {}, "noise_snr_db": {}, "gain": {}, "eq": {}},
            "ber_regen": {"griffinlim": {}, "dac": None, "musicgen": None}}


def save_checkpoint(results):
    with open(LOCAL_RESULTS, "w") as f:
        json.dump(results, f, indent=2)
    copy_to_project(LOCAL_RESULTS, RESULTS_REMOTE)


def main():
    val_wavs = load_val_clips_16k()
    model = load_wavmark_and_selftest(val_wavs[0])
    results = load_checkpoint()
    ber_benign, ber_regen = results["ber_benign"], results["ber_regen"]

    print(f"[{now()}] Embedding WavMark payloads in all {len(val_wavs)} clips "
          f"(re-embedded fresh each run with new random payloads — each "
          f"condition's BER is a valid, unbiased estimate regardless of which "
          f"random payload draw computed it, so this does not need to be "
          f"checkpointed separately; only the slow per-condition scoring below does).")
    import wavmark
    true_payloads = []
    watermarked_list = []
    for i in tqdm(range(len(val_wavs)), desc="wavmark embed"):
        payload = np.random.choice([0, 1], size=N_MSG_BITS)
        watermarked, _ = wavmark.encode_watermark(model, val_wavs[i], payload, show_progress=False)
        watermarked_list.append(watermarked.astype(np.float32))
        true_payloads.append(payload)
    true_payloads = np.stack(true_payloads)

    def ber_for_variant(variant_fn):
        n_wrong, n_total = 0, 0
        for i in range(len(watermarked_list)):
            variant = variant_fn(watermarked_list[i])
            variant = match_length(variant, len(watermarked_list[i]))
            decoded, _ = wavmark.decode_watermark(model, variant.astype(np.float32), show_progress=False)
            if decoded is None:
                n_wrong += N_MSG_BITS
            else:
                n_wrong += int((decoded != true_payloads[i]).sum())
            n_total += N_MSG_BITS
        return float(n_wrong / n_total)

    def ber_and_detection_for_variant(variant_fn):
        """Same computation as ber_for_variant, but also tracks how many
        clips WavMark failed to detect at all (decode() returns None) vs.
        how many it detected but decoded WRONG. A BER of 1.0 could mean
        either — this distinguishes them. Used only for the targeted
        diagnostic pass below, not the main battery, so it doesn't cost
        anything on conditions that are already known to work fine."""
        n_wrong, n_total, n_none = 0, 0, 0
        for i in range(len(watermarked_list)):
            variant = variant_fn(watermarked_list[i])
            variant = match_length(variant, len(watermarked_list[i]))
            decoded, _ = wavmark.decode_watermark(model, variant.astype(np.float32), show_progress=False)
            if decoded is None:
                n_wrong += N_MSG_BITS
                n_none += 1
            else:
                n_wrong += int((decoded != true_payloads[i]).sum())
            n_total += N_MSG_BITS
        ber = float(n_wrong / n_total)
        detection_rate = float((len(watermarked_list) - n_none) / len(watermarked_list))
        return ber, detection_rate, n_none

    def run_condition(group_dict, key, label, compute_fn):
        """Runs one condition unless it's already in the checkpoint; saves
        immediately after. `key` is always used as a string since reloaded
        JSON dicts have string keys — keeps fresh and resumed runs consistent."""
        key = str(key)
        if key in group_dict and group_dict[key] is not None:
            print(f"[{now()}]   [skip, already checkpointed] {label}: {group_dict[key]:.4f}")
            return
        print(f"[{now()}]   {label}...")
        group_dict[key] = compute_fn()
        save_checkpoint(results)

    for kbps in (32, 64, 128, 192, 320):
        run_condition(ber_benign["compression_kbps"], kbps, f"[benign] compression {kbps}kbps",
                      lambda k=kbps: ber_for_variant(lambda w, kk=k: real_mp3_roundtrip(w, kk, WM_SR)))

    # Resample battery: WM_SR -> target -> WM_SR, target rates scaled to stay
    # meaningful at a 16kHz base rather than reusing the 24kHz-derived list.
    for rate in (8000, 11025, 16000, 22050, 24000):
        run_condition(ber_benign["resample_hz"], rate, f"[benign] resample {rate}Hz",
                      lambda r=rate: ber_for_variant(lambda w, rr=r: resample_roundtrip(w, WM_SR, rr)))

    for snr in (10, 20, 30, 40):
        run_condition(ber_benign["noise_snr_db"], snr, f"[benign] noise {snr}dB SNR",
                      lambda s=snr: ber_for_variant(lambda w, ss=s: add_noise_np(w, ss)))

    for g in (0.7, 1.0, 1.3):
        run_condition(ber_benign["gain"], g, f"[benign] gain {g}",
                      lambda gg=g: ber_for_variant(lambda w, ggg=gg: apply_gain_np(w, ggg)))

    run_condition(ber_benign["eq"], "max_6db", "[benign] EQ (+-6dB)",
                  lambda: ber_for_variant(lambda w: eq_6db_np(w, WM_SR)))

    for n_iter in (0, 4, 16, 64):
        run_condition(ber_regen["griffinlim"], n_iter, f"[regen] Griffin-Lim n_iter={n_iter}",
                      lambda ni=n_iter: ber_for_variant(lambda w, nn=ni: griffinlim_regen_np(w, WM_SR, n_iter=nn)))

    if ber_regen.get("dac") is None:
        print(f"[{now()}]   [regen] loading DAC (16kHz variant)...")
        dac_model = load_dac()
        print(f"[{now()}]   [regen] DAC full-clip resynthesis...")
        ber_regen["dac"] = ber_for_variant(lambda w: dac_regen_np(dac_model, w, WM_SR))
        save_checkpoint(results)
        del dac_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    else:
        print(f"[{now()}]   [skip, already checkpointed] [regen] DAC: {ber_regen['dac']:.4f}")

    if ber_regen.get("musicgen") is None:
        print(f"[{now()}]   [regen] loading MusicGen-small...")
        mg_processor, mg_model, mg_sr = load_musicgen()
        print(f"[{now()}]   [regen] MusicGen partial continuation (slow)...")
        ber_regen["musicgen"] = ber_for_variant(
            lambda w: musicgen_regen_np(mg_processor, mg_model, mg_sr, w, WM_SR))
        save_checkpoint(results)
    else:
        print(f"[{now()}]   [skip, already checkpointed] [regen] MusicGen: {ber_regen['musicgen']:.4f}")

    # --- Targeted detection-rate diagnostic pass ---
    # Only runs on the conditions listed in DIAGNOSE_CONDITIONS (the ones
    # that showed BER at or near 1.0 in the production run) — does NOT
    # touch or re-run anything already checkpointed above. Checkpointed
    # separately under results["detection_diagnostics"] so a BER of 1.0
    # from a prior run is never silently overwritten by this pass.
    if "detection_diagnostics" not in results:
        results["detection_diagnostics"] = {}
    diag = results["detection_diagnostics"]

    for group, key, transform_fn in DIAGNOSE_CONDITIONS:
        diag_key = f"{group}_{key}" if key is not None else group
        if diag_key in diag:
            d = diag[diag_key]
            print(f"[{now()}]   [skip, already diagnosed] {diag_key}: "
                  f"BER={d['ber']:.4f}, detection_rate={d['detection_rate']:.4f}, "
                  f"n_none={d['n_none']}/{len(watermarked_list)}")
            continue

        if group == "dac":
            print(f"[{now()}]   [diagnose] loading DAC (16kHz variant)...")
            dac_model = load_dac()
            ber, det_rate, n_none = ber_and_detection_for_variant(lambda w: dac_regen_np(dac_model, w, WM_SR))
            del dac_model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        elif group == "musicgen":
            if not INCLUDE_MUSICGEN_DIAGNOSIS:
                print(f"[{now()}]   [diagnose] SKIPPING MusicGen diagnosis "
                      f"(INCLUDE_MUSICGEN_DIAGNOSIS=False) — set to True to include it "
                      f"(adds ~1hr+ for the slow regeneration step).")
                continue
            print(f"[{now()}]   [diagnose] loading MusicGen-small...")
            mg_processor, mg_model, mg_sr = load_musicgen()
            print(f"[{now()}]   [diagnose] MusicGen partial continuation (slow)...")
            ber, det_rate, n_none = ber_and_detection_for_variant(
                lambda w: musicgen_regen_np(mg_processor, mg_model, mg_sr, w, WM_SR))
        else:
            print(f"[{now()}]   [diagnose] {diag_key}...")
            ber, det_rate, n_none = ber_and_detection_for_variant(transform_fn)

        diag[diag_key] = {"ber": ber, "detection_rate": det_rate, "n_none": n_none,
                           "n_clips": len(watermarked_list)}
        print(f"[{now()}]   [diagnose] {diag_key}: BER={ber:.4f}, detection_rate={det_rate:.4f} "
              f"({len(watermarked_list) - n_none}/{len(watermarked_list)} clips detected at all)")
        save_checkpoint(results)

    results["n_clips"] = len(val_wavs)
    results["n_msg_bits"] = N_MSG_BITS
    results["operating_sample_rate_hz"] = WM_SR
    results["note"] = ("WavMark requires 16kHz mono input, a lower rate than our 24kHz "
                        "corpus and AudioSeal's native rate — evaluated at its required "
                        "rate throughout, matching real deployment constraints.")
    save_checkpoint(results)

    print(f"[{now()}] WavMark BER benign: {ber_benign}")
    print(f"[{now()}] WavMark BER regen: {ber_regen}")

    fig, ax = plt.subplots(figsize=(10, 5))
    benign_flat = [v for g in ber_benign.values() for v in g.values()]
    regen_flat = list(ber_regen["griffinlim"].values()) + [ber_regen["dac"], ber_regen["musicgen"]]
    ax.boxplot([benign_flat, regen_flat], labels=["Benign (18 settings)", "Regen (Griffin-Lim/DAC/MusicGen)"])
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="chance")
    ax.set_ylabel("BER (16-bit payload)")
    ax.set_title(f"WavMark baseline — n={len(val_wavs)} held-out clips, 16kHz")
    ax.legend()
    fig.tight_layout()
    fig.savefig(LOCAL_FIG, dpi=300)
    plt.close(fig)
    copy_to_project(LOCAL_FIG, "fig_10_01_wavmark_vs_ours.png")

    print(f"[{now()}] DONE. exp_baseline_wavmark_results.json, "
          f"fig_10_01_wavmark_vs_ours.png saved to PROJECT_DIR.")


if __name__ == "__main__":
    main()