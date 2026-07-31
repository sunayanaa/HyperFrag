# =============================================================================
# Program      : 11_baseline_silentcipher.py
# Version      : 1.4
# Description  : Quantitative baseline comparison — SilentCipher.
#
#                Third baseline added after the SPL prescreening rejection
#                for missing baseline comparison (see 09_baseline_audioseal.py
#                header for full context). SilentCipher (Singh et al.,
#                Interspeech 2024) is confirmed to have public, pretrained
#                checkpoints on Hugging Face (loaded automatically by the
#                `silentcipher` PyPI package) — verified against the
#                official README before writing this, not assumed.
#
#                SilentCipher's message format is five 8-bit values (list
#                of ints 0-255) — 40 bits total, higher capacity than
#                AudioSeal/WavMark's 16 bits and our own K=4. Decomposed
#                into individual bits here for a BER metric at the same
#                granularity as every other result in this project.
#
#                REQUIRED SAMPLE RATE: SilentCipher ships '44.1k' and '16k'
#                model variants; the 44.1k variant is used here as the
#                paper's own recommended, higher-fidelity configuration.
#                Our 24kHz clips are upsampled to 44.1kHz once before
#                embedding, and the full battery runs at 44.1kHz throughout.
#
#                ENTIRELY INFERENCE-ONLY, same as 09 and 10.
#
# PRE-FLIGHT SELF-TEST: encode+decode round-trip on one clip with NO attack
# applied, confirms near-zero BER and a "SUCCESS" status before running the
# full battery.
#
# INPUT:
#                  PROJECT_DIR (/content/drive/MyDrive/paper/HyperFrag/):
#                    dataset_e0.h5   — from 01_e0_dataextract.py
#
# STEPS:
#                  Step 1  Download dataset_e0.h5, extract the SAME 105
#                          held-out clips used by every other experiment,
#                          resample to 44.1kHz once
#                  Step 2  Load SilentCipher (pretrained, 44.1k), self-test
#                  Step 3  Embed a random 5x8-bit message in all 105 clips
#                  Step 4  Score BER under the identical benign+regen
#                          battery used for AudioSeal (09) / WavMark (10)
#                  Step 5  Save results in the same JSON structure as 09/10
#                          for direct table merging
#
# OUTPUT FILES : (stored at /content/drive/MyDrive/paper/HyperFrag/)
#                  exp_baseline_silentcipher_results.json
#                  fig_11_01_silentcipher_vs_ours.png
#
# GPU Required : Optional (inference-only)
# Dependencies : torch, torchaudio, silentcipher, librosa, pydub,
#                descript-audio-codec, transformers, h5py, matplotlib, tqdm
#
# Change Log   :
#   v1.0  2026-07-27  Initial version
#   v1.1  2026-07-27  Added PROJECT_DIR checkpoint-and-resume, same fix as
#                      09_baseline_audioseal.py v1.1 and
#                      10_baseline_wavmark.py v1.1.
#   v1.2  2026-07-28  Fixed a torch/torchaudio ABI mismatch
#                      (OSError: undefined symbol: aoti_torch_abi_version)
#                      hit on the very first import in production. This is
#                      a known class of failure: installing silentcipher
#                      let pip's dependency resolver pull a torch/torchaudio
#                      pairing that didn't match what was already active in
#                      the session — 09/10/12 all share descript-audio-codec/
#                      transformers/pydub without hitting this, so
#                      silentcipher (a much less mainstream package, more
#                      likely to pin its own torch version) was the prime
#                      suspect. Fixed by installing silentcipher with
#                      --no-deps, isolating it from torch/torchaudio
#                      resolution entirely. IMPORTANT: this code fix alone
#                      does not repair an already-broken session — the
#                      mismatched torch was already loaded into memory
#                      before the error surfaced, so a genuine Colab
#                      runtime restart is required before re-running,
#                      not just re-executing the cell.
#   v1.3  2026-07-28  Fixed encode_wav's message format. Hit
#                      AssertionError "5 | 1 Mismatch in the number of
#                      messages and channels" on first production run —
#                      encode_wav expects one message PER AUDIO CHANNEL
#                      (message_list length must equal the input's channel
#                      count after internal reshape), not a flat N-byte
#                      message. Our audio is mono (1 channel), so the fix
#                      wraps the 5-byte message as a single-element list,
#                      [message], not message directly. NOTE: the official
#                      README's usage example passes a flat 5-int list and
#                      is presented as correct — this fix contradicts that
#                      example. Reasoned directly from the assertion's
#                      actual shape values (5 vs. 1) rather than the docs,
#                      since the docs and the runtime behavior conflicted;
#                      verify this is correct via the self-test output
#                      before trusting a full run, rather than assuming
#                      this reasoning was right.
#   v1.4  2026-07-28  Added a targeted detection-rate diagnostic pass.
#                      Production results showed several conditions well
#                      ABOVE chance (32kbps compression 0.712, 10/20dB
#                      noise 0.813/0.780, Griffin-Lim@0/4 0.774/0.735, DAC
#                      0.768, MusicGen 0.800) and, critically, a benign
#                      condition (10dB noise) scoring WORSE than a
#                      regeneration condition (Griffin-Lim@64) — meaning
#                      BER here may track transform severity rather than
#                      benign-vs-regenerated category at all. Unlike
#                      09/10, this uses SilentCipher's own direct
#                      detection signals (result['status'],
#                      result['confidences']) rather than a threshold
#                      proxy — no guessing needed, the API exposes this
#                      directly. DIAGNOSE_CONDITIONS controls which
#                      flagged conditions get re-run; does NOT touch or
#                      re-run the ~11 already-good conditions.
#                      INCLUDE_MUSICGEN_DIAGNOSIS toggles the one
#                      genuinely slow (~1hr+) condition.
#
# !pip install torch torchaudio librosa pydub descript-audio-codec transformers h5py matplotlib tqdm
# !pip install --no-deps silentcipher
# =============================================================================

!pip install -q librosa pydub descript-audio-codec transformers h5py matplotlib tqdm
!pip install -q --no-deps silentcipher

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE} (inference-only, same as 09/10).")

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

SR = 24000
SC_SR = 44100          # SilentCipher's higher-fidelity operating rate (also offers 16k)
VAL_FRACTION = 0.10
SEED = 20260716
N_MSG_BYTES = 5         # 5 x 8-bit values = 40 bits total, per SilentCipher's message format

# Targeted diagnostic pass (added after production results showed multiple
# conditions well ABOVE chance — 32kbps compression 0.712, 10/20dB noise
# 0.813/0.780, Griffin-Lim@0/4 0.774/0.735, DAC 0.768, MusicGen 0.800 — and
# critically, benign conditions scoring WORSE than regeneration conditions
# (10dB noise 0.813 vs. Griffin-Lim@64 0.142), meaning BER here tracks
# transform severity, not benign-vs-regenerated category, at all). Re-runs
# ONLY these flagged conditions with detection-rate + confidence tracking;
# does NOT touch or re-run the ~11 already-good conditions. Set
# INCLUDE_MUSICGEN_DIAGNOSIS = False to skip the slow (~1hr+) MusicGen
# regeneration step if you only need the fast conditions diagnosed.
INCLUDE_MUSICGEN_DIAGNOSIS = True
DIAGNOSE_CONDITIONS = [
    ("compression_kbps", 32, lambda w: real_mp3_roundtrip(w, 32, SC_SR)),
    ("noise_snr_db", 10, lambda w: add_noise_np(w, 10)),
    ("noise_snr_db", 20, lambda w: add_noise_np(w, 20)),
    ("griffinlim", 0, lambda w: griffinlim_regen_np(w, SC_SR, n_iter=0)),
    ("griffinlim", 4, lambda w: griffinlim_regen_np(w, SC_SR, n_iter=4)),
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
LOCAL_RESULTS = f"{LOCAL_SCRATCH}/exp_baseline_silentcipher_results.json"
LOCAL_FIG = f"{LOCAL_SCRATCH}/fig_11_01_silentcipher_vs_ours.png"


def load_val_clips_44k():
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
          f"09/10). Resampling to {SC_SR}Hz for SilentCipher...")
    val_wavs_44k = []
    for w in val_wavs_24k:
        t = torch.from_numpy(w).float().unsqueeze(0)
        t44 = torchaudio.functional.resample(t, SR, SC_SR)
        val_wavs_44k.append(t44.squeeze(0).numpy())
    return np.stack(val_wavs_44k)


def bytes_to_bits(byte_list):
    bits = []
    for b in byte_list:
        bits.extend([(b >> i) & 1 for i in range(7, -1, -1)])
    return np.array(bits, dtype=int)


def load_silentcipher_and_selftest(sample_wav_np):
    print(f"[{now()}] Loading SilentCipher (pretrained, 44.1kHz)...")
    import silentcipher
    model = silentcipher.get_model(
        model_type="44.1k", device=str(DEVICE))

    print(f"[{now()}] Self-test: encode + decode on one clip, NO attack applied...")
    message = list(np.random.randint(0, 256, size=N_MSG_BYTES))
    # encode_wav expects one message PER AUDIO CHANNEL (message_list length must
    # equal y.shape[1] after internal reshape) — our audio is mono, so that's a
    # list containing exactly one message, not the flat 5-byte message itself.
    # Discovered via AssertionError "5 | 1 Mismatch in the number of messages
    # and channels" on the first production run — the official README's usage
    # example passes a flat list, which appears to work for a different
    # audio shape than a mono 1-D array; this is what mono actually requires.
    encoded, sdr = model.encode_wav(sample_wav_np, SC_SR, [message])
    result = model.decode_wav(encoded, SC_SR, phase_shift_decoding=False)
    print(f"[selftest] decode status: {result['status']}, SDR: {sdr}")
    if result["status"] and result["messages"]:
        decoded_bits = bytes_to_bits(result["messages"][0])
        true_bits = bytes_to_bits(message)
        n_wrong = int((decoded_bits != true_bits).sum())
        print(f"[selftest] message bits wrong (no attack): {n_wrong}/{N_MSG_BYTES * 8} "
              f"(expect 0 or very close to it)")
        if n_wrong > (N_MSG_BYTES * 8) // 4:
            print(f"[selftest] [FATAL] more than a quarter of bits wrong under zero attack.")
            raise SystemExit("Self-test failed — SilentCipher round-trip did not recover the message.")
    else:
        print(f"[selftest] [FATAL] decode status False / no message returned under zero attack — "
              f"the encode/decode call is not behaving as documented.")
        raise SystemExit("Self-test failed — SilentCipher reported no detection under zero attack.")
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


# --- Benign transforms, all operating at SC_SR (44.1kHz) --------------------
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
    dac_model_path = dac.utils.download(model_type="44khz")  # matches SilentCipher's operating rate
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
RESULTS_REMOTE = "exp_baseline_silentcipher_results.json"


def load_checkpoint():
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
    val_wavs = load_val_clips_44k()
    model = load_silentcipher_and_selftest(val_wavs[0])
    results = load_checkpoint()
    ber_benign, ber_regen = results["ber_benign"], results["ber_regen"]

    print(f"[{now()}] Embedding SilentCipher messages in all {len(val_wavs)} clips "
          f"(re-embedded fresh each run — see 10_baseline_wavmark.py's main() "
          f"for why this doesn't need separate checkpointing)...")
    true_bits_list = []
    watermarked_list = []
    for i in tqdm(range(len(val_wavs)), desc="silentcipher embed"):
        message = list(np.random.randint(0, 256, size=N_MSG_BYTES))
        encoded, sdr = model.encode_wav(val_wavs[i], SC_SR, [message])
        watermarked_list.append(encoded.astype(np.float32))
        true_bits_list.append(bytes_to_bits(message))
    true_bits = np.stack(true_bits_list)  # [N, 40]

    def ber_for_variant(variant_fn):
        n_wrong, n_total = 0, 0
        for i in range(len(watermarked_list)):
            variant = variant_fn(watermarked_list[i])
            variant = match_length(variant, len(watermarked_list[i]))
            result = model.decode_wav(variant.astype(np.float32), SC_SR, phase_shift_decoding=False)
            if result["status"] and result["messages"]:
                decoded_bits = bytes_to_bits(result["messages"][0])
                n_wrong += int((decoded_bits != true_bits[i]).sum())
            else:
                n_wrong += N_MSG_BYTES * 8
            n_total += N_MSG_BYTES * 8
        return float(n_wrong / n_total)

    def ber_and_detection_for_variant(variant_fn):
        """Same as ber_for_variant, but also tracks SilentCipher's own
        direct detection signals — result['status'] (did it detect
        anything at all, the direct analog of WavMark's None) and
        result['confidences'] (a graded confidence score for whatever WAS
        detected, richer than what either WavMark or AudioSeal's API
        exposes directly, no threshold-guessing needed). Used only for the
        targeted diagnostic pass below, not the main battery."""
        n_wrong, n_total, n_undetected = 0, 0, 0
        confidences_collected = []
        for i in range(len(watermarked_list)):
            variant = variant_fn(watermarked_list[i])
            variant = match_length(variant, len(watermarked_list[i]))
            result = model.decode_wav(variant.astype(np.float32), SC_SR, phase_shift_decoding=False)
            if result["status"] and result["messages"]:
                decoded_bits = bytes_to_bits(result["messages"][0])
                n_wrong += int((decoded_bits != true_bits[i]).sum())
                if result.get("confidences"):
                    confidences_collected.append(float(result["confidences"][0]))
            else:
                n_wrong += N_MSG_BYTES * 8
                n_undetected += 1
            n_total += N_MSG_BYTES * 8
        ber = float(n_wrong / n_total)
        detection_rate = float((len(watermarked_list) - n_undetected) / len(watermarked_list))
        mean_confidence = float(np.mean(confidences_collected)) if confidences_collected else None
        return ber, detection_rate, n_undetected, mean_confidence

    def run_condition(group_dict, key, label, compute_fn):
        key = str(key)
        if key in group_dict and group_dict[key] is not None:
            print(f"[{now()}]   [skip, already checkpointed] {label}: {group_dict[key]:.4f}")
            return
        print(f"[{now()}]   {label}...")
        group_dict[key] = compute_fn()
        save_checkpoint(results)

    for kbps in (32, 64, 128, 192, 320):
        run_condition(ber_benign["compression_kbps"], kbps, f"[benign] compression {kbps}kbps",
                      lambda k=kbps: ber_for_variant(lambda w, kk=k: real_mp3_roundtrip(w, kk, SC_SR)))

    for rate in (16000, 22050, 32000, 44100, 48000):
        run_condition(ber_benign["resample_hz"], rate, f"[benign] resample {rate}Hz",
                      lambda r=rate: ber_for_variant(lambda w, rr=r: resample_roundtrip(w, SC_SR, rr)))

    for snr in (10, 20, 30, 40):
        run_condition(ber_benign["noise_snr_db"], snr, f"[benign] noise {snr}dB SNR",
                      lambda s=snr: ber_for_variant(lambda w, ss=s: add_noise_np(w, ss)))

    for g in (0.7, 1.0, 1.3):
        run_condition(ber_benign["gain"], g, f"[benign] gain {g}",
                      lambda gg=g: ber_for_variant(lambda w, ggg=gg: apply_gain_np(w, ggg)))

    run_condition(ber_benign["eq"], "max_6db", "[benign] EQ (+-6dB)",
                  lambda: ber_for_variant(lambda w: eq_6db_np(w, SC_SR)))

    for n_iter in (0, 4, 16, 64):
        run_condition(ber_regen["griffinlim"], n_iter, f"[regen] Griffin-Lim n_iter={n_iter}",
                      lambda ni=n_iter: ber_for_variant(lambda w, nn=ni: griffinlim_regen_np(w, SC_SR, n_iter=nn)))

    if ber_regen.get("dac") is None:
        print(f"[{now()}]   [regen] loading DAC (44kHz variant)...")
        dac_model = load_dac()
        print(f"[{now()}]   [regen] DAC full-clip resynthesis...")
        ber_regen["dac"] = ber_for_variant(lambda w: dac_regen_np(dac_model, w, SC_SR))
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
            lambda w: musicgen_regen_np(mg_processor, mg_model, mg_sr, w, SC_SR))
        save_checkpoint(results)
    else:
        print(f"[{now()}]   [skip, already checkpointed] [regen] MusicGen: {ber_regen['musicgen']:.4f}")

    # --- Targeted detection-rate diagnostic pass ---
    # Only runs on the conditions listed in DIAGNOSE_CONDITIONS. Does NOT
    # touch or re-run anything already checkpointed above. Checkpointed
    # separately under results["detection_diagnostics"] so a BER already
    # on record is never silently overwritten by this pass.
    if "detection_diagnostics" not in results:
        results["detection_diagnostics"] = {}
    diag = results["detection_diagnostics"]

    for group, key, transform_fn in DIAGNOSE_CONDITIONS:
        diag_key = f"{group}_{key}" if key is not None else group
        if diag_key in diag:
            d = diag[diag_key]
            print(f"[{now()}]   [skip, already diagnosed] {diag_key}: "
                  f"BER={d['ber']:.4f}, detection_rate={d['detection_rate']:.4f}, "
                  f"mean_confidence={d['mean_confidence']}")
            continue

        if group == "dac":
            print(f"[{now()}]   [diagnose] loading DAC (44kHz variant)...")
            dac_model = load_dac()
            ber, det_rate, n_undet, mean_conf = ber_and_detection_for_variant(
                lambda w: dac_regen_np(dac_model, w, SC_SR))
            del dac_model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        elif group == "musicgen":
            if not INCLUDE_MUSICGEN_DIAGNOSIS:
                print(f"[{now()}]   [diagnose] SKIPPING MusicGen diagnosis "
                      f"(INCLUDE_MUSICGEN_DIAGNOSIS=False).")
                continue
            print(f"[{now()}]   [diagnose] loading MusicGen-small...")
            mg_processor, mg_model, mg_sr = load_musicgen()
            print(f"[{now()}]   [diagnose] MusicGen partial continuation (slow)...")
            ber, det_rate, n_undet, mean_conf = ber_and_detection_for_variant(
                lambda w: musicgen_regen_np(mg_processor, mg_model, mg_sr, w, SC_SR))
        else:
            print(f"[{now()}]   [diagnose] {diag_key}...")
            ber, det_rate, n_undet, mean_conf = ber_and_detection_for_variant(transform_fn)

        diag[diag_key] = {"ber": ber, "detection_rate": det_rate, "n_undetected": n_undet,
                           "mean_confidence": mean_conf, "n_clips": len(watermarked_list)}
        print(f"[{now()}]   [diagnose] {diag_key}: BER={ber:.4f}, detection_rate={det_rate:.4f} "
              f"({len(watermarked_list) - n_undet}/{len(watermarked_list)} clips detected), "
              f"mean_confidence={mean_conf}")
        save_checkpoint(results)

    results["n_clips"] = len(val_wavs)
    results["n_msg_bits"] = N_MSG_BYTES * 8
    results["operating_sample_rate_hz"] = SC_SR
    results["note"] = ("SilentCipher's message format is 5x8-bit (40 bits total), "
                        "higher capacity than AudioSeal/WavMark's 16 bits and our "
                        "own K=4 — a harder task, not an advantage chosen either way.")
    save_checkpoint(results)

    print(f"[{now()}] SilentCipher BER benign: {ber_benign}")
    print(f"[{now()}] SilentCipher BER regen: {ber_regen}")

    fig, ax = plt.subplots(figsize=(10, 5))
    benign_flat = [v for g in ber_benign.values() for v in g.values()]
    regen_flat = list(ber_regen["griffinlim"].values()) + [ber_regen["dac"], ber_regen["musicgen"]]
    ax.boxplot([benign_flat, regen_flat], labels=["Benign (18 settings)", "Regen (Griffin-Lim/DAC/MusicGen)"])
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="chance")
    ax.set_ylabel("BER (40-bit message)")
    ax.set_title(f"SilentCipher baseline — n={len(val_wavs)} held-out clips, 44.1kHz")
    ax.legend()
    fig.tight_layout()
    fig.savefig(LOCAL_FIG, dpi=300)
    plt.close(fig)
    copy_to_project(LOCAL_FIG, "fig_11_01_silentcipher_vs_ours.png")

    print(f"[{now()}] DONE. exp_baseline_silentcipher_results.json, "
          f"fig_11_01_silentcipher_vs_ours.png saved to PROJECT_DIR.")


if __name__ == "__main__":
    main()