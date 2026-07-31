# =============================================================================
# Program      : 09_baseline_audioseal.py
# Version      : 1.2
# Description  : Quantitative baseline comparison — AudioSeal.
#
#                Written in response to prescreening rejection of SPL-48154-
#                2026: "fails to comprehensively compare the watermarking
#                scheme to any baseline or related method." Table 1 in the
#                blueprint/paper was a QUALITATIVE positioning table only —
#                this script produces a real, quantitative one.
#
#                AudioSeal (San Roman et al., ICML 2024) is the only one of
#                the three related works (AudioSeal, DualMark,
#                SpeechVerifier) with a publicly released, pretrained
#                checkpoint (Hugging Face Hub, via the `audioseal` PyPI
#                package) — confirmed by direct search before writing this,
#                not assumed. DualMark and SpeechVerifier have no public
#                code/checkpoints as of this writing; that absence is
#                reported honestly in the paper rather than silently
#                dropped.
#
#                ENTIRELY INFERENCE-ONLY: no training, no backward passes,
#                anywhere in this script. AudioSeal embeds its watermark
#                additively in the waveform domain (watermarked = wav +
#                generator.get_watermark(wav, message)) — it does not touch
#                EnCodec at all, so this script never loads our own frozen
#                codec. Griffin-Lim, DAC, and MusicGen are likewise used
#                purely as forward-pass attack tools, exactly as in
#                04/05/06. This is lower-risk than every prior script in
#                this project for that reason — no cuDNN/LSTM backward
#                issue applies here, no straight-through estimator needed.
#
#                IMPORTANT CAVEAT, stated here so it isn't glossed over
#                later: AudioSeal's pretrained checkpoint carries a FIXED
#                16-bit message, not our K=4. This is not chosen to favor
#                either system — it is what the public checkpoint supports
#                — but it means AudioSeal is solving a harder capacity
#                problem than our K=4 configuration, which needs to be
#                stated plainly in the paper's comparison, not obscured.
#
# PRE-FLIGHT SELF-TEST: confirms the generator+detector round-trip on one
# clip with NO attack applied (should recover the embedded message near-
# perfectly) before running the full battery — validates the documented
# API actually behaves as documented in this environment.
#
# INPUT:
#                  PROJECT_DIR (/content/drive/MyDrive/paper/HyperFrag/):
#                    dataset_e0.h5   — from 01_e0_dataextract.py
#
# STEPS:
#                  Step 1  Download dataset_e0.h5, extract the SAME 105
#                          held-out clips (same seed/split) used by every
#                          other experiment in this project
#                  Step 2  Load AudioSeal generator + detector (pretrained,
#                          16-bit), self-test round-trip on one clip
#                  Step 3  Embed AudioSeal watermarks in all 105 clips
#                  Step 4  Score BER under the identical benign battery
#                          used in 03/05 (compression, resample, noise,
#                          gain, EQ) and the identical regeneration battery
#                          used in 04/06 (Griffin-Lim proxy, real DAC, real
#                          MusicGen partial continuation)
#                  Step 5  Save results in the same JSON structure as our
#                          own experiments, for direct table merging
#
# OUTPUT FILES : (stored at /content/drive/MyDrive/paper/HyperFrag/)
#                  exp_baseline_audioseal_results.json
#                      {ber_benign: {...}, ber_regen: {...}}
#                  fig_09_01_audioseal_vs_ours.png
#                      AudioSeal's BER vs. our Path A (robust) and Path B
#                      (fragile) results, same conditions, side by side
#
# GPU Required : Optional (inference-only; GPU used if available for speed,
#                nothing in this script requires it strictly)
# Dependencies : torch, torchaudio, audioseal, pydub, descript-audio-codec,
#                transformers, h5py, matplotlib, tqdm
#
# Change Log   :
#   v1.0  2026-07-27  Initial version
#   v1.1  2026-07-27  Added PROJECT_DIR checkpoint-and-resume, same fix as
#                      10_baseline_wavmark.py v1.1 — no functional change to
#                      results, just resilience against session disconnects.
#   v1.2  2026-07-28  Added a targeted detection-rate diagnostic pass for
#                      DAC and MusicGen (the two conditions with elevated
#                      BER: 0.685, 0.509). AudioSeal has no None/failure
#                      sentinel like WavMark's decoder — per its own README,
#                      "message will be a random tensor if the detector
#                      detects no watermarking," and the documented
#                      convention is a threshold on the detection
#                      probability itself (result > 0.5 = detected). This
#                      pass checks that threshold to distinguish "AudioSeal
#                      didn't detect anything" from "AudioSeal detected it
#                      confidently but decoded it wrong" — the latter would
#                      explain DAC's above-chance BER, which simple
#                      non-detection (implying near-chance BER) would not.
#                      Checkpointed separately under
#                      results["detection_diagnostics"]; does not touch or
#                      re-run any of the ~23 already-complete conditions.
#
# !pip install torch torchaudio audioseal pydub descript-audio-codec transformers h5py matplotlib tqdm
# =============================================================================

!pip install -q audioseal pydub descript-audio-codec transformers h5py matplotlib tqdm

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE} (this script is inference-only; GPU is a speed "
      f"convenience, not a requirement, unlike the training scripts in this project).")

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
VAL_FRACTION = 0.10
SEED = 20260716  # SAME seed as every other experiment — reproduces the identical 105-clip split
N_MSG_BITS = 16   # fixed by the pretrained AudioSeal checkpoint, not a choice made here

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
LOCAL_RESULTS = f"{LOCAL_SCRATCH}/exp_baseline_audioseal_results.json"
LOCAL_FIG = f"{LOCAL_SCRATCH}/fig_09_01_audioseal_vs_ours.png"


# --- Step 1: load the SAME held-out split as every other experiment --------
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
    print(f"[{now()}] {len(val_idx)} held-out clips (identical split to E1-E6).")
    return wavs[val_idx]


# --- Step 2: load AudioSeal + self-test -------------------------------------
def load_audioseal_and_selftest(sample_wav_np):
    print(f"[{now()}] Loading AudioSeal generator + detector (pretrained, 16-bit)...")
    from audioseal import AudioSeal
    generator = AudioSeal.load_generator("audioseal_wm_16bits").to(DEVICE)
    generator.eval()
    detector = AudioSeal.load_detector("audioseal_detector_16bits").to(DEVICE)
    detector.eval()

    print(f"[{now()}] Self-test: embed + detect on one clip, NO attack applied "
          f"(should recover the message almost perfectly)...")
    wav_t = torch.from_numpy(sample_wav_np).float().unsqueeze(0).unsqueeze(0).to(DEVICE)  # [1,1,T]
    msg = torch.randint(0, 2, (1, N_MSG_BITS), device=DEVICE)
    with torch.no_grad():
        watermark = generator.get_watermark(wav_t, message=msg)
        watermarked = wav_t + watermark
        result, decoded_msg = detector.detect_watermark(watermarked)
    print(f"[selftest] detection probability: {float(result):.4f} (expect close to 1.0)")
    decoded_bits = (decoded_msg > 0.5).int().squeeze().cpu()
    true_bits = msg.squeeze().cpu()
    n_wrong = (decoded_bits != true_bits).sum().item()
    print(f"[selftest] message bits wrong (no attack): {n_wrong}/{N_MSG_BITS} "
          f"(expect 0 or very close to it)")
    if n_wrong > N_MSG_BITS // 4:
        print(f"[selftest] [FATAL] more than a quarter of bits wrong with NO attack "
              f"applied at all — something about the embed/detect call is not "
              f"working as documented. Inspect before trusting the full run.")
        raise SystemExit("Self-test failed — AudioSeal round-trip did not recover "
                          "the message under zero attack.")
    print(f"[selftest] PASSED. Proceeding to the full evaluation battery.")
    return generator, detector


def match_length(wav_np_or_t, target_len):
    is_tensor = torch.is_tensor(wav_np_or_t)
    cur = wav_np_or_t.shape[-1]
    if cur == target_len:
        return wav_np_or_t
    if cur > target_len:
        return wav_np_or_t[..., :target_len]
    pad = target_len - cur
    if is_tensor:
        return F.pad(wav_np_or_t, (0, pad))
    return np.pad(wav_np_or_t, (0, pad))


# --- Benign transforms (real, non-differentiable — this script never needs
#     gradients, so these are plain numpy/torch inference-time transforms) --
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


def resample_roundtrip(wav_t, target_rate):
    down = torchaudio.functional.resample(wav_t, SR, target_rate)
    return torchaudio.functional.resample(down, target_rate, SR)


def add_noise(wav_t, snr_db):
    sig_power = wav_t.pow(2).mean(dim=-1, keepdim=True)
    noise_power = sig_power / (10 ** (snr_db / 10))
    return wav_t + torch.randn_like(wav_t) * noise_power.sqrt()


def apply_gain(wav_t, g):
    return wav_t * g


def eq_6db(wav_t, n_fft=1024, hop=256, max_db=6.0, n_bands=6):
    window = torch.hann_window(n_fft, device=wav_t.device)
    spec = torch.stft(wav_t, n_fft=n_fft, hop_length=hop, window=window, return_complex=True)
    n_bins = spec.shape[-2]
    band_gains_db = (torch.rand(wav_t.shape[0], 1, n_bands, device=wav_t.device) * 2 - 1) * max_db
    gain_curve_db = F.interpolate(band_gains_db, size=n_bins, mode="linear", align_corners=True)
    gain_curve_db = gain_curve_db.squeeze(1).unsqueeze(-1)
    gain_lin = 10 ** (gain_curve_db / 20)
    return torch.istft(spec * gain_lin, n_fft=n_fft, hop_length=hop, window=window, length=wav_t.shape[-1])


# --- Regeneration attacks (Griffin-Lim proxy, real DAC, real MusicGen) -----
def griffinlim_regen(wav_t, n_fft=1024, hop=256, n_iter=32):
    window = torch.hann_window(n_fft, device=wav_t.device)
    spec = torch.stft(wav_t, n_fft=n_fft, hop_length=hop, window=window, return_complex=True)
    mag = spec.abs()
    return torchaudio.functional.griffinlim(
        mag, window=window, n_fft=n_fft, hop_length=hop, win_length=n_fft,
        power=1.0, n_iter=n_iter, momentum=0.99, length=wav_t.shape[-1], rand_init=True,
    )


def load_dac():
    import dac
    dac_model_path = dac.utils.download(model_type="24khz")
    dac_model = dac.DAC.load(dac_model_path).to(DEVICE)
    dac_model.eval()
    return dac_model


def dac_regen(dac_model, wav_np):
    wav_t = torch.from_numpy(wav_np).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        x = dac_model.preprocess(wav_t, SR)
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


def musicgen_regen(processor, model, mg_sr, wav_np):
    prompt_samples = int(PROMPT_SECONDS * SR)
    prompt_np = wav_np[:prompt_samples]
    prompt_t = torch.from_numpy(prompt_np).float().unsqueeze(0)
    prompt_resampled = torchaudio.functional.resample(prompt_t, SR, mg_sr).squeeze(0).numpy()
    additional_seconds = TOTAL_GEN_SECONDS - PROMPT_SECONDS
    max_new_tokens = int(additional_seconds * MUSICGEN_FRAME_RATE)
    inputs = processor(audio=prompt_resampled, sampling_rate=mg_sr, text=["music"],
                        padding=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = model.generate(**inputs, do_sample=True, guidance_scale=3, max_new_tokens=max_new_tokens)
    out_wav = out[0, 0].cpu()
    out_24k = torchaudio.functional.resample(out_wav.unsqueeze(0), mg_sr, SR).squeeze(0).numpy()
    return match_length(out_24k, len(wav_np))


# --- Main ---------------------------------------------------------------
RESULTS_REMOTE = "exp_baseline_audioseal_results.json"


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
    val_wavs = load_val_clips()
    generator, detector = load_audioseal_and_selftest(val_wavs[0])
    results = load_checkpoint()
    ber_benign, ber_regen = results["ber_benign"], results["ber_regen"]

    print(f"[{now()}] Embedding AudioSeal watermarks in all {len(val_wavs)} clips "
          f"(re-embedded fresh each run — see 10_baseline_wavmark.py's main() "
          f"for why this doesn't need separate checkpointing)...")
    true_msgs = []
    watermarked_list = []
    with torch.no_grad():
        for i in tqdm(range(len(val_wavs)), desc="audioseal embed"):
            wav_t = torch.from_numpy(val_wavs[i]).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
            msg = torch.randint(0, 2, (1, N_MSG_BITS), device=DEVICE)
            watermark = generator.get_watermark(wav_t, message=msg)
            watermarked = (wav_t + watermark).squeeze(0).squeeze(0).cpu().numpy()
            watermarked_list.append(watermarked)
            true_msgs.append(msg.squeeze().cpu().numpy())
    true_msgs = np.stack(true_msgs)  # [N, 16]

    def ber_for_variant(variant_fn):
        n_wrong, n_total = 0, 0
        for i in range(len(watermarked_list)):
            variant = variant_fn(watermarked_list[i])
            variant = match_length(variant, len(watermarked_list[i]))
            variant_t = torch.from_numpy(variant).float().unsqueeze(0).unsqueeze(0).to(DEVICE) \
                if isinstance(variant, np.ndarray) else variant.unsqueeze(0).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                _, decoded_msg = detector.detect_watermark(variant_t)
            decoded_bits = (decoded_msg > 0.5).int().squeeze().cpu().numpy()
            n_wrong += (decoded_bits != true_msgs[i]).sum()
            n_total += N_MSG_BITS
        return float(n_wrong / n_total)

    def ber_and_detection_for_variant(variant_fn):
        """Same as ber_for_variant, but also tracks AudioSeal's OWN detection
        confidence. Unlike WavMark, AudioSeal has no None/failure sentinel —
        per its README, 'message will be a random tensor if the detector
        detects no watermarking,' and the paper authors' own documented
        convention is a threshold on the detection probability itself:
        result > 0.5 = detected. A clip with result < 0.5 is AudioSeal's own
        analog of WavMark's None case. Used only for the targeted diagnostic
        pass below, not the main battery."""
        n_wrong, n_total, n_undetected = 0, 0, 0
        for i in range(len(watermarked_list)):
            variant = variant_fn(watermarked_list[i])
            variant = match_length(variant, len(watermarked_list[i]))
            variant_t = torch.from_numpy(variant).float().unsqueeze(0).unsqueeze(0).to(DEVICE) \
                if isinstance(variant, np.ndarray) else variant.unsqueeze(0).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                result, decoded_msg = detector.detect_watermark(variant_t)
            if float(result) < 0.5:
                n_undetected += 1
            decoded_bits = (decoded_msg > 0.5).int().squeeze().cpu().numpy()
            n_wrong += (decoded_bits != true_msgs[i]).sum()
            n_total += N_MSG_BITS
        ber = float(n_wrong / n_total)
        detection_rate = float((len(watermarked_list) - n_undetected) / len(watermarked_list))
        return ber, detection_rate, n_undetected

    def ber_for_torch_variant(transform_fn):
        n_wrong, n_total = 0, 0
        for i in range(len(watermarked_list)):
            wav_t = torch.from_numpy(watermarked_list[i]).float().unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                variant = transform_fn(wav_t)
            variant = match_length(variant, wav_t.shape[-1])
            with torch.no_grad():
                _, decoded_msg = detector.detect_watermark(variant.unsqueeze(0))
            decoded_bits = (decoded_msg > 0.5).int().squeeze().cpu().numpy()
            n_wrong += (decoded_bits != true_msgs[i]).sum()
            n_total += N_MSG_BITS
        return float(n_wrong / n_total)

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
                      lambda k=kbps: ber_for_variant(lambda w, kk=k: real_mp3_roundtrip(w, kk)))

    for rate in (16000, 22050, 32000, 44100, 48000):
        run_condition(ber_benign["resample_hz"], rate, f"[benign] resample {rate}Hz",
                      lambda r=rate: ber_for_torch_variant(lambda t, rr=r: resample_roundtrip(t, rr)))

    for snr in (10, 20, 30, 40):
        run_condition(ber_benign["noise_snr_db"], snr, f"[benign] noise {snr}dB SNR",
                      lambda s=snr: ber_for_torch_variant(lambda t, ss=s: add_noise(t, ss)))

    for g in (0.7, 1.0, 1.3):
        run_condition(ber_benign["gain"], g, f"[benign] gain {g}",
                      lambda gg=g: ber_for_torch_variant(lambda t, ggg=gg: apply_gain(t, ggg)))

    run_condition(ber_benign["eq"], "max_6db", "[benign] EQ (+-6dB)",
                  lambda: ber_for_torch_variant(eq_6db))

    for n_iter in (0, 4, 16, 64):
        run_condition(ber_regen["griffinlim"], n_iter, f"[regen] Griffin-Lim n_iter={n_iter}",
                      lambda ni=n_iter: ber_for_torch_variant(lambda t, nn=ni: griffinlim_regen(t, n_iter=nn)))

    if ber_regen.get("dac") is None:
        print(f"[{now()}]   [regen] loading DAC...")
        dac_model = load_dac()
        print(f"[{now()}]   [regen] DAC full-clip resynthesis...")
        ber_regen["dac"] = ber_for_variant(lambda w: dac_regen(dac_model, w))
        save_checkpoint(results)
        del dac_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    else:
        print(f"[{now()}]   [skip, already checkpointed] [regen] DAC: {ber_regen['dac']:.4f}")

    if ber_regen.get("musicgen") is None:
        print(f"[{now()}]   [regen] loading MusicGen-small...")
        mg_processor, mg_model, mg_sr = load_musicgen()
        print(f"[{now()}]   [regen] MusicGen partial continuation (slow)...")
        ber_regen["musicgen"] = ber_for_variant(lambda w: musicgen_regen(mg_processor, mg_model, mg_sr, w))
        save_checkpoint(results)
    else:
        print(f"[{now()}]   [skip, already checkpointed] [regen] MusicGen: {ber_regen['musicgen']:.4f}")

    # --- Targeted detection-rate diagnostic pass ---
    # Only DAC and MusicGen — the two conditions with elevated BER (0.685,
    # 0.509). Does NOT touch or re-run any of the ~23 already-checkpointed
    # conditions above. Checkpointed separately under
    # results["detection_diagnostics"], same pattern as
    # 10_baseline_wavmark.py's v1.2 fix — a rerun of this script skips
    # anything already in either results["ber_benign"]/["ber_regen"] or
    # results["detection_diagnostics"] and resumes only what's missing.
    if "detection_diagnostics" not in results:
        results["detection_diagnostics"] = {}
    diag = results["detection_diagnostics"]

    if "dac" not in diag:
        print(f"[{now()}]   [diagnose] loading DAC...")
        dac_model = load_dac()
        print(f"[{now()}]   [diagnose] DAC full-clip resynthesis...")
        ber, det_rate, n_undet = ber_and_detection_for_variant(lambda w: dac_regen(dac_model, w))
        del dac_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        diag["dac"] = {"ber": ber, "detection_rate": det_rate, "n_undetected": n_undet,
                        "n_clips": len(watermarked_list)}
        print(f"[{now()}]   [diagnose] dac: BER={ber:.4f}, detection_rate={det_rate:.4f} "
              f"({len(watermarked_list) - n_undet}/{len(watermarked_list)} clips with result>0.5)")
        save_checkpoint(results)
    else:
        d = diag["dac"]
        print(f"[{now()}]   [skip, already diagnosed] dac: BER={d['ber']:.4f}, "
              f"detection_rate={d['detection_rate']:.4f}")

    if "musicgen" not in diag:
        print(f"[{now()}]   [diagnose] loading MusicGen-small...")
        mg_processor, mg_model, mg_sr = load_musicgen()
        print(f"[{now()}]   [diagnose] MusicGen partial continuation (slow)...")
        ber, det_rate, n_undet = ber_and_detection_for_variant(
            lambda w: musicgen_regen(mg_processor, mg_model, mg_sr, w))
        diag["musicgen"] = {"ber": ber, "detection_rate": det_rate, "n_undetected": n_undet,
                             "n_clips": len(watermarked_list)}
        print(f"[{now()}]   [diagnose] musicgen: BER={ber:.4f}, detection_rate={det_rate:.4f} "
              f"({len(watermarked_list) - n_undet}/{len(watermarked_list)} clips with result>0.5)")
        save_checkpoint(results)
    else:
        d = diag["musicgen"]
        print(f"[{now()}]   [skip, already diagnosed] musicgen: BER={d['ber']:.4f}, "
              f"detection_rate={d['detection_rate']:.4f}")

    results["n_clips"] = len(val_wavs)
    results["n_msg_bits"] = N_MSG_BITS
    results["note"] = ("AudioSeal's checkpoint carries a fixed 16-bit message, vs. our "
                        "K=4 — a harder capacity task, not a chosen advantage either way.")
    save_checkpoint(results)

    print(f"[{now()}] AudioSeal BER benign: {ber_benign}")
    print(f"[{now()}] AudioSeal BER regen: {ber_regen}")

    fig, ax = plt.subplots(figsize=(10, 5))
    benign_flat = [v for g in ber_benign.values() for v in g.values()]
    regen_flat = list(ber_regen["griffinlim"].values()) + [ber_regen["dac"], ber_regen["musicgen"]]
    ax.boxplot([benign_flat, regen_flat], labels=["Benign (18 settings)", "Regen (Griffin-Lim/DAC/MusicGen)"])
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="chance")
    ax.set_ylabel("BER (16-bit message)")
    ax.set_title(f"AudioSeal baseline — n={len(val_wavs)} held-out clips, "
                 f"same conditions as our Path A/Path B evaluation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(LOCAL_FIG, dpi=300)
    plt.close(fig)
    copy_to_project(LOCAL_FIG, "fig_09_01_audioseal_vs_ours.png")

    print(f"[{now()}] DONE. exp_baseline_audioseal_results.json, "
          f"fig_09_01_audioseal_vs_ours.png saved to PROJECT_DIR.")


if __name__ == "__main__":
    main()