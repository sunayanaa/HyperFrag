# =============================================================================
# Program      : 12_baseline_naive_dual_stacking.py
# Version      : 1.1
# Description  : Naive dual-watermark baseline — the direct answer to the
#                "appears to be a combination of existing work" reviewer
#                comment. Rather than assert that our jointly-trained,
#                channel-sliced design is better than a naive combination
#                of two independent systems, this measures it.
#
#                METHOD: AudioSeal (robust, waveform-domain, additive) is
#                applied FIRST to each clean clip, then our ALREADY-TRAINED
#                E2 fragile-path checkpoint (04_e2_fragile_path.py — Path B
#                alone, channels [32:64], K=4) is stacked on top, applied
#                completely independently — it has no knowledge AudioSeal's
#                watermark exists, exactly as two off-the-shelf systems
#                combined naively would be. This ordering (robust ID first,
#                fragile tamper-evidence layer second) mirrors a plausible
#                real deployment: a platform tags ownership at upload, then
#                a later ingestion step adds tamper-evidence.
#
#                WHY THIS ORDERING MATTERS, STATED EXPLICITLY: applying our
#                fragile path means running the audio through OUR frozen
#                EnCodec's full encode-quantize-decode cycle. That cycle is
#                a real, lossy transformation of WHATEVER audio it's given
#                — including AudioSeal's careful waveform-domain
#                perturbation, which was never trained with any codec
#                bottleneck in its own optimization loop. We expect this to
#                be a genuine, structural source of interference distinct
#                from anything in our own system, where both paths are
#                trained end-to-end THROUGH the same codec bottleneck from
#                the start. This is a stated hypothesis for interpreting
#                results, not an assumption baked into the measurement
#                itself — the script measures what actually happens.
#
#                THREE THINGS MEASURED, each against a number we already
#                have from an existing experiment:
#                  1. AudioSeal's own BER on the STACKED audio, vs. its
#                     standalone BER from 09_baseline_audioseal.py (does
#                     our fragile layer disturb AudioSeal's signal?)
#                  2. Our fragile path's BER on the STACKED audio, vs. its
#                     standalone BER from 04_e2_fragile_path.py (does
#                     AudioSeal's watermark disturb OUR path's benign-
#                     survival / regeneration-collapse behavior?)
#                  3. Perceptual quality of the doubly-watermarked audio
#                     (PESQ/ViSQOL/STFT loss vs. the original clean clip),
#                     vs. E3's JOINTLY-trained quality numbers (ViSQOL
#                     4.40->3.96) — the direct "naive stacking degrades
#                     quality more than integrated joint training" test.
#
#                ENTIRELY INFERENCE-ONLY: AudioSeal is pretrained, and E2's
#                checkpoint is loaded and used exactly as already trained
#                — no training or backward passes anywhere in this script.
#
# PRE-FLIGHT SELF-TEST: (a) AudioSeal round-trip on one clip, same check as
# 09; (b) our E2 checkpoint's embed/extract round-trip on one CLEAN
# (unstacked) clip, confirming it reproduces its own known standalone
# behavior (near-zero BER) before being combined with AudioSeal at all —
# isolates whether any later interference is real, not an artifact of a
# broken checkpoint load.
#
# INPUT:
#                  PROJECT_DIR (/content/drive/MyDrive/paper/HyperFrag/):
#                    dataset_e0.h5              — from 01_e0_dataextract.py
#                    e2_fragile_checkpoint.pth  — from 04_e2_fragile_path.py
#                                                 (REQUIRED)
#
# STEPS:
#                  Step 1  Download dataset_e0.h5, extract the SAME 105
#                          held-out clips used throughout this project
#                  Step 2  Load AudioSeal (pretrained), self-test
#                  Step 3  Load frozen EnCodec + E2's trained checkpoint,
#                          self-test on a clean, unstacked clip
#                  Step 4  For all 105 clips: embed AudioSeal first, then
#                          stack our fragile path on top
#                  Step 5  Score AudioSeal's BER and our fragile path's BER
#                          separately, on the stacked audio, under the same
#                          benign+regen battery used throughout
#                  Step 6  Measure perceptual quality of the stacked result
#                          vs. the clean original
#
# OUTPUT FILES : (stored at /content/drive/MyDrive/paper/HyperFrag/)
#                  exp_baseline_naive_stacking_results.json
#                  fig_12_01_naive_stacking_vs_joint.png
#                      AudioSeal BER (standalone vs. stacked), our fragile
#                      path BER (standalone vs. stacked), and perceptual
#                      quality (E3 joint vs. naive stacking) side by side
#
# GPU Required : Optional (inference-only)
# Dependencies : torch, torchaudio, audioseal, encodec, pydub, pesq,
#                visqol-python, descript-audio-codec, transformers, h5py,
#                matplotlib, tqdm
#
# Change Log   :
#   v1.0  2026-07-27  Initial version
#   v1.1  2026-07-27  Added PROJECT_DIR checkpoint-and-resume, same fix as 09/10/11
#                      v1.1 — extended here to cover both battery trees
#                      (AudioSeal-on-stacked, our-fragile-on-stacked) and
#                      the perceptual-quality section, each independently
#                      resumable.
#
# !pip install torch torchaudio audioseal encodec pydub pesq visqol-python descript-audio-codec transformers h5py matplotlib tqdm
# =============================================================================

!pip install -q audioseal encodec pydub pesq visqol-python descript-audio-codec transformers h5py matplotlib tqdm

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE} (inference-only — AudioSeal is pretrained, "
      f"E2's checkpoint is loaded and used as-is, no training here).")

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

SR = 24000
TARGET_BANDWIDTH_KBPS = 6.0
VAL_FRACTION = 0.10
SEED = 20260716   # SAME seed — identical 105-clip split as every other experiment
KEY_BITS = 4        # matches E2's trained checkpoint
N_AS_MSG_BITS = 16   # AudioSeal's fixed message size

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
LOCAL_E2_CKPT = f"{LOCAL_SCRATCH}/e2_fragile_checkpoint.pth"
LOCAL_RESULTS = f"{LOCAL_SCRATCH}/exp_baseline_naive_stacking_results.json"
LOCAL_FIG = f"{LOCAL_SCRATCH}/fig_12_01_naive_stacking_vs_joint.png"


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
    print(f"[{now()}] {len(val_idx)} held-out clips (identical split to E1-E6, "
          f"09/10/11).")
    return wavs[val_idx]


# --- AudioSeal (robust layer, applied first) --------------------------------
def load_audioseal_and_selftest(sample_wav_np):
    print(f"[{now()}] Loading AudioSeal generator + detector (pretrained, 16-bit)...")
    from audioseal import AudioSeal
    generator = AudioSeal.load_generator("audioseal_wm_16bits").to(DEVICE)
    generator.eval()
    detector = AudioSeal.load_detector("audioseal_detector_16bits").to(DEVICE)
    detector.eval()

    wav_t = torch.from_numpy(sample_wav_np).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
    msg = torch.randint(0, 2, (1, N_AS_MSG_BITS), device=DEVICE)
    with torch.no_grad():
        watermark = generator.get_watermark(wav_t, message=msg)
        watermarked = wav_t + watermark
        result, decoded_msg = detector.detect_watermark(watermarked)
    decoded_bits = (decoded_msg > 0.5).int().squeeze().cpu()
    n_wrong = (decoded_bits != msg.squeeze().cpu()).sum().item()
    print(f"[selftest] AudioSeal: detection probability {float(result):.4f}, "
          f"bits wrong (no attack) {n_wrong}/{N_AS_MSG_BITS}")
    if n_wrong > N_AS_MSG_BITS // 4:
        raise SystemExit("Self-test failed — AudioSeal round-trip did not recover the message.")
    print(f"[selftest] AudioSeal PASSED.")
    return generator, detector


# --- Frozen EnCodec + E2's trained fragile-path checkpoint ------------------
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


def embed_fragile(model, wav_batch_1ch, hypernet, key_bits, d_start, d_end, frame_rate, bandwidth):
    with torch.no_grad():
        raw_emb = model.encoder(wav_batch_1ch)
        gamma, beta = hypernet(key_bits)
        gamma, beta = gamma.unsqueeze(-1), beta.unsqueeze(-1)
        emb_fragile_part = raw_emb[:, d_start:d_end, :] * gamma + beta
        emb_mod = torch.cat([raw_emb[:, :d_start, :], emb_fragile_part, raw_emb[:, d_end:, :]], dim=1)
        qres = model.quantizer(emb_mod, frame_rate, bandwidth)
        return model.decoder(qres.quantized)


def load_encodec_and_e2_and_selftest(sample_wav_np):
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
    d_start = d_total // 4
    d_end = 2 * (d_total // 4)
    print(f"[{now()}] D_total={d_total}, Path B channels [{d_start}:{d_end}] "
          f"(matches 04_e2_fragile_path.py exactly).")

    print(f"[{now()}] Downloading e2_fragile_checkpoint.pth from PROJECT_DIR (REQUIRED)...")
    if not copy_from_project("e2_fragile_checkpoint.pth", LOCAL_E2_CKPT):
        raise SystemExit("e2_fragile_checkpoint.pth not found in PROJECT_DIR — run "
                          "04_e2_fragile_path.py to completion first. This script "
                          "evaluates the trained E2 model, it does not train one.")
    ckpt = torch.load(LOCAL_E2_CKPT, map_location=DEVICE)
    hypernet = HyperNet(KEY_BITS, d_end - d_start).to(DEVICE)
    extractor = Extractor(KEY_BITS).to(DEVICE)
    hypernet.load_state_dict(ckpt["hypernet_state"])
    extractor.load_state_dict(ckpt["extractor_state"])
    hypernet.eval()
    extractor.eval()
    print(f"[{now()}] Loaded E2 checkpoint from epoch {ckpt.get('epoch')}.")

    print(f"[{now()}] Self-test: E2's embed/extract round-trip on a CLEAN, "
          f"unstacked clip (should reproduce its known standalone behavior, "
          f"near-zero BER) before combining with AudioSeal at all...")
    wav_t = torch.from_numpy(sample_wav_np).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
    key_pm1 = (torch.randint(0, 2, (1, KEY_BITS), device=DEVICE) * 2 - 1).float()
    target = (key_pm1 > 0).float()
    x_wm = embed_fragile(model, wav_t, hypernet, key_pm1, d_start, d_end, frame_rate, bandwidth)
    with torch.no_grad():
        logits = extractor(x_wm.squeeze(1))
    pred = (logits > 0).float()
    n_wrong = int((pred != target).sum().item())
    print(f"[selftest] E2 checkpoint: bits wrong on clean unstacked clip "
          f"{n_wrong}/{KEY_BITS} (expect 0)")
    if n_wrong > 0:
        print(f"[selftest] [WARNING] E2's own checkpoint doesn't perfectly reproduce "
              f"clean-signal recovery on this single sample — E2's full validation "
              f"set BER was reported as ~0 in 04_e2_fragile_path.py, so a single-clip "
              f"miss here is not necessarily fatal, but worth noting before "
              f"attributing any later interference entirely to AudioSeal.")
    else:
        print(f"[selftest] E2 checkpoint PASSED.")

    return model, hypernet, extractor, d_start, d_end, frame_rate, bandwidth


def match_length(arr, target_len):
    is_tensor = torch.is_tensor(arr)
    cur = arr.shape[-1]
    if cur == target_len:
        return arr
    if cur > target_len:
        return arr[..., :target_len]
    pad = target_len - cur
    return F.pad(arr, (0, pad)) if is_tensor else np.pad(arr, (0, pad))


# --- Benign + regen battery (identical to 09/10/11) -------------------------
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


def resample_roundtrip_np(wav_np, target_rate):
    t = torch.from_numpy(wav_np).float().unsqueeze(0)
    down = torchaudio.functional.resample(t, SR, target_rate)
    return torchaudio.functional.resample(down, target_rate, SR).squeeze(0).numpy()


def add_noise_np(wav_np, snr_db):
    sig_power = np.mean(wav_np ** 2)
    noise_power = sig_power / (10 ** (snr_db / 10))
    return wav_np + np.random.randn(*wav_np.shape).astype(np.float32) * np.sqrt(noise_power)


def apply_gain_np(wav_np, g):
    return wav_np * g


def eq_6db_np(wav_np, n_fft=1024, hop=256, max_db=6.0, n_bands=6):
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


def griffinlim_regen_np(wav_np, n_fft=1024, hop=256, n_iter=32):
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
    dac_model_path = dac.utils.download(model_type="24khz")
    dac_model = dac.DAC.load(dac_model_path).to(DEVICE)
    dac_model.eval()
    return dac_model


def dac_regen_np(dac_model, wav_np):
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


def musicgen_regen_np(processor, model, mg_sr, wav_np):
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


# --- Perceptual quality (same as 02/05) -------------------------------------
def resample_np_generic(wav_np, orig_sr, new_sr):
    t = torch.from_numpy(wav_np).float().unsqueeze(0)
    return torchaudio.functional.resample(t, orig_sr, new_sr).squeeze(0).numpy()


def compute_pesq_wb(ref_np, deg_np):
    from pesq import pesq as pesq_fn
    ref_16k = resample_np_generic(ref_np, SR, 16000)
    deg_16k = resample_np_generic(deg_np, SR, 16000)
    return float(pesq_fn(16000, ref_16k, deg_16k, "wb"))


def compute_visqol(api, ref_np, deg_np):
    ref_48k = resample_np_generic(ref_np, SR, 48000)
    deg_48k = resample_np_generic(deg_np, SR, 48000)
    result = api.measure_from_arrays(ref_48k, deg_48k, sample_rate=48000)
    return float(result.moslqo)


def multi_res_stft_loss(ref_np, deg_np, resolutions=((1024, 120, 600), (2048, 240, 1200), (512, 50, 240))):
    ref_t = torch.from_numpy(ref_np).float().unsqueeze(0)
    deg_t = torch.from_numpy(deg_np).float().unsqueeze(0)
    total = 0.0
    for n_fft, hop, win in resolutions:
        window = torch.hann_window(win)
        ref_spec = torch.stft(ref_t, n_fft=n_fft, hop_length=hop, win_length=win, window=window, return_complex=True)
        deg_spec = torch.stft(deg_t, n_fft=n_fft, hop_length=hop, win_length=win, window=window, return_complex=True)
        ref_mag = torch.clamp(ref_spec.abs(), min=1e-7)
        deg_mag = torch.clamp(deg_spec.abs(), min=1e-7)
        sc = torch.norm(ref_mag - deg_mag, p="fro") / torch.norm(ref_mag, p="fro")
        mag = torch.mean(torch.abs(torch.log(ref_mag) - torch.log(deg_mag)))
        total += (sc + mag).item()
    return total / len(resolutions)


# --- Main ---------------------------------------------------------------
RESULTS_REMOTE = "exp_baseline_naive_stacking_results.json"


def fresh_battery():
    return {"compression_kbps": {}, "resample_hz": {}, "noise_snr_db": {}, "gain": {}, "eq": {}}


def load_checkpoint():
    if copy_from_project(RESULTS_REMOTE, LOCAL_RESULTS, skip_if_exists=False):
        with open(LOCAL_RESULTS) as f:
            results = json.load(f)
        n_done = sum(
            len(v) if isinstance(v, dict) else (1 if v is not None else 0)
            for group in ("audioseal_ber_benign_stacked", "audioseal_ber_regen_stacked",
                          "fragile_ber_benign_stacked", "fragile_ber_regen_stacked")
            for v in results.get(group, {}).values())
        print(f"[{now()}] Found existing checkpoint in PROJECT_DIR with {n_done} conditions "
              f"already scored — resuming, not restarting from scratch.")
        return results
    print(f"[{now()}] No existing checkpoint found — starting fresh.")
    return {
        "audioseal_ber_benign_stacked": fresh_battery(),
        "fragile_ber_benign_stacked": fresh_battery(),
        "audioseal_ber_regen_stacked": {"griffinlim": {}, "dac": None, "musicgen": None},
        "fragile_ber_regen_stacked": {"griffinlim": {}, "dac": None, "musicgen": None},
        "perceptual_stacked_vs_clean": None,
    }


def save_checkpoint(results):
    with open(LOCAL_RESULTS, "w") as f:
        json.dump(results, f, indent=2)
    copy_to_project(LOCAL_RESULTS, RESULTS_REMOTE)


def main():
    val_wavs = load_val_clips()
    as_generator, as_detector = load_audioseal_and_selftest(val_wavs[0])
    (model, hypernet, extractor, d_start, d_end,
     frame_rate, bandwidth) = load_encodec_and_e2_and_selftest(val_wavs[0])
    results = load_checkpoint()

    print(f"[{now()}] Stacking: AudioSeal first, our fragile path second, "
          f"on all {len(val_wavs)} clips (re-stacked fresh each run — see "
          f"10_baseline_wavmark.py's main() for why this doesn't need "
          f"separate checkpointing; only the slow per-condition scoring below does)...")
    true_as_msgs, true_frag_keys = [], []
    stacked_list = []
    with torch.no_grad():
        for i in tqdm(range(len(val_wavs)), desc="naive stacking"):
            clean_t = torch.from_numpy(val_wavs[i]).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
            as_msg = torch.randint(0, 2, (1, N_AS_MSG_BITS), device=DEVICE)
            watermark = as_generator.get_watermark(clean_t, message=as_msg)
            x_audioseal = clean_t + watermark

            frag_key_pm1 = (torch.randint(0, 2, (1, KEY_BITS), device=DEVICE) * 2 - 1).float()
            x_stacked = embed_fragile(model, x_audioseal, hypernet, frag_key_pm1,
                                       d_start, d_end, frame_rate, bandwidth)

            stacked_list.append(x_stacked.squeeze().cpu().numpy())
            true_as_msgs.append(as_msg.squeeze().cpu().numpy())
            true_frag_keys.append((frag_key_pm1 > 0).float().squeeze().cpu().numpy())
    true_as_msgs = np.stack(true_as_msgs)
    true_frag_keys = np.stack(true_frag_keys)
    orig_np = val_wavs

    def as_ber_for_variant(variant_fn):
        n_wrong, n_total = 0, 0
        for i in range(len(stacked_list)):
            variant = variant_fn(stacked_list[i])
            variant = match_length(variant, len(stacked_list[i]))
            variant_t = torch.from_numpy(variant).float().unsqueeze(0).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                _, decoded_msg = as_detector.detect_watermark(variant_t)
            decoded_bits = (decoded_msg > 0.5).int().squeeze().cpu().numpy()
            n_wrong += (decoded_bits != true_as_msgs[i]).sum()
            n_total += N_AS_MSG_BITS
        return float(n_wrong / n_total)

    def frag_ber_for_variant(variant_fn):
        n_wrong, n_total = 0, 0
        for i in range(len(stacked_list)):
            variant = variant_fn(stacked_list[i])
            variant = match_length(variant, len(stacked_list[i]))
            variant_t = torch.from_numpy(variant).float().unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logits = extractor(variant_t)
            pred = (logits > 0).float().squeeze().cpu().numpy()
            n_wrong += (pred != true_frag_keys[i]).sum()
            n_total += KEY_BITS
        return float(n_wrong / n_total)

    def run_condition(group_dict, key, label, compute_fn):
        key = str(key)
        if key in group_dict and group_dict[key] is not None:
            print(f"[{now()}]   [skip, already checkpointed] {label}: {group_dict[key]:.4f}")
            return
        print(f"[{now()}]   {label}...")
        group_dict[key] = compute_fn()
        save_checkpoint(results)

    def run_benign_battery(battery_key, ber_fn, tag):
        battery = results[battery_key]
        for kbps in (32, 64, 128, 192, 320):
            run_condition(battery["compression_kbps"], kbps, f"[{tag}] compression {kbps}kbps",
                          lambda k=kbps: ber_fn(lambda w, kk=k: real_mp3_roundtrip(w, kk)))
        for rate in (16000, 22050, 32000, 44100, 48000):
            run_condition(battery["resample_hz"], rate, f"[{tag}] resample {rate}Hz",
                          lambda r=rate: ber_fn(lambda w, rr=r: resample_roundtrip_np(w, rr)))
        for snr in (10, 20, 30, 40):
            run_condition(battery["noise_snr_db"], snr, f"[{tag}] noise {snr}dB SNR",
                          lambda s=snr: ber_fn(lambda w, ss=s: add_noise_np(w, ss)))
        for g in (0.7, 1.0, 1.3):
            run_condition(battery["gain"], g, f"[{tag}] gain {g}",
                          lambda gg=g: ber_fn(lambda w, ggg=gg: apply_gain_np(w, ggg)))
        run_condition(battery["eq"], "max_6db", f"[{tag}] EQ (+-6dB)", lambda: ber_fn(eq_6db_np))

    print(f"[{now()}] Scoring AudioSeal's BER on the STACKED audio (benign battery)...")
    run_benign_battery("audioseal_ber_benign_stacked", as_ber_for_variant, "AudioSeal benign")
    print(f"[{now()}] Scoring our fragile path's BER on the STACKED audio (benign battery)...")
    run_benign_battery("fragile_ber_benign_stacked", frag_ber_for_variant, "fragile benign")

    as_regen = results["audioseal_ber_regen_stacked"]
    frag_regen = results["fragile_ber_regen_stacked"]
    for n_iter in (0, 4, 16, 64):
        run_condition(as_regen["griffinlim"], n_iter, f"[AudioSeal regen] Griffin-Lim n_iter={n_iter}",
                      lambda ni=n_iter: as_ber_for_variant(lambda w, nn=ni: griffinlim_regen_np(w, n_iter=nn)))
        run_condition(frag_regen["griffinlim"], n_iter, f"[fragile regen] Griffin-Lim n_iter={n_iter}",
                      lambda ni=n_iter: frag_ber_for_variant(lambda w, nn=ni: griffinlim_regen_np(w, n_iter=nn)))

    if as_regen.get("dac") is None or frag_regen.get("dac") is None:
        print(f"[{now()}] Loading DAC...")
        dac_model = load_dac()
        print(f"[{now()}] DAC full-clip resynthesis...")
        if as_regen.get("dac") is None:
            as_regen["dac"] = as_ber_for_variant(lambda w: dac_regen_np(dac_model, w))
            save_checkpoint(results)
        if frag_regen.get("dac") is None:
            frag_regen["dac"] = frag_ber_for_variant(lambda w: dac_regen_np(dac_model, w))
            save_checkpoint(results)
        del dac_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    else:
        print(f"[{now()}]   [skip, already checkpointed] DAC: AudioSeal={as_regen['dac']:.4f}, "
              f"fragile={frag_regen['dac']:.4f}")

    if as_regen.get("musicgen") is None or frag_regen.get("musicgen") is None:
        print(f"[{now()}] Loading MusicGen-small...")
        mg_processor, mg_model, mg_sr = load_musicgen()
        print(f"[{now()}] MusicGen partial continuation (slow)...")
        if as_regen.get("musicgen") is None:
            as_regen["musicgen"] = as_ber_for_variant(lambda w: musicgen_regen_np(mg_processor, mg_model, mg_sr, w))
            save_checkpoint(results)
        if frag_regen.get("musicgen") is None:
            frag_regen["musicgen"] = frag_ber_for_variant(lambda w: musicgen_regen_np(mg_processor, mg_model, mg_sr, w))
            save_checkpoint(results)
    else:
        print(f"[{now()}]   [skip, already checkpointed] MusicGen: AudioSeal={as_regen['musicgen']:.4f}, "
              f"fragile={frag_regen['musicgen']:.4f}")

    if results.get("perceptual_stacked_vs_clean") is None:
        print(f"[{now()}] Measuring perceptual quality of stacked audio vs. clean original...")
        from visqol import VisqolApi
        visqol_api = VisqolApi()
        visqol_api.create(mode="audio")
        pesq_scores, visqol_scores, stft_scores = [], [], []
        n_pesq_fail = 0
        for i in range(len(stacked_list)):
            try:
                pesq_scores.append(compute_pesq_wb(orig_np[i], stacked_list[i]))
            except Exception:
                n_pesq_fail += 1
            try:
                visqol_scores.append(compute_visqol(visqol_api, orig_np[i], stacked_list[i]))
            except Exception:
                pass
            stft_scores.append(multi_res_stft_loss(orig_np[i], stacked_list[i]))
        results["perceptual_stacked_vs_clean"] = {
            "pesq_wb": {"mean": float(np.mean(pesq_scores)) if pesq_scores else None, "n_fail": n_pesq_fail},
            "visqol_moslqo": {"mean": float(np.mean(visqol_scores)) if visqol_scores else None},
            "mrstft_loss": {"mean": float(np.mean(stft_scores))},
        }
        save_checkpoint(results)
    else:
        print(f"[{now()}]   [skip, already checkpointed] perceptual quality: "
              f"{results['perceptual_stacked_vs_clean']}")

    as_ber_benign = results["audioseal_ber_benign_stacked"]
    frag_ber_benign = results["fragile_ber_benign_stacked"]
    as_ber_regen = results["audioseal_ber_regen_stacked"]
    frag_ber_regen = results["fragile_ber_regen_stacked"]
    perceptual = results["perceptual_stacked_vs_clean"]

    results["n_clips"] = len(val_wavs)
    results["comparison_points"] = {
        "audioseal_standalone_source": "09_baseline_audioseal.py results",
        "fragile_standalone_source": "04_e2_fragile_path.py results (BER=0 benign, ~0.47-0.56 regen)",
        "joint_quality_source": "05_e3_joint_path.py results (ViSQOL 4.40->3.96, PESQ 2.40->1.36, STFT 1.10->1.58)",
    }
    save_checkpoint(results)

    print(f"[{now()}] AudioSeal BER on stacked audio (benign): {as_ber_benign}")
    print(f"[{now()}] AudioSeal BER on stacked audio (regen): {as_ber_regen}")
    print(f"[{now()}] Fragile path BER on stacked audio (benign): {frag_ber_benign}")
    print(f"[{now()}] Fragile path BER on stacked audio (regen): {frag_ber_regen}")
    print(f"[{now()}] Perceptual quality of stacked audio vs. clean: {perceptual}")
    print(f"[{now()}] Compare against: AudioSeal standalone (09), E2 fragile standalone "
          f"(04, BER=0 benign / ~0.47-0.56 regen), E3 joint quality "
          f"(05, ViSQOL 4.40->3.96, PESQ 2.40->1.36, STFT 1.10->1.58).")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    as_flat = [v for g in as_ber_benign.values() for v in g.values()]
    frag_flat = [v for g in frag_ber_benign.values() for v in g.values()]
    axes[0].boxplot([as_flat, frag_flat], labels=["AudioSeal\n(stacked)", "Our fragile\n(stacked)"])
    axes[0].set_title("BER under benign transforms\n(stacked audio)")
    axes[0].set_ylabel("BER")

    metrics = ["pesq_wb", "visqol_moslqo", "mrstft_loss"]
    e3_joint_vals = [1.36, 3.96, 1.58]
    stacked_vals = [perceptual["pesq_wb"]["mean"] or 0, perceptual["visqol_moslqo"]["mean"] or 0,
                     perceptual["mrstft_loss"]["mean"]]
    x = np.arange(len(metrics))
    axes[1].bar(x - 0.2, e3_joint_vals, width=0.4, label="E3 (our joint)")
    axes[1].bar(x + 0.2, stacked_vals, width=0.4, label="Naive stacking")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(metrics, rotation=15)
    axes[1].set_title("Perceptual quality:\njoint vs. naive stacking")
    axes[1].legend()

    regen_labels = list(as_ber_regen["griffinlim"].keys()) + ["dac", "musicgen"]
    as_regen_vals = list(as_ber_regen["griffinlim"].values()) + [as_ber_regen["dac"], as_ber_regen["musicgen"]]
    frag_regen_vals = list(frag_ber_regen["griffinlim"].values()) + [frag_ber_regen["dac"], frag_ber_regen["musicgen"]]
    xr = np.arange(len(regen_labels))
    axes[2].bar(xr - 0.2, as_regen_vals, width=0.4, label="AudioSeal (stacked)")
    axes[2].bar(xr + 0.2, frag_regen_vals, width=0.4, label="Our fragile (stacked)")
    axes[2].axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    axes[2].set_xticks(xr)
    axes[2].set_xticklabels([f"gl={g}" for g in regen_labels], rotation=45)
    axes[2].set_title("BER under regeneration\n(stacked audio)")
    axes[2].legend()

    fig.suptitle(f"Naive dual-watermark stacking (AudioSeal + our fragile path), n={len(val_wavs)}")
    fig.tight_layout()
    fig.savefig(LOCAL_FIG, dpi=300)
    plt.close(fig)
    copy_to_project(LOCAL_FIG, "fig_12_01_naive_stacking_vs_joint.png")

    print(f"[{now()}] DONE. exp_baseline_naive_stacking_results.json, "
          f"fig_12_01_naive_stacking_vs_joint.png saved to PROJECT_DIR.")


if __name__ == "__main__":
    main()