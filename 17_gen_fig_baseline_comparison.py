# =============================================================================
# Program      : 17_gen_fig_baseline_comparison.py
# Purpose      : Diagnostic/figure-generation only — not part of the
#                numbered E0-E10 pipeline. Produces ONE unified figure for
#                the paper's "Baseline Comparison" subsection, replacing
#                the idea of reusing 09-12's four separate per-baseline
#                plots (which would visually fragment the comparison that
#                subsection is actually making).
#
#                Pulls REAL per-condition BER values from the five actual
#                result JSONs already in PROJECT_DIR — not the aggregate min/max
#                ranges summarized in prior discussion — so the figure is
#                built from the same numbers the paper's table cites, not
#                an approximation of them.
#
#                Two panels (Benign left, Regen right), matching the
#                two-panel box-plot convention used in every other figure
#                in this project (09-12's own figures), for visual
#                consistency across the paper. SIX systems per panel:
#                  Ours (joint)      — from 05_e3_joint_path.py's own
#                                       results (Path A + Path B combined
#                                       benign BER; Path B's regen BER,
#                                       since Path A has no regen claim)
#                  AudioSeal         — from 09's standalone results
#                  WavMark           — from 10's standalone results
#                  SilentCipher      — from 11's standalone results
#                  AudioSeal (stacked) — from 12's naive-stacking results
#                  Ours (stacked)      — from 12's naive-stacking results
#                                         (our fragile path, stacked on
#                                         top of AudioSeal)
#                Putting the stacked pair alongside the standalone
#                baselines in the SAME figure is deliberate: it makes the
#                naive-combination finding (AudioSeal's benign/regen
#                separation collapsing entirely once stacked) directly
#                visible by comparison, not just stated in prose.
#
# GPU Required : NO (pure plotting, no models loaded)
# =============================================================================

!pip install -q h5py matplotlib tqdm

import json
import shutil
import time
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_DIR = "/content/drive/MyDrive/paper/HyperFrag/"
MAX_RETRIES = 3
TIMEOUT_SECONDS = 60

LOCAL_SCRATCH = "/content/scratch"
os.makedirs(LOCAL_SCRATCH, exist_ok=True)
LOCAL_FIG = f"{LOCAL_SCRATCH}/fig_baseline_unified_comparison.png"


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


def copy_from_project(remote_filename, local_filepath):
    """Copy a file from PROJECT_DIR (Google Drive) to local path."""
    project_filepath = os.path.join(PROJECT_DIR, remote_filename)
    if not os.path.isfile(project_filepath):
        print(f"[PROJECT_DIR] {remote_filename} not found in PROJECT_DIR")
        return False
    
    if os.path.exists(local_filepath):
        print(f"[PROJECT_DIR] {local_filepath} already exists, skipping copy.")
        return True
    
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            os.makedirs(os.path.dirname(local_filepath), exist_ok=True)
            total_size = os.path.getsize(project_filepath)
            
            with open(local_filepath, "wb") as f:
                with open(project_filepath, "rb") as src:
                    while True:
                        chunk = src.read(8192)
                        if not chunk:
                            break
                        f.write(chunk)
            return True
        except Exception as e:
            print(f"[PROJECT_DIR] download attempt {attempt}/{MAX_RETRIES} for "
                  f"{remote_filename} failed: {e}")
            if attempt == MAX_RETRIES:
                return False
            time.sleep(5)
    return False


def flatten_benign(ber_benign_dict):
    """ber_benign is always structured as {group: {condition: value}} —
    flattens to a plain list of BER values across all 18 conditions."""
    return [v for group in ber_benign_dict.values() for v in group.values()]


def flatten_regen(ber_regen_dict):
    """ber_regen structures vary slightly by script (griffinlim is nested,
    dac/musicgen are flat), but all follow the same {griffinlim: {...},
    dac: val, musicgen: val} shape used since 09."""
    vals = list(ber_regen_dict.get("griffinlim", {}).values())
    if ber_regen_dict.get("dac") is not None:
        vals.append(ber_regen_dict["dac"])
    if ber_regen_dict.get("musicgen") is not None:
        vals.append(ber_regen_dict["musicgen"])
    return vals


def main():
    files_needed = {
        "ours": "exp3_joint_results.json",
        "audioseal": "exp_baseline_audioseal_results.json",
        "wavmark": "exp_baseline_wavmark_results.json",
        "silentcipher": "exp_baseline_silentcipher_results.json",
        "stacking": "exp_baseline_naive_stacking_results.json",
    }
    data = {}
    for key, remote_name in files_needed.items():
        local_path = f"{LOCAL_SCRATCH}/{remote_name}"
        print(f"Copying {remote_name} from PROJECT_DIR...")
        if not copy_from_project(remote_name, local_path):
            raise SystemExit(
                f"{remote_name} not found in PROJECT_DIR — this figure needs all five "
                f"result files (05, 09, 10, 11, 12) to already exist. Run the "
                f"missing one first."
            )
        with open(local_path) as f:
            data[key] = json.load(f)

    # --- Ours (joint): benign = Path A + Path B combined; regen = Path B only ---
    # ber_b_regen is flat ({n_iter: value}), confirmed against 05's actual
    # source — unlike 09-12's nested {"griffinlim": {...}, "dac": ..., "musicgen": ...}.
    ours_benign = flatten_benign(data["ours"]["ber_a_benign"]) + flatten_benign(data["ours"]["ber_b_benign"])
    ours_regen = list(data["ours"]["ber_b_regen"].values())

    # --- Standalone baselines ---
    audioseal_benign = flatten_benign(data["audioseal"]["ber_benign"])
    audioseal_regen = flatten_regen(data["audioseal"]["ber_regen"])
    wavmark_benign = flatten_benign(data["wavmark"]["ber_benign"])
    wavmark_regen = flatten_regen(data["wavmark"]["ber_regen"])
    silentcipher_benign = flatten_benign(data["silentcipher"]["ber_benign"])
    silentcipher_regen = flatten_regen(data["silentcipher"]["ber_regen"])

    # --- Naive stacking: AudioSeal-on-stacked and our-fragile-on-stacked ---
    as_stacked_benign = flatten_benign(data["stacking"]["audioseal_ber_benign_stacked"])
    as_stacked_regen = flatten_regen(data["stacking"]["audioseal_ber_regen_stacked"])
    frag_stacked_benign = flatten_benign(data["stacking"]["fragile_ber_benign_stacked"])
    frag_stacked_regen = flatten_regen(data["stacking"]["fragile_ber_regen_stacked"])

    labels = ["Ours\n(joint)", "AudioSeal", "WavMark", "SilentCipher",
              "AudioSeal\n(stacked)", "Ours\n(stacked)"]
    benign_series = [ours_benign, audioseal_benign, wavmark_benign, silentcipher_benign,
                      as_stacked_benign, frag_stacked_benign]
    regen_series = [ours_regen, audioseal_regen, wavmark_regen, silentcipher_regen,
                     as_stacked_regen, frag_stacked_regen]

    print("\nSummary (min-max) per system, for sanity-checking against the paper's table:")
    for label, b, r in zip(labels, benign_series, regen_series):
        label_flat = label.replace("\n", " ")
        print(f"  {label_flat:22s}  benign [{min(b):.3f}, {max(b):.3f}]   "
              f"regen [{min(r):.3f}, {max(r):.3f}]")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    colors = ["#2E7D32", "#4C72B0", "#C44E52", "#8172B2", "#7BA7D9", "#8FBC8F"]

    for ax, series, title in zip(
        axes, [benign_series, regen_series],
        ["Benign-transform BER\n(low is correct for a working watermark)",
         "Regeneration BER\n(near-chance, i.e.\\ high, indicates fragility working as intended)"]
    ):
        bp = ax.boxplot(series, tick_labels=labels, patch_artist=True)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="chance")
        ax.set_ylabel("BER")
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="x", labelsize=8)
        ax.legend(fontsize=8)

    fig.suptitle("Unified baseline comparison: Ours vs. three published watermarks "
                 "vs. naive dual-watermark stacking", fontsize=12)
    fig.tight_layout()
    fig.savefig(LOCAL_FIG, dpi=300)
    plt.close(fig)
    copy_to_project(LOCAL_FIG, "fig_baseline_unified_comparison.png")
    print(f"\nDONE. fig_baseline_unified_comparison.png saved to PROJECT_DIR and at {LOCAL_FIG}.")


if __name__ == "__main__":
    main()