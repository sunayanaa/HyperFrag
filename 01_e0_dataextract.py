# =============================================================================
# Program      : 01_e0_dataextract.py
# Version      : 1.3
# Description  : Experiment 0 — Data Preparation.
#
#                Builds the fixed audio corpus shared by E0-E4: a stratified
#                ~1000-clip subset of MTG-Jamendo (10s segments, genre-
#                stratified via autotagging_genre.tsv) plus the official
#                MUSDB18 test split (50 tracks, one 10s segment each). Both
#                sources are verified/resampled to 24 kHz mono to match
#                EnCodec's 24kHz pretrained model. Everything is packed into
#                a single HDF5 archive + manifest JSON and saved to PROJECT_DIR
#                in Google Drive so 02_e0_codec_sanity.py onward never needs to
#                touch Google Drive raw datasets again.
#
#                Drive is only touched here, and only for the exact files
#                needed (targeted per-file copy for Jamendo; a single zip
#                copy for MUSDB18) — not a bulk mirror of either dataset.
#
#                NOTE ON MUSDB18: musdb18_wav_24k.zip turned out to be a
#                flat 2-stem layout (<track>.stem_vocal.wav +
#                <track>.stem_instrumental.wav per track), not the standard
#                train/test/mixture.wav folder structure, and it carries no
#                train/test marker at all. This script fetches the
#                authoritative 50 official test-track names from the `musdb`
#                package itself (a small 7-second-preview metadata download,
#                not the full dataset), matches them against the zip's flat
#                filenames, and reconstructs each mixture as
#                vocal + instrumental. That reconstructed-mixture caveat
#                belongs in the paper's dataset description, not just here.
#
# INPUT:
#                  Google Drive:
#                    /content/drive/MyDrive/datasets/Jamendo/
#                      mtg-jamendo-dataset/data/autotagging_genre.tsv
#                      audio_data/<00-99>/<track_id>.mp3
#                    /content/drive/MyDrive/datasets/MUSDB18/
#                      musdb18_wav_24k.zip
#                  Network (via `musdb` package, Colab's default internet):
#                      7s-preview metadata download from the sigsep-mus-db
#                      GitHub release, used only to get the 50 official
#                      test-track names — no audio from this download is
#                      used in the actual dataset.
#
# STEPS:
#                  Step 1  Mount Google Drive
#                  Step 2  Parse autotagging_genre.tsv; genre-stratified
#                          round-robin selection of Jamendo tracks, checking
#                          file existence on Drive AS it selects and copying
#                          to local Colab disk — any candidate missing on
#                          Drive is skipped and backfilled with the next
#                          candidate in rotation until JAMENDO_TARGET_N files
#                          are actually secured (not a fixed-size pass that
#                          silently caps out at whatever fraction exists)
#                  Step 3  Copy musdb18_wav_24k.zip to local disk once;
#                          fetch official 50 test-track names via `musdb`;
#                          match against the zip's flat stem filenames;
#                          reconstruct each mixture as vocal + instrumental
#                  Step 4  Decode/resample every clip to 24kHz mono, take a
#                          fixed 10s window, write into a resumable HDF5
#                          archive with a manifest (id, source, genre/track,
#                          split, offset, original path)
#                  Step 5  Save dataset_e0.h5 and manifest_e0.json to PROJECT_DIR
#                          in Google Drive (checkpointed every CHECKPOINT_EVERY
#                          clips, so a disconnected session can resume without
#                          redoing work already saved)
#
# OUTPUT FILES : (stored at /content/drive/MyDrive/paper/HyperFrag/)
#                  dataset_e0.h5
#                      /jamendo/<track_id>   float32, 240000 samples (10s @ 24kHz)
#                      /musdb18/<track_id>   float32, 240000 samples (10s @ 24kHz)
#                                            NOTE: musdb18 clips are a
#                                            reconstructed mixture
#                                            (vocal + instrumental sum), not
#                                            the dataset's original mixture.
#                  manifest_e0.json
#                      List of {id, source, genre, split, orig_path, offset_sec}
#                      — for musdb18 entries, orig_path names the two stem
#                      files that were summed to build the clip.
#
# GPU Required : NO
# Dependencies : soundfile, librosa, h5py, tqdm, musdb (pulls in stempeg;
#                needs ffmpeg, which Colab has preinstalled)
#
# NOTE: the `musdb` step requires outbound internet access to fetch its
# 7s-preview metadata from a GitHub release — fine on Colab's default
# runtime, but will fail in a network-restricted environment.
#
# Change Log   :
#   v1.0  2026-07-14  Initial version
#   v1.1  2026-07-15  Replaced the guessed train/test/mixture.wav zip layout
#                      (wrong — zip is a flat 2-stem list) with musdb-package
#                      name matching + vocal/instrumental mixture reconstruction
#   v1.2  2026-07-15  Fixed Jamendo clip-offset bug: was computing the 10s
#                      window offset from the tsv's DURATION field (full
#                      original track length), which routinely landed past
#                      the end of the ~30s preview file actually on disk —
#                      caused only 264/999 clips to decode successfully. Now
#                      probes the real local file duration, same as MUSDB18
#                      already did correctly.
#   v1.3  2026-07-15  v1.2's theory was wrong — the uploaded manifest showed
#                      successful Jamendo offsets up to 432s, which is only
#                      possible if the Drive files are full-length (not 30s
#                      previews), so the offset fix wasn't the real cause of
#                      264/999. Actual cause: stratified_sample() picked a
#                      fixed set of 1000 candidates with no check that the
#                      file existed on Drive, so every missing file (~735 of
#                      them) was a silent shortfall, not a decode failure.
#                      Replaced stratified_sample() + copy_selected_jamendo_
#                      files() with select_and_copy_jamendo(), which checks
#                      existence AS it selects and backfills from the next
#                      candidate in genre rotation until JAMENDO_TARGET_N
#                      files are actually secured.
#
# !pip install soundfile librosa h5py tqdm musdb
# =============================================================================

!pip install -q soundfile librosa h5py tqdm musdb

import os
import csv
import json
import random
import shutil
import zipfile
import time
import datetime
from collections import defaultdict

import numpy as np
import h5py
import librosa
import soundfile as sf
from tqdm import tqdm

# --- Google Drive -----------------------------------------------------------
from google.colab import drive
drive.mount('/content/drive')

DRIVE_DIR = "/content/drive/MyDrive/datasets"
JAMENDO_TSV = f"{DRIVE_DIR}/Jamendo/mtg-jamendo-dataset/data/autotagging_genre.tsv"
JAMENDO_AUDIO_ROOT = f"{DRIVE_DIR}/Jamendo/audio_data"
MUSDB_ZIP = f"{DRIVE_DIR}/MUSDB18/musdb18_wav_24k.zip"

# --- Google Drive PROJECT_DIR for persistent storage -------------------------
PROJECT_DIR = "/content/drive/MyDrive/paper/HyperFrag/"

# --- Local scratch (fast local disk, not Drive) ------------------------------
LOCAL_SCRATCH = "/content/scratch"
LOCAL_JAMENDO_MP3 = f"{LOCAL_SCRATCH}/jamendo_mp3"
LOCAL_MUSDB_ZIP = f"{LOCAL_SCRATCH}/musdb18_wav_24k.zip"
LOCAL_MUSDB_EXTRACT = f"{LOCAL_SCRATCH}/musdb18_extract"
LOCAL_H5 = f"{LOCAL_SCRATCH}/dataset_e0.h5"
LOCAL_MANIFEST = f"{LOCAL_SCRATCH}/manifest_e0.json"
os.makedirs(LOCAL_SCRATCH, exist_ok=True)
os.makedirs(LOCAL_JAMENDO_MP3, exist_ok=True)
os.makedirs(LOCAL_MUSDB_EXTRACT, exist_ok=True)

# --- Corpus sizing ------------------------------------------------------------
SR = 24000
CLIP_SECONDS = 10.0
CLIP_SAMPLES = int(SR * CLIP_SECONDS)
JAMENDO_TARGET_N = 1000
MUSDB_TEST_TARGET_N = 50
RANDOM_SEED = 20260714
CHECKPOINT_EVERY = 50  # clips, not epochs — this program has no training loop

random.seed(RANDOM_SEED)


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


# --- Step 1 (drive mount already done above) --------------------------------

# --- Step 2: genre-stratified Jamendo sampling -------------------------------
def parse_jamendo_tsv(tsv_path):
    """Returns dict: track_id -> {'path': str, 'duration': float, 'genre': str}
    autotagging_genre.tsv columns: TRACK_ID ARTIST_ID ALBUM_ID PATH DURATION TAGS...
    TAGS is variable-length (one or more genre---X tags), so we read with the
    stdlib csv module rather than a fixed-width parser."""
    tracks = {}
    with open(tsv_path, newline="") as fp:
        reader = csv.reader(fp, delimiter="\t")
        header = next(reader, None)
        for row in reader:
            if len(row) < 5:
                continue
            track_id = row[0]
            path = row[3]
            try:
                duration = float(row[4])
            except ValueError:
                duration = 0.0
            tags = row[5:]
            genre_tags = [t.split("---")[-1] for t in tags if t.startswith("genre---")]
            genre = genre_tags[0] if genre_tags else "unknown"
            tracks[track_id] = {"path": path, "duration": duration, "genre": genre}
    return tracks


def select_and_copy_jamendo(tracks, target_n, seed=RANDOM_SEED):
    """Genre-stratified round-robin selection, but unlike a fixed single pass,
    this checks file existence on Drive AS it selects, and keeps drawing
    replacement candidates from the same rotation until target_n files are
    actually secured (or every genre bucket is exhausted). A fixed-size
    single-pass selection would silently cap out at however many of the
    original picks happened to exist on Drive — which is what caused the
    264/999 shortfall in the first run: most of the missing 735 were simply
    absent from Drive's audio_data/, not decode failures."""
    by_genre = defaultdict(list)
    for tid, meta in tracks.items():
        if meta["duration"] >= CLIP_SECONDS:  # need enough audio for a 10s window
            by_genre[meta["genre"]].append(tid)

    rng = random.Random(seed)
    for genre in by_genre:
        rng.shuffle(by_genre[genre])

    genres = list(by_genre.keys())
    rng.shuffle(genres)
    cursors = {g: 0 for g in genres}

    local_paths = {}
    selected_meta = {}
    n_checked = 0
    n_missing = 0

    print(f"[{now()}] Selecting + copying up to {target_n} Jamendo clips "
          f"(round-robin across {len(genres)} genres; missing files are "
          f"skipped and backfilled from the next candidate in rotation, "
          f"not left as a shortfall)...")

    with tqdm(total=target_n, desc="jamendo select+copy") as pbar:
        while len(local_paths) < target_n:
            progressed = False
            for g in genres:
                if len(local_paths) >= target_n:
                    break
                c = cursors[g]
                if c >= len(by_genre[g]):
                    continue
                cursors[g] += 1
                progressed = True
                tid = by_genre[g][c]
                if tid in local_paths:  # shouldn't happen, but stay safe
                    continue
                n_checked += 1
                meta = tracks[tid]
                src = os.path.join(JAMENDO_AUDIO_ROOT, meta["path"])
                dst = os.path.join(LOCAL_JAMENDO_MP3, os.path.basename(meta["path"]))
                if os.path.exists(dst):
                    local_paths[tid] = dst
                    selected_meta[tid] = meta
                    pbar.update(1)
                    continue
                if not os.path.exists(src):
                    n_missing += 1
                    continue  # backfilled automatically by the next candidate in this genre
                shutil.copyfile(src, dst)
                local_paths[tid] = dst
                selected_meta[tid] = meta
                pbar.update(1)
            if not progressed:
                break  # every genre bucket exhausted before reaching target_n

    print(f"[{now()}] Jamendo selection done: {len(local_paths)}/{target_n} clips secured "
          f"after checking {n_checked} candidates ({n_missing} missing on Drive, "
          f"replaced with the next candidate in rotation).")
    if len(local_paths) < target_n:
        print(f"  [WARN] Ran out of candidate tracks before reaching {target_n} — only "
              f"{len(local_paths)} tracks in the tsv metadata have a file actually present "
              f"on Drive. This is a data-availability limit (your Jamendo audio_data/ "
              f"download appears incomplete), not a script bug — if you want more than "
              f"this many clips, the fix is downloading more of the dataset to Drive.")

    return local_paths, selected_meta


# --- Step 4: MUSDB18 — official test-track matching + mixture reconstruction
def stage_musdb18_test_split():
    if not os.path.exists(LOCAL_MUSDB_ZIP):
        print(f"[{now()}] Copying musdb18_wav_24k.zip to local disk (one-time, ~3GB)...")
        shutil.copyfile(MUSDB_ZIP, LOCAL_MUSDB_ZIP)
    else:
        print(f"[{now()}] musdb18_wav_24k.zip already staged locally, skipping copy.")

    with zipfile.ZipFile(LOCAL_MUSDB_ZIP) as zf:
        names = zf.namelist()

    # This zip is a flat 2-stem layout (<track>.stem_vocal.wav +
    # <track>.stem_instrumental.wav), with no directory-level train/test
    # marker and no mixture file. Get the authoritative 50 test-track names
    # from the `musdb` package (a small 7s-preview metadata download —
    # not the full dataset), then match by name.
    print(f"[{now()}] Fetching official MUSDB18 test-track names via `musdb` package...")
    try:
        import musdb
    except ImportError:
        import subprocess
        import sys as _sys
        subprocess.check_call([_sys.executable, "-m", "pip", "install", "-q", "musdb"])
        import musdb

    mus_test = musdb.DB(subsets="test", download=True)
    official_test_names = {t.name for t in mus_test.tracks}
    print(f"[{now()}] musdb package reports {len(official_test_names)} official test tracks.")

    # Map zip entries: "<track name>.stem_vocal.wav" / "<track name>.stem_instrumental.wav"
    stem_pairs = defaultdict(dict)  # track_name -> {'vocal': member, 'instrumental': member}
    for n in names:
        base = os.path.basename(n)
        for suffix, kind in ((".stem_vocal.wav", "vocal"), (".stem_instrumental.wav", "instrumental")):
            if base.lower().endswith(suffix):
                stem_pairs[base[: -len(suffix)]][kind] = n
                break

    def normalize(s):
        return "".join(ch.lower() for ch in s if ch.isalnum())

    norm_official = {normalize(nm): nm for nm in official_test_names}

    matched = {}  # official track name -> {'vocal': member, 'instrumental': member}
    unmatched_official = set(official_test_names)
    for zip_track_name, stems in stem_pairs.items():
        if zip_track_name in official_test_names:
            matched[zip_track_name] = stems
            unmatched_official.discard(zip_track_name)
        else:
            key = normalize(zip_track_name)
            if key in norm_official:
                matched[norm_official[key]] = stems
                unmatched_official.discard(norm_official[key])

    incomplete = [tn for tn, s in matched.items() if "vocal" not in s or "instrumental" not in s]
    for tn in incomplete:
        matched.pop(tn)

    print(f"[{now()}] Matched {len(matched)}/{len(official_test_names)} official test "
          f"tracks against the zip's flat file list "
          f"({len(incomplete)} dropped for missing a stem).")
    if unmatched_official:
        print(f"  [WARN] {len(unmatched_official)} official test tracks not found in the "
              f"zip (name mismatch or missing file) — check spelling/punctuation below "
              f"if this list is non-empty:")
        for nm in sorted(unmatched_official):
            print("    ", nm)

    if not matched:
        print("First 20 entries in the zip, for manual inspection:")
        for n in names[:20]:
            print("   ", n)
        raise SystemExit(
            "No test-split tracks could be matched between the musdb package's "
            "official name list and musdb18_wav_24k.zip's contents. Compare the "
            "two name lists printed above for a systematic naming difference "
            "(e.g. featuring-artist formatting) and adjust normalize()."
        )
    if len(matched) != MUSDB_TEST_TARGET_N:
        print(f"  [WARN] expected {MUSDB_TEST_TARGET_N} test tracks, matched "
              f"{len(matched)} — proceeding with what was found.")

    # Reconstruct each mixture as vocal + instrumental, sample-aligned.
    extracted_paths = {}
    with zipfile.ZipFile(LOCAL_MUSDB_ZIP) as zf:
        for track_name, stems in tqdm(matched.items(), desc="musdb18 reconstruct mixtures"):
            dst = os.path.join(LOCAL_MUSDB_EXTRACT, f"{track_name}.wav")
            if os.path.exists(dst):
                extracted_paths[track_name] = (dst, "cached")
                continue
            vocal_tmp = zf.extract(stems["vocal"], path=LOCAL_MUSDB_EXTRACT)
            instr_tmp = zf.extract(stems["instrumental"], path=LOCAL_MUSDB_EXTRACT)
            vocal, sr_v = sf.read(vocal_tmp, dtype="float32")
            instrumental, sr_i = sf.read(instr_tmp, dtype="float32")
            if sr_v != sr_i:
                print(f"  [WARN] {track_name}: vocal/instrumental sample-rate mismatch "
                      f"({sr_v} vs {sr_i}), skipping.")
                continue
            n = min(len(vocal), len(instrumental))
            mixture = vocal[:n] + instrumental[:n]
            sf.write(dst, mixture, sr_v)
            for tmp in (vocal_tmp, instr_tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
            extracted_paths[track_name] = (
                dst, f"reconstructed: {stems['vocal']} + {stems['instrumental']}")

    return extracted_paths


# --- Step 5: decode/resample/window + resumable HDF5 write ------------------
def load_10s_window(path, offset_sec):
    y, sr = librosa.load(path, sr=SR, mono=True, offset=offset_sec, duration=CLIP_SECONDS)
    if len(y) < CLIP_SAMPLES:
        y = np.pad(y, (0, CLIP_SAMPLES - len(y)))
    elif len(y) > CLIP_SAMPLES:
        y = y[:CLIP_SAMPLES]
    return y.astype(np.float32)


def build_dataset(jamendo_local_paths, jamendo_meta, musdb_local_paths):
    # Resume: pull down whatever's already in PROJECT_DIR before doing any work.
    have_h5 = copy_from_project("dataset_e0.h5", LOCAL_H5)
    have_manifest = copy_from_project("manifest_e0.json", LOCAL_MANIFEST)

    if have_manifest:
        with open(LOCAL_MANIFEST) as f:
            manifest = json.load(f)
        print(f"[{now()}] Resumed manifest with {len(manifest)} entries already done.")
    else:
        manifest = []

    done_ids = {(e["source"], e["id"]) for e in manifest}

    h5_mode = "a" if have_h5 else "w"
    h5f = h5py.File(LOCAL_H5, h5_mode)
    if "jamendo" not in h5f:
        h5f.create_group("jamendo")
    if "musdb18" not in h5f:
        h5f.create_group("musdb18")

    processed_since_checkpoint = 0

    def checkpoint():
        h5f.flush()
        with open(LOCAL_MANIFEST, "w") as f:
            json.dump(manifest, f, indent=2)
        copy_to_project(LOCAL_H5, "dataset_e0.h5")
        copy_to_project(LOCAL_MANIFEST, "manifest_e0.json")
        print(f"[{now()}] Checkpoint saved to PROJECT_DIR ({len(manifest)} clips so far).")

    # --- Jamendo ---
    print(f"[{now()}] Processing Jamendo clips...")
    for i, (tid, local_path) in enumerate(tqdm(jamendo_local_paths.items(), desc="jamendo encode")):
        if ("jamendo", tid) in done_ids:
            continue
        meta = jamendo_meta[tid]
        try:
            real_dur = librosa.get_duration(path=local_path)
        except Exception as e:
            print(f"  [WARN] could not probe duration for {local_path}: {e}")
            continue
        # Use the ACTUAL local preview file's duration, not the tsv's DURATION
        # field (which reflects the original full track, not the ~30s preview
        # that ships in audio_data/) — this was the bug behind the 264/999
        # shortfall: offsets computed from the tsv routinely landed past the
        # end of the file that's actually on disk.
        offset = max(0.0, (real_dur - CLIP_SECONDS) / 2.0) if real_dur >= CLIP_SECONDS else 0.0
        if real_dur < CLIP_SECONDS:
            print(f"  [WARN] {tid}: local file only {real_dur:.1f}s, "
                  f"shorter than {CLIP_SECONDS}s — clip will be zero-padded.")
        try:
            clip = load_10s_window(local_path, offset)
        except Exception as e:
            print(f"  [WARN] failed to decode {local_path}: {e}")
            continue
        h5f["jamendo"].create_dataset(tid, data=clip)
        manifest.append({
            "id": tid, "source": "jamendo", "genre": meta["genre"],
            "split": "n/a", "orig_path": meta["path"], "offset_sec": offset,
        })
        processed_since_checkpoint += 1
        if processed_since_checkpoint >= CHECKPOINT_EVERY:
            checkpoint()
            processed_since_checkpoint = 0
        if (i + 1) % 100 == 0:
            print(f"[{now()}] Jamendo {i + 1}/{len(jamendo_local_paths)} processed")

    # --- MUSDB18 ---
    print(f"[{now()}] Processing MUSDB18 test clips...")
    for i, (track_name, (local_path, orig_member)) in enumerate(
            tqdm(musdb_local_paths.items(), desc="musdb18 encode")):
        if ("musdb18", track_name) in done_ids:
            continue
        try:
            dur = librosa.get_duration(path=local_path)
            offset = 30.0 if dur >= (30.0 + CLIP_SECONDS) else max(0.0, (dur - CLIP_SECONDS) / 2.0)
            clip = load_10s_window(local_path, offset)
        except Exception as e:
            print(f"  [WARN] failed to decode {local_path}: {e}")
            continue
        h5f["musdb18"].create_dataset(track_name, data=clip)
        manifest.append({
            "id": track_name, "source": "musdb18", "genre": "n/a",
            "split": "test", "orig_path": orig_member, "offset_sec": offset,
        })
        processed_since_checkpoint += 1
        if processed_since_checkpoint >= CHECKPOINT_EVERY:
            checkpoint()
            processed_since_checkpoint = 0
        if (i + 1) % 10 == 0:
            print(f"[{now()}] MUSDB18 {i + 1}/{len(musdb_local_paths)} processed")

    checkpoint()  # final flush, in case the last batch was smaller than CHECKPOINT_EVERY
    h5f.close()
    return manifest


# --- Main ---------------------------------------------------------------
def main():
    print(f"[{now()}] Parsing {JAMENDO_TSV} ...")
    all_tracks = parse_jamendo_tsv(JAMENDO_TSV)
    print(f"[{now()}] {len(all_tracks)} Jamendo tracks with genre tags found.")

    jamendo_local_paths, selected_meta = select_and_copy_jamendo(all_tracks, JAMENDO_TARGET_N)
    n_genres = len(set(m["genre"] for m in selected_meta.values()))
    print(f"[{now()}] Secured {len(jamendo_local_paths)} tracks across {n_genres} genres.")

    musdb_local_paths = stage_musdb18_test_split()

    manifest = build_dataset(jamendo_local_paths, selected_meta, musdb_local_paths)

    n_jam = sum(1 for e in manifest if e["source"] == "jamendo")
    n_mus = sum(1 for e in manifest if e["source"] == "musdb18")
    print(f"[{now()}] DONE. dataset_e0.h5 + manifest_e0.json saved to PROJECT_DIR: "
          f"{n_jam} Jamendo clips, {n_mus} MUSDB18 clips.")


if __name__ == "__main__":
    main()