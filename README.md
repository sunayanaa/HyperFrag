# HyperFrag: Detecting AI Re-Synthesis of Music via Fragile Latent Watermarking
### Sridharan Sankaran (sridharan.sankaran@ieee.org)

This repo implements and evaluates **HyperFrag**, a dual-path audio watermark
built on a frozen, pretrained EnCodec (24 kHz) codec:

- **Path A (robust):** a HyperNetwork-conditioned FiLM modulation of the
  encoder's pre-quantization latent, recovered reliably under benign
  processing (compression, resampling, noise, gain, EQ).
- **Path B (fragile):** an independent, disjointly-embedded modulation
  trained to survive the same benign transforms while collapsing toward
  chance-level recoverability under generative re-synthesis — the signal
  meant to flag "this was AI-regenerated," not just "this was processed."

Both paths live on non-overlapping channel slices of the same 128-channel
latent (`[0:32]` for Path A, `[32:64]` for Path B), decoded through the
*same* frozen decoder. All training/evaluation ran on a single Google Colab
T4 GPU, checkpointing to PROJECT_DIR in Google Drive
(`/content/drive/MyDrive/paper/HyperFrag/`) so sessions can be interrupted
and resumed without losing progress.


---

## Pipeline at a Glance

| # | Script | Experiment | GPU | Trains? | Depends on |
|---|---|---|---|---|---|
| 01 | `01_e0_dataextract.py` | E0 (data prep) | No | — | Google Drive (Jamendo, MUSDB18) |
| 02 | `02_e0_codec_sanity.py` | E0 (codec ceiling) | Yes | No | `01` |
| 03 | `03_e1_robust_path.py` | E1 (Path A alone) | Yes | Yes | `01` |
| 04 | `04_e2_fragile_path.py` | E2 (Path B alone) | Yes | Yes | `01` |
| 05 | `05_e3_joint_path.py` | E3 (A + B jointly) | Yes | Yes | `01` |
| 06 | `06_e4_generative_attack.py` | E4 (real generative attack) | Yes | No (eval only) | `01`, `05` |
| 07 | `07_e5_overwriting_attack.py` | E5 (overwriting attack) | Yes | Yes | `01`, `03`, `05` |
| 08 | `08_e6_ablations.py` | E6 (fixed-FiLM ablation) | Yes | Yes | `01` |

Every script (except the two diagnostics) checkpoints to the same PROJECT_DIR
directory, so `01` only needs to run once — every later script downloads
`dataset_e0.h5` from PROJECT_DIR rather than touching Google Drive again.

---

## 01 — `01_e0_dataextract.py` (Data Preparation)

**Input:** Google Drive — MTG-Jamendo (`autotagging_genre.tsv` +
`audio_data/<00-99>/<track_id>.mp3`) and MUSDB18 (`musdb18_wav_24k.zip`).
Also fetches the official 50 MUSDB18 test-track names via the `musdb`
package's lightweight metadata download (~tens of MB, not the full
dataset) — needed because the zip ships as a flat 2-stem layout with no
train/test marker.

**Output (PROJECT_DIR):**
- `dataset_e0.h5` — 1000 genre-stratified Jamendo clips + 50 official
  MUSDB18 test clips, all 24 kHz mono, 10 s each. MUSDB18 clips are a
  **reconstructed mixture** (`vocal + instrumental` stem sum), not the
  dataset's original mixture file.
- `manifest_e0.json` — per-clip metadata (id, source, genre, split, offset).

**Notes:** Selection checks file existence on Drive *as it selects*, and
backfills from the next candidate in genre rotation until the full target
count is actually secured — a fixed-size pass silently capped out at
264/999 clips in an earlier version because ~735 files were missing on
Drive. GPU not required. Dependencies: `soundfile`, `librosa`, `h5py`,
`tqdm`, `musdb`.

---

## 02 — `02_e0_codec_sanity.py` (Reconstruction Ceiling)

**Purpose:** establishes the quality ceiling every later watermarked
variant gets compared against — frozen EnCodec, no watermark, at 6 kbps.

**Input (PROJECT_DIR):** `dataset_e0.h5`, `manifest_e0.json`.

**Output (PROJECT_DIR):** `exp0_codec_sanity.json` (PESQ-wb, ViSQOL-audio-mode,
multi-res STFT loss per clip + summary stats), two diagnostic figures
(metric distributions, one example original-vs-reconstructed spectrogram).

**Notes:** Three different sample rates are used deliberately — PESQ at
16 kHz (its only defined rates are 8/16 kHz), ViSQOL audio-mode at
48 kHz, STFT loss at native 24 kHz. `visqol-python` (pure-Python PyPI
port) is used instead of the official `google/visqol`, which requires a
bazel build impractical on Colab. Dependencies: `pesq`, `visqol-python`,
plus the standard `torch`/`torchaudio`/`encodec` stack.

---

## 03 — `03_e1_robust_path.py` (Path A, Robust ID)

**Input (PROJECT_DIR):** `dataset_e0.h5`.

**Output (PROJECT_DIR):** `e1_robust_checkpoint.pth`, `exp1_robust_results.json`
(training history + BER by benign transform), `fig_03_01_ber_by_transform.png`.

**Final config:** `KEY_BITS=4`, `N_EPOCHS=30`. Reaches BER = 0 across
every benign transform tested (compression, resample, noise, gain, EQ).

**This was the hardest script to get right — worth reading the full
in-file changelog (v1.0→v3.0) if reproducing from scratch.** Summary of
what actually went wrong, in order:
1. File copy helpers had no timeout → a stalled operation hung silently
   for extended periods. Fixed with timeout + retry + progress reporting.
2. `model.quantizer.decode()` expects codes as `[n_q,B,T]`, but
   `model.encode()` returns `[B,n_q,T]` — required a `.transpose(0,1)`.
3. EnCodec's encoder/decoder contain an LSTM; cuDNN's fused kernel
   refuses to `backward()` through a layer that ran forward in eval mode.
   Fixed with `torch.backends.cudnn.enabled = False`.
4. Evaluation ran all 105 val clips through the full encoder/decoder in
   one batch (vs. batch size 8 in training) → OOM. Fixed by batching eval.
5. **The big one:** the original design modulated the *post-quantization*
   embedding — decoded from fixed integer codes that never went through
   an actual quantization step, feeding the decoder an off-manifold input
   it was never trained on. Five independent configurations (bound width,
   loss weighting, a batch-size-validated fix) all produced statistically
   identical chance-level results. Pivoted to modulating the *encoder's
   raw pre-quantization output* instead, with a manually-implemented
   straight-through estimator (`emb + (quantizer(emb) - emb).detach()`)
   around the frozen quantizer.
6. A capacity search (binary search on `KEY_BITS`) found a sharp cliff:
   1 and 4 bits converge cleanly; 6 bits is real but slow and unsettled
   even at 30 epochs; 8/16/32 bits never move off chance. `K=4` was
   committed as the production value for this training budget (945
   clips, batch 8, 30 epochs, single T4) — not a fixed architectural
   limit, see the paper's Discussion section.

**Dependencies:** `torch`, `torchaudio`, `encodec`, `pydub`, `h5py`,
`matplotlib`, `tqdm`.

---

## 03b — `03b_extractor_sanity_check.py` (Diagnostic, one-off)

Not part of the E0-E7 sequence. Written mid-debugging of `03` to isolate
one variable cheaply: can the extractor architecture + training loop
learn *anything at all*, independent of whether the EnCodec-mediated
embedding pipeline produces a learnable signal? Trains on a synthetic,
obviously-strong injected signal with no codec involved — confirmed the
mechanics worked (bit_acc climbed 0.47→0.62 over 150 steps) before the
actual bug (post- vs. pre-quantization insertion) was found. No GPU
required, no PROJECT_DIR I/O. Kept for reference; not needed to reproduce results.

---

## 04 — `04_e2_fragile_path.py` (Path B, Fragile Integrity)

**Input (PROJECT_DIR):** `dataset_e0.h5`.

**Output (PROJECT_DIR):** `e2_fragile_checkpoint.pth`, `exp2_fragile_results.json`
(BER under benign transforms *and* under the regeneration proxy),
`fig_04_01_survive_vs_break.png`.

**Final config:** `KEY_BITS=4`, `N_EPOCHS=30`. Same channel-slice and
pre-quantization/straight-through pipeline as `03`, but a **different**
channel range (`[32:64]`, disjoint from Path A's `[0:32]`) and an
additional loss term.

**Key design choice — Griffin-Lim as the training-time regeneration
proxy:** real generative-model evaluation (MusicGen, DAC) is reserved for
`06`; training instead uses Griffin-Lim magnitude-only reconstruction
(STFT → discard phase → iterative phase reconstruction → ISTFT) as a
cheap, theoretically-motivated stand-in for "regeneration loses fine
structure but keeps macro content."

**Loss:** adds a hinge fragile-break term,
`L_break = relu(ln(2) − BCE(extractor(regen(x_wm)), K))`, which only
penalizes the extractor if bits are *still* recoverable after
regeneration — not an unbounded push toward confidently-wrong
predictions, which would be numerically unstable.

**Result:** BER = 0 (essentially) under every benign transform, BER ≈ 0.47
(near chance) under Griffin-Lim at every severity tested — the core
survives/breaks asymmetry, demonstrated cleanly. Dependencies: same as `03`.

---

## 05 — `05_e3_joint_path.py` (Path A + Path B, Jointly)

**Input (PROJECT_DIR):** `dataset_e0.h5`.

**Output (PROJECT_DIR):** `e3_joint_checkpoint.pth`, `exp3_joint_results.json`
(Path A benign BER, Path B benign+regen BER, perceptual quality vs. E0's
ceiling), `fig_05_01_interference_check.png`.

**Final config:** `KEY_BITS=4` for both paths, `N_EPOCHS=30`.

**Purpose:** `03` and `04` each produced their *own* independently
watermarked audio, touching only their own channel slice. This is the
actual proposed system — one watermarked signal carrying both, embedded
in a single forward pass through the shared decoder, so the two paths
genuinely interact. Also the first place watermarked-audio PESQ/ViSQOL is
measured (E0 only measured the *unwatermarked* codec ceiling).

**Result:** both paths still learn, but joint training has a real,
asymmetric cost — Path A's BER rises from 0 to 0.01-0.03, Path B's rises
further to 0.09-0.11, and perceptual quality drops (ViSQOL 4.40→3.96).
Path B's core survive/break asymmetry is preserved regardless.
Dependencies: same as `03`, plus `pesq`, `visqol-python`.

---

## 06 — `06_e4_generative_attack.py` (Real Generative Attack, Eval Only)

**Input (PROJECT_DIR):** `dataset_e0.h5`, **and `e3_joint_checkpoint.pth` from
`05` (required — this script trains nothing, only evaluates).**

**Output (PROJECT_DIR):** `exp4_generative_attack_results.json`,
`fig_06_01_real_generative_attack.png`.

**Purpose:** replaces the Griffin-Lim training proxy with two real models
never seen during training — **MusicGen-small** (audio-prompted
continuation: first 3 s of a clip kept real, remaining 7 s AI-generated —
a partial-regeneration/remix scenario) and **DAC** (full-clip
resynthesis through an architecturally unrelated codec).

**`N_EVAL_CLIPS=16`**, not the full 105 — MusicGen's autoregressive
generation is far slower than anything else in this pipeline.

**Dependency note:** originally used `audiocraft` for MusicGen, which
pulled in a fragile dependency chain (`av==11.0.0` pinned to a 2024-era
torch, plus `xformers`, `spacy`, `demucs`) that failed to build — a
documented, common issue independent of this environment (see
GitHub issues #476, #463 on `facebookresearch/audiocraft`). **Switched to
`transformers`' `MusicgenForConditionalGeneration`** — same model
weights, no fragile compiled dependencies, confirmed to support genuine
audio-prompted continuation.

**Result:** DAC (full-clip) drives both paths to near-chance BER,
closely matching the Griffin-Lim proxy's behavior — evidence the proxy
generalizes. MusicGen (partial continuation) gives intermediate BER for
both paths, since 30% of each clip is untouched — a different, milder
threat than full regeneration, not a failure of the fragility claim.
Dependencies: `torch`, `torchaudio`, `encodec`, `transformers`,
`descript-audio-codec`, `h5py`, `matplotlib`, `tqdm`.

---

## 07 — `07_e5_overwriting_attack.py` (Overwriting-Attack Robustness)

**Input (PROJECT_DIR):** `dataset_e0.h5`, and **both** `e1_robust_checkpoint.pth`
(from `03`) and `e3_joint_checkpoint.pth` (from `05`) — required.

**Output (PROJECT_DIR):** `e5_attacker_checkpoint.pth`,
`exp5_overwriting_results.json`, `fig_07_01_overwriting_attack.png`.

**Two-phase gray-box attack simulation:**
- **Phase 1** trains an independent "attacker" HyperNetwork + extractor
  from scratch, using the *exact* published architecture and training
  recipe as `03` (a competent gray-box attacker who read the paper would
  just reproduce its methodology) — but with their own key, no knowledge
  of the real trained weights. `N_EPOCHS=30` directly, no cautious
  first-run, since this recipe is already proven.
- **Phase 2** is the actual attack: the trained attacker model overwrites
  *already-watermarked* audio (not clean audio — realistic, since an
  attacker only has access to distributed content) from both `03`'s and
  `05`'s checkpoints, testing whether Path B's presence changes Path A's
  vulnerability.

**Result:** the original watermark is meaningfully degraded in both
conditions (not fully resisted — reported honestly, not oversold), but
whether the attack *succeeds* at imposing the attacker's own identity
differs: it succeeds against standalone Path A, but fails (near chance)
against jointly-trained Path A. Path B's BER also rises measurably even
though it was never the attack's target — incidental tamper evidence.
Dependencies: `torch`, `torchaudio`, `encodec`, `h5py`, `matplotlib`, `tqdm`.

---

## 08 — `08_e6_ablations.py` (Ablation: Fixed-FiLM vs. HyperNetwork)

**Input (PROJECT_DIR):** `dataset_e0.h5`.

**Output (PROJECT_DIR):** `e6_fixedfilm_checkpoint.pth`,
`exp6_ablation_fixedfilm_results.json`,
`fig_08_01_hypernet_vs_fixedfilm.png`.

**Scope note:** the blueprint's E6 has four arms; three needed zero new
training and are cited from existing results instead of re-run:
"remove Path B" = `03`'s result, "remove Path A" = `04`'s result, and
"insertion point (pre- vs. post-quantization)" = `03`'s own development
history (five independent configurations already settled this more
thoroughly than one clean ablation run could). "Residual-VQ-stage"
insertion was scoped out entirely as future work — it would require
intervening between individual RVQ codebook layers, more invasive than
anything else attempted here.

**This script covers the one arm that needed new training:** replacing
the 2-hidden-layer HyperNetwork with a single linear layer (`FixedFiLM`),
everything else identical to `03`, to isolate whether the HyperNetwork's
extra depth/nonlinearity actually matters.

**Final config:** `N_EPOCHS=30` (an 8-epoch first check showed a large
gap that mostly closed by full convergence — see in-file changelog).

**Result:** FixedFiLM converges to BER ≈ 0.0048 vs. HyperNetwork's exact
0 — evidence the HyperNetwork mainly buys *convergence speed*, not a
materially higher asymptotic ceiling at this capacity. Dependencies: same
as `03`.

---

## `test_dac_audiocraft_install.py` (Diagnostic, one-off)

Not part of the E0-E7 sequence. Written when `06`'s original
`audiocraft` install failed silently under `pip install -q`. Installs
`descript-audio-codec` and `audiocraft` **one at a time**, without `-q`,
capturing full stdout/stderr so the real underlying build error is
visible instead of hidden behind a downstream `ModuleNotFoundError`. This
is what identified the `av==11.0.0` build failure that led to `06`'s
pivot away from `audiocraft`. No GPU or PROJECT_DIR required.

---

## Datasets

- **MTG-Jamendo:** 1000 clips, genre-stratified round-robin selection
  from `autotagging_genre.tsv`, resampled to 24 kHz mono, 10 s windows.
- **MUSDB18:** official 50-track test split. The specific zip used
  (`musdb18_wav_24k.zip`) ships as a flat 2-stem layout (`<track>.stem_
  vocal.wav` + `<track>.stem_instrumental.wav`), not the standard
  `train/test/mixture.wav` structure, with no train/test marker at all —
  `01` fetches the authoritative 50 test-track names via the `musdb`
  package and reconstructs each mixture as the stem sum.
- **Combined corpus:** 1050 clips, split once (945 train / 105
  validation, fixed seed `20260716`) and reused **unchanged** across
  every experiment (E1-E6) for direct comparability of reported BER
  numbers.

---

## Shared Infrastructure Notes

- **PROJECT_DIR checkpointing:** every training script checkpoints every 4
  epochs (`CHECKPOINT_EVERY_EPOCHS`) to PROJECT_DIR in Google Drive
  (`/content/drive/MyDrive/paper/HyperFrag/`), so a disconnected Colab
  session resumes without redoing work. Download helpers skip re-downloading
  a file that already exists locally with a matching size, and use `tqdm`
  for progress rather than periodic print statements.
- **`torch.backends.cudnn.enabled = False`** is set in every training
  script (03 onward) — EnCodec's encoder/decoder contain an LSTM, and
  cuDNN's fused kernel cannot `backward()` through a layer whose forward
  pass ran in eval mode. This is a property of EnCodec + this training
  approach, not an environment quirk, and will need to be set again in
  any new script that trains through the frozen codec.
- **`TARGET_BANDWIDTH_KBPS = 6.0`** is the project-wide EnCodec operating
  point throughout — set once in `02` and never varied.
- **`FORCE_FRESH_START`** appears in every training script — set `True`
  when starting a genuinely new configuration (different architecture,
  different loss weights, different key size) that isn't compatible with
  whatever checkpoint already exists in PROJECT_DIR; set back to `False` for
  normal interrupt-and-resume behavior within the same configuration.
  