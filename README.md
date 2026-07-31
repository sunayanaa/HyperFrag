# HyperFrag: Tamper-Evident Latent Watermarking for Detecting AI Re-Synthesis of Music
## Sridharan Sankaran (sridharan.sankaran@ieee.org)

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

This repo covers all eleven experiments reported in the paper (E0-E10),
plus four published-watermark baselines (E7) and a comparison against a
fifth, desynchronization-robust system (E8).

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
| 09 | `09_baseline_audioseal.py` | E7 (baseline: AudioSeal) | Optional | No | `01` |
| 10 | `10_baseline_wavmark.py` | E7 (baseline: WavMark) | Optional | No | `01` |
| 11 | `11_baseline_silentcipher.py` | E7 (baseline: SilentCipher) | Optional | No | `01` |
| 12 | `12_baseline_naive_dual_stacking.py` | E7 (baseline: naive stacking) | Optional | No | `01`, `04` |
| 13 | `13_anti_forensics_adaptive_adversary.py` | E9 (adaptive evasion) | Yes | Yes (attacker only) | `01`, `05` |
| 14 | `14_chain_of_custody_test.py` | E10 (chain-of-custody) | Yes | No (eval only) | `01`, `05` |
| 15 | `15_pitch_time_robustness.py` | Limitations (desync gap) | Yes | No (eval only) | `01`, `05` |
| 16 | `16_baseline_aware.py` | E8 (baseline: AWARE) | Yes | No\* | `01` |
| 17 | `17_gen_fig_baseline_comparison.py` | E7 (unified figure) | No | No | `05`, `09`-`12` results |

\* 16 trains nothing itself, but AWARE's own embedding is a 500-iteration
per-clip adversarial optimization, not a single forward pass — see its
section below.

Every script checkpoints to the same PROJECT_DIR directory, so `01` only needs to
run once — every later script downloads `dataset_e0.h5` from PROJECT_DIR rather
than touching Google Drive again.

**Note on script numbering vs. the paper's experiment numbering:** the
script numbers above are pipeline/production order, not the paper's E0-E10
labels — they were assigned before the paper's final numbering was
settled, and the two only coincide for E0-E6 (scripts 01-08). Baseline
Comparison, AWARE, adaptive evasion, and chain-of-custody are E7-E10 in
the paper regardless of which script number produced them.

Two development-only diagnostics used while debugging `03` and `06` are
not part of this repo, since they don't produce anything reported in the
paper: a synthetic extractor-architecture sanity check, and an
install-failure triage script for `audiocraft`. Both are described in
their respective sections below (03 and 06) for context, since they
explain *why* certain design decisions were made, even though the scripts
themselves aren't included.

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
The paper calls this a *ceiling* rather than an oracle: every watermarked
clip in this project passes through this same frozen pipeline and cannot
exceed its quality.

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
   straight-through estimator (`sg[Q(emb) - emb]`) around the frozen
   quantizer. (A synthetic-signal, no-codec sanity check written
   mid-debugging — confirming the extractor architecture could learn
   *anything* before this bug was found — isolated that the training
   mechanics themselves were sound; not included in this repo since it
   produced nothing reported in the paper.)
6. A capacity search (binary search on `KEY_BITS`) found a sharp cliff:
   1 and 4 bits converge cleanly; 6 bits is real but slow and unsettled
   even at 30 epochs; 8/16/32 bits never move off chance. `K=4` was
   committed as the production value for this training budget (945
   clips, batch 8, 30 epochs, single T4) — not a fixed architectural
   limit, see the paper's Discussion section.

**Dependencies:** `torch`, `torchaudio`, `encodec`, `pydub`, `h5py`,
`matplotlib`, `tqdm`.

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

**This is the "defender" checkpoint used throughout the rest of the
pipeline** — every script from `06` onward that needs an already-trained
joint system (`06`, `12`'s fragile half, `13`, `14`, `15`) loads
`e3_joint_checkpoint.pth` from this script.

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
GitHub issues #476, #463 on `facebookresearch/audiocraft`). A small
triage script (installing `audiocraft` alone, without `-q`, to surface
the real build error rather than a downstream `ModuleNotFoundError`)
identified the `av` pin as the culprit; not included in this repo since
it produced nothing reported in the paper. **Switched to `transformers`'
`MusicgenForConditionalGeneration`** — same model weights, no fragile
compiled dependencies, confirmed to support genuine audio-prompted
continuation.

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

## 09-12 — Published-Watermark Baselines (E7, Part 1: AudioSeal / WavMark / SilentCipher / Naive Stacking)

Written in response to a prescreening rejection for lacking any
quantitative baseline comparison. All four scripts are **entirely
inference-only** — no training, no backward passes — since each embeds
via its own pretrained, publicly-released checkpoint. Each was confirmed
to have a genuinely public checkpoint (Hugging Face Hub, PyPI package, or
equivalent) *before* being written, not assumed. Each scores BER under
the identical benign+regeneration battery used throughout this project,
for direct table merging.

**09 — `09_baseline_audioseal.py` (AudioSeal, San Roman et al., ICML
2024).** Fixed 16-bit message (the checkpoint's own capacity, not chosen
to favor either system). Embeds additively in the waveform domain — never
touches EnCodec. v1.2 added a detection-rate diagnostic pass after
production results showed DAC's BER *above* chance (0.685) — distinguishing
"AudioSeal didn't detect anything" from "detected confidently, decoded
wrong," using AudioSeal's own documented `result > 0.5` detection
threshold.

**10 — `10_baseline_wavmark.py` (WavMark, Chen et al., 2023).** Requires
16 kHz mono — the entire battery runs at 16 kHz for this script only,
mirroring how any real WavMark deployment would have to use it. 16-bit
payload. v1.1 added per-condition PROJECT_DIR checkpointing after a production
run showed ~17 min per benign condition (WavMark's internal sync-position
search is far slower than AudioSeal's single forward pass) — an
uncheckpointed multi-hour Colab run is a real disconnection risk. v1.2
added detection-rate diagnostics after several conditions returned
BER=1.0, ambiguous between total non-detection and confident-wrong
decoding.

**11 — `11_baseline_silentcipher.py` (SilentCipher, Singh et al.,
Interspeech 2024).** Requires 44.1 kHz (the paper's own recommended
configuration). 40-bit message (5×8-bit), decomposed to individual bits
for BER comparability. **Two real bugs worth knowing if reproducing
this:** (1) v1.2 fixed a torch/torchaudio ABI mismatch
(`OSError: undefined symbol: aoti_torch_abi_version`) — installing
`silentcipher` let pip pull a torch/torchaudio pairing that conflicted
with what was already active; fixed by installing `silentcipher` with
`--no-deps`, and **required a genuine Colab runtime restart**, since the
mismatched torch was already loaded into memory before the error
surfaced. (2) v1.3 fixed `encode_wav`'s message format — the official
README's usage example passes a flat 5-int list, but the actual runtime
requires it wrapped as `[message]` (one message per audio channel,
mono = 1 channel) — the fix was reasoned from the assertion's actual
shape values, not the docs, since the two conflicted.

**12 — `12_baseline_naive_dual_stacking.py` (Naive Combination).** Direct
answer to a reviewer comment that the system "appears to be a combination
of existing work." Applies AudioSeal to a clean clip, then stacks `04`'s
already-trained fragile-path checkpoint on top, completely independently
— exactly as two off-the-shelf systems combined naively would be.
**Result: AudioSeal's BER is driven to chance (0.49-0.52) under every
condition including benign, while the fragile path survives with BER=0
across all benign settings** — naive combination can destroy one signal
outright rather than averaging weaknesses, a finding independently
consistent with DeepMark Benchmark's Process Disruption Attacks (see 16).

**Shared dependencies (09-12):** `torch`, `torchaudio`, the
system-specific package (`audioseal`, `wavmark`, `silentcipher`),
`pydub`, `descript-audio-codec`, `transformers`, `h5py`, `matplotlib`,
`tqdm`.

---

## 13 — `13_anti_forensics_adaptive_adversary.py` (E9, Adaptive Anti-Forensics Evasion)

**Input (PROJECT_DIR):** `dataset_e0.h5`, **and `e3_joint_checkpoint.pth` from
`05` (required — this is the defender).**

**Output (PROJECT_DIR):** `e13_attacker_checkpoint.pth`,
`exp13_anti_forensics_results.json`, `fig_13_01_evasion_transfer.png`.

**Purpose:** every prior attack in this project either forges a new
watermark (`07`) or passively regenerates content (`06`/`12`). This
models an adversary who specifically knows Path B's fragility mechanism
and actively tries to defeat it while still regenerating the content —
the standard anti-forensics question.

**Threat model (gray-box, consistent with `07`):** knows the published
architecture, not our trained weights or key — critically, doesn't know
the true key either, ruling out an attack that directly targets the true
bits. Implements a **self-consistency evasion attack**: the adversary
trains their own Path B clone (same recipe as `07`), then uses PGD
(40 steps, $L_\infty$-bounded) to find a waveform perturbation making
their *own* surrogate's output on regenerated-plus-perturbed audio match
what it saw pre-regeneration. The actual test is transfer: does this,
crafted entirely against the attacker's own surrogate, also fool the
real defender?

**Worth reading the full in-file changelog (v1.0→v1.4) if reproducing —
this was the most involved debugging in the T-IFS expansion.** The
attacker converged to only ~0.62 bit accuracy for three straight
attempts (v1.0-v1.2), with two reasonable-but-wrong hypotheses tested and
rejected in turn (RNG-state divergence, then a hinge-loss interaction)
before the real cause was found in v1.3: `embed_fragile()`'s entire body,
including the HyperNetwork's FiLM computation, was wrapped in a single
`torch.no_grad()` block — the HyperNetwork never received gradient in
*any* prior run, only the extractor trained, on a fixed, arbitrary,
untrained embedding. Fixed by restructuring into separate `no_grad()`
scopes with the FiLM computation and straight-through estimator outside
both. v1.4 additionally found the defender had been embedded using
`embed_fragile` (Path B alone) rather than the true deployed joint
configuration, and added `embed_joint()` to fix this.

**Final result:** the attacker achieves near-total self-evasion against
its own surrogate (BER down to 0.05) but this only narrows the real
defender's BER by 16% at the highest tested budget (0.424→0.355) — far
short of compromise.

**Dependencies:** `torch`, `torchaudio`, `encodec`, `descript-audio-codec`,
`h5py`, `matplotlib`, `tqdm`.

---

## 14 — `14_chain_of_custody_test.py` (E10, Multi-Generation Chain-of-Custody)

**Input (PROJECT_DIR):** `dataset_e0.h5`, **and `e3_joint_checkpoint.pth` from
`05` (required).**

**Output (PROJECT_DIR):** `exp14_chain_of_custody_results.json`,
`fig_14_01_chain_trajectories.png`.

**Purpose:** every prior attack applies one transform to freshly-embedded
audio. Real content has a multi-hop lifecycle — compressed for one
platform, re-shared, possibly regenerated at some point, processed
further. This tests whether a regeneration event partway through a
realistic processing history remains detectable after more benign
processing on top of it.

**Five 4-hop chains**, same fixed benign-transform pool
(MP3@128kbps, resample@22050Hz, noise@30dB, gain@0.85) and DAC as the
regeneration step throughout: `all_benign`, `regen_first`, `regen_last`,
`regen_middle`, `double_regen`. BER for both paths tracked after
**every** hop, not just the endpoint — the trajectory is the actual
result.

**Result:** the all-benign control shows no drift (ruling out
cumulative false-positive risk from ordinary processing alone). Wherever
regeneration occurs, both paths jump to the same near-chance state and
**stay there through every subsequent benign hop**, regardless of how
many follow — the regeneration signature persists as evidence rather
than fading with further ordinary handling.

**Dependencies:** `torch`, `torchaudio`, `encodec`, `pydub`,
`descript-audio-codec`, `h5py`, `matplotlib`, `tqdm`.

---

## 15 — `15_pitch_time_robustness.py` (Limitations: Desynchronization Gap)

**Input (PROJECT_DIR):** `dataset_e0.h5`, **and `e3_joint_checkpoint.pth` from
`05` (required).**

**Output (PROJECT_DIR):** `exp15_pitch_time_results.json`,
`fig_15_01_pitch_time_robustness.png`.

**Purpose:** motivated by DeepMark Benchmark (Kovacevic et al., IEEE
Access 2026), which reports pitch shift and time stretch as unusually
effective at breaking watermarks even at subtle magnitudes. Neither was
in this project's benign-transform battery at any point before this
script — a genuine, previously-untested gap. Severities chosen for direct
comparability to DeepMark's own reported points: pitch shift 5/25/50/100
cents (5 = their exact sharp-transition point), time stretch rate
0.9/1.1/1.4 (1.4 = their exact default).

**Result — a genuine weakness, not a confirming one:** pitch shift at
even 5 cents (an imperceptible 1/20th of a semitone) already produces BER
overlapping the regeneration range (0.405-0.469) — no gradual onset.
Time stretch is a smaller but real problem (0.229-0.329). This became the
basis for the Limitations section's desynchronization discussion and
directly motivated adding AWARE as a fifth baseline (`16`), since AWARE's
own paper specifically targets this failure mode.

**Dependencies:** `torch`, `torchaudio`, `encodec`, `librosa`, `h5py`,
`matplotlib`, `tqdm`.

---

## 16 — `16_baseline_aware.py` (E8, Comparison Against a Desynchronization-Robust Baseline)

**Input (PROJECT_DIR):** `dataset_e0.h5`.

**Output (PROJECT_DIR):** `aware_embedded_cache.npz` (checkpointed embeddings —
see below), `exp_baseline_aware_results.json`, `fig_16_01_aware_vs_ours.png`.

**Purpose:** AWARE (Pavlovic et al., 2025, arXiv:2510.17512) is chosen
specifically because its own paper reports strong robustness to pitch
shift and time stretch — exactly the failure mode `15` found in our own
system. Tested against both the standard benign/regen battery and `15`'s
exact pitch/time severities, not just AWARE's own reported single points.

**Two deliberate departures from 09-12's pattern:** (1) embedding *is*
checkpointed here, since AWARE's embedding is a 500-iteration per-clip
adversarial optimization (Algorithm 1 in its paper), not a single forward
pass — the "re-embed fresh each run" justification used for 09-12 doesn't
hold. (2) Detection-rate diagnostics are built in from the start, since
`detect_watermark` natively returns a confidence score with a documented
threshold.

**Worth reading the full in-file changelog (v1.0→v1.4) if reproducing —
five distinct issues surfaced across the first five runs**, none
requiring a restart except v1.1's:
1. **v1.1** — numpy binary-compatibility crash on first import
   (`ValueError: numpy.dtype size changed...`), same class of bug as
   `11`'s torch/torchaudio issue: the AWARE repo's `pip install -e .` ran
   first and pulled a conflicting numpy version. Fixed by reordering
   installs and isolating the repo install with `--no-deps`.
   **Required a genuine runtime restart.**
2. **v1.2** — `ModuleNotFoundError: No module named 'aware'` despite pip
   confirming a successful install. Root cause (confirmed via `pip show`,
   not guessed): the repo is a `src/`-layout project, and its editable
   install's path-registration mechanism only gets picked up by Python at
   interpreter *startup* — since the install ran mid-session, the
   already-running kernel never saw it. Fixed with an explicit
   `sys.path.insert()`. **No restart needed.**
3. **v1.3** — missing `webrtcvad` (one of several real AWARE dependencies
   `--no-deps` skipped; `pesq`, `pydantic`, `pystoi` added preemptively
   too, based on `pip show`'s own `Requires:` line).
4. **v1.4a** — missing `resampy` (same class as v1.3).
5. **v1.4b** — `ValueError: Invalid watermark length. Expected 20, got
   16`. AWARE's paper states "16 bps" for its own comparison table, but
   the actual distributed checkpoint requires 20 bits — confirmed by the
   error itself. The repo's own README usage example (which embeds 20
   bits even for this same default profile) was the correct signal and
   should have been trusted over the paper's prose from the start.

**Result:** confirms AWARE's claimed advantage — BER=0.005 at 5-cent
pitch shift, vs. our own system's collapse into the regeneration range at
the same severity — but AWARE is not immune either, reaching full
non-detection by 25 cents. Also reveals a specific weakness AWARE's own
paper reports on EnCodec-based compression (underperforming AudioSeal,
attributed to AudioSeal's training exposure to that distortion which
AWARE lacks) — this project's DAC-regen result is consistent with, and
somewhat more severe than, that reported weakness.

**Dependencies:** `torch`, `torchaudio`, `pydub`, `descript-audio-codec`,
`transformers`, `librosa`, `h5py`, `matplotlib`, `tqdm`, `pesq`,
`pydantic`, `pystoi`, `webrtcvad`, `resampy`, plus `aware` itself
(`git clone https://github.com/deepmarkpy/aware.git && cd aware && pip
install --no-deps -e .`).

---

## 17 — `17_gen_fig_baseline_comparison.py` (E7, Unified Baseline Figure)

**Purpose:** figure-generation only, no models loaded, no GPU needed.
Produces one unified figure for the paper's Baseline Comparison
subsection, rather than reusing 09-12's four separate per-baseline plots,
which would visually fragment the comparison the subsection is actually
making. Pulls real per-condition BER values directly from the five result
JSONs already in PROJECT_DIR (`05`'s and `09`-`12`'s), not from summarized
aggregate ranges.

**Input (PROJECT_DIR):** `exp3_joint_results.json` (from `05`),
`exp_baseline_audioseal_results.json`,
`exp_baseline_wavmark_results.json`,
`exp_baseline_silentcipher_results.json`,
`exp_baseline_naive_stacking_results.json` (from `09`-`12`).

**Output (PROJECT_DIR):** `fig_baseline_unified_comparison.png` — two panels
(benign left, regeneration right), six systems per panel: our joint
system, the three standalone baselines, and both halves of the naive
stacking condition (AudioSeal-stacked, ours-stacked). Including the
stacked pair alongside the standalones makes the naive-combination
finding (AudioSeal's separation collapsing entirely once stacked)
directly visible by comparison rather than only stated in prose.

**Dependencies:** `h5py`, `matplotlib`, `tqdm` (no `torch` — pure data
loading and plotting).

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
  every experiment in this project (E0-E10, and all baseline comparisons)
  for direct comparability of reported BER numbers. Baseline scripts
  requiring a different sample rate (WavMark at 16 kHz, SilentCipher at
  44.1 kHz, AWARE at 16 kHz) resample this same 105-clip validation split
  rather than drawing a new one.

---

## Shared Infrastructure Notes

- **PROJECT_DIR checkpointing:** every training script checkpoints every 4
  epochs (`CHECKPOINT_EVERY_EPOCHS`) to PROJECT_DIR in Google Drive
  (`/content/drive/MyDrive/paper/HyperFrag/`), so a disconnected Colab
  session resumes without redoing work. Download helpers skip re-downloading
  a file that already exists locally with a matching size, and use `tqdm`
  for progress rather than periodic print statements. Baseline scripts
  (09-12, 16) extend this to per-condition results, not just training
  epochs, since they have no training phase of their own to checkpoint
  against.
- **`torch.backends.cudnn.enabled = False`** is set in every script that
  trains through the frozen codec (03 onward) — EnCodec's encoder/decoder
  contain an LSTM, and cuDNN's fused kernel cannot `backward()` through a
  layer whose forward pass ran in eval mode. This is a property of
  EnCodec + this training approach, not an environment quirk.
- **`TARGET_BANDWIDTH_KBPS = 6.0`** is the project-wide EnCodec operating
  point throughout — set once in `02` and never varied.
- **`FORCE_FRESH_START`** appears in every training script — set `True`
  when starting a genuinely new configuration that isn't compatible with
  whatever checkpoint already exists in PROJECT_DIR; set back to `False` for
  normal interrupt-and-resume behavior within the same configuration.
- **Installing a fresh third-party package mid-session is a recurring
  risk, not a one-off.** Three separate incidents (11's torch/torchaudio
  ABI mismatch, 16's numpy ABI mismatch, 16's `src/`-layout path issue)
  all trace back to the same root cause: a `pip install` or
  `git clone && pip install -e .` run partway through an already-running
  Colab session can silently change a core package version, or register
  an editable install's path in a way the current kernel never picks up.
  The general fixes that worked here: install well-established packages
  *before* any unproven third-party repo; isolate the unproven package
  with `--no-deps` and add back only its *actual* dependencies (checked
  via `pip show <package>`'s `Requires:` line, not guessed); if the
  failure is a binary/ABI mismatch (an `OSError` or `ValueError` about
  symbol or dtype mismatches), a genuine Colab runtime restart is
  required — the code fix alone does not repair a session where the
  wrong version is already loaded into memory. If the failure is a bare
  `ModuleNotFoundError` despite `pip show` confirming success, check
  whether the package uses a `src/`-layout and whether the install ran
  mid-session (no restart needed — `sys.path.insert()` fixes this
  directly).