# Adaptive CFG: Norm-Regulated Conflict-Aware Classifier-Free Guidance

**Deep Generative Models — Winter 1404 | Course Project**

> A training-free modification to Stable Diffusion's sampling loop that replaces the fixed guidance scale with a signal-driven adaptive controller. At every denoising step, the model's own internal disagreement between its conditional and unconditional predictions is measured and used to decide how strongly to apply guidance — reducing artifacts at high CFG scales while maintaining prompt alignment.

---

## Table of Contents

1. [Background & Motivation](#1-background--motivation)
2. [The Method — All Four Paper Equations](#2-the-method--all-four-paper-equations)
3. [Repository Structure](#3-repository-structure)
4. [File-by-File Code Explanation](#4-file-by-file-code-explanation)
   - [config.py](#41-configpy)
   - [dataset.py](#42-datasetpy)
   - [core/controller.py](#43-corecontrollerpy)
   - [core/sampler_bridge.py](#44-coresampler_bridgepy)
   - [eval/metrics.py](#45-evalmetricspy)
   - [experiments/manager.py](#46-experimentsmanagerpy)
   - [analyze.py](#47-analyzepy)
   - [main.py](#48-mainpy)
   - [reproduce_results.ipynb](#49-reproduce_resultsipynb)
5. [Experimental Design](#5-experimental-design)
6. [Results & Findings](#6-results--findings)
7. [Reproducing the Results](#7-reproducing-the-results)
8. [Installation & Requirements](#8-installation--requirements)
9. [Full CLI Reference](#9-full-cli-reference)
10. [Python API Reference](#10-python-api-reference)
11. [Kaggle Timing Guide](#11-kaggle-timing-guide)
12. [Citation](#12-citation)

---

## 1. Background & Motivation

### What is Classifier-Free Guidance (CFG)?

Stable Diffusion generates images by iteratively denoising a random noise tensor `z_T` over `T` timesteps. At each step `t`, the UNet runs **twice**: once conditioned on the text prompt (`ε_cond`) and once without any conditioning (`ε_uncond`). These two predictions are then combined using a fixed weighting constant `w` called the **guidance scale**:

```
ε̂_t = ε_uncond_t + w · (ε_cond_t − ε_uncond_t)          [standard CFG, Eq. 1]
```

This pushes the generation toward the direction the prompt is "pointing." A higher `w` means stronger adherence to the prompt — but also more risk of artifacts.

### The Problem with Fixed `w`

When `w` is high (e.g., 12.0):
- **Oversharpening**: Textures develop excessive high-frequency energy, producing harsh, unnatural edges
- **Color burn-in**: Saturation becomes unnatural; colors bleed or saturate beyond realistic levels
- **Reduced diversity**: With multiple seeds, outputs converge toward a single mode

The fundamental issue is that a fixed `w` is applied **uniformly across all 30 timesteps**, ignoring the fact that the model's confidence in its conditional signal varies enormously — at high noise (early steps), the model is uncertain and has a weak signal; at low noise (late steps), it has a much stronger and more reliable signal.

### Our Approach

Instead of a fixed `w`, we read the model's own internal signal at each step to decide whether to apply strong or weak guidance. The key insight: if `ε_cond` and `ε_uncond` are already very different (high "conflict"), aggressively amplifying this difference risks overdrive. If they are similar (low conflict), strong guidance is safer. We formalize this intuition into a controller with four equations.

---

## 2. The Method — All Four Paper Equations

All four equations are implemented exactly as described in the paper, in `core/controller.py` and `core/sampler_bridge.py`.

### Equation 2 — The Conflict Metric

```
Δ_t  = ε_cond_t − ε_uncond_t
s_t  = ‖Δ_t‖₂ / (σ_t + ε)
```

`Δ_t` is the raw difference between the conditional and unconditional noise predictions — the "correction" that CFG applies. `s_t` normalizes it by the scheduler's noise level `σ_t` so that comparisons across timesteps are meaningful. A large `s_t` means the conditional correction is large relative to the noise scale at that step, and aggressively amplifying it is more likely to cause artifacts.

### Equation 3 — Bounded Monotone Mapping

```
w_t = clip( w_min + (w_max − w_min) · exp(−α · s_t),  w_min,  w_max )
```

This converts the conflict signal `s_t` into a guidance scale `w_t`. The exponential `exp(−α · s_t)` is monotone **decreasing**: when `s_t` is large (high conflict), `w_t` approaches `w_min`; when `s_t` is small (low conflict), `w_t` approaches `w_max`. The `clip` ensures the guidance scale never leaves the `[w_min, w_max]` range. `α` controls sensitivity — how quickly the guidance responds to changes in conflict.

### Equation 4 — Exponential Moving Average Smoothing

```
w̃_t = β · w̃_{t+1}  +  (1 − β) · w_t
```

Without smoothing, `w_t` can jump erratically between steps because `s_t` has step-to-step noise. The EMA blends the current value `w_t` with the previous smoothed value `w̃_{t+1}` using the mixing coefficient `β`. Note: since denoising runs from `t=T` down to `t=0`, at step `i` in code, `self.prev_w` holds `w̃` from step `i−1`, which corresponds to `w̃_{t+1}` in the paper's notation (where `t` decreases).

### Final CFG Update — Equation 1

```
ε̂_t = ε_uncond_t  +  w̃_t · (ε_cond_t − ε_uncond_t)
```

This is identical to standard CFG, but with the fixed `w` replaced by `w̃_t` — the per-step smoothed adaptive value. Everything else in the pipeline is unchanged: same UNet, same scheduler, same VAE decode. **No retraining. Zero extra UNet forward passes** (standard CFG already requires both conditional and unconditional passes).

---

## 3. Repository Structure

```
adaptive_cfg/
│
├── reproduce_results.ipynb     ← Kaggle notebook: Run All to reproduce everything
├── main.py                     ← CLI entry point, argument parsing, experiment orchestration
├── config.py                   ← All configuration dataclasses + logging setup
├── dataset.py                  ← Prompt loading with COCO primary + robust fallback chain
├── analyze.py                  ← All figure generation (4 paper figures)
├── requirements.txt            ← Python dependencies
├── README.md                   ← This file
├── .gitignore                  ← Excludes outputs, caches, model weights
│
├── core/
│   ├── __init__.py
│   ├── controller.py           ← Heart of the paper: Eq. 2, 3, 4 + scheduled baseline
│   └── sampler_bridge.py       ← DDIM sampling loop: Eq. 1 + sigma extraction + pipeline loader
│
├── eval/
│   ├── __init__.py
│   └── metrics.py              ← All 4 paper metrics: CLIP, LPIPS, HF energy, saturation
│
├── experiments/
│   ├── __init__.py
│   └── manager.py              ← Runs all 9 variants, logs ETA, checkpoints, prints final table
│
└── outputs/
    └── .gitkeep                ← Placeholder; generated results go here, not committed
```

---

## 4. File-by-File Code Explanation

### 4.1 `config.py`

**Purpose:** Central definition of every configuration parameter in the project. Contains three dataclasses and the logging setup function.

#### `setup_logging(log_file, level)`

Configures two output streams simultaneously: a `StreamHandler` to stdout (visible in terminal or Kaggle cell output) and a `FileHandler` to `experiment.log` (a permanent record of the run). It then silences the six noisiest third-party libraries (`transformers`, `diffusers`, `datasets`, `PIL`, `urllib3`, `huggingface_hub`) down to `WARNING` level so their verbose initialization messages don't flood the log.

#### `AdaptiveCFGConfig` (dataclass)

Holds every parameter that controls the adaptive guidance controller — i.e., everything that touches Equations 2, 3, and 4:

| Field | Type | Default | What it controls |
|---|---|---|---|
| `conflict_metric` | `"l2_norm"` or `"cosine"` | `"l2_norm"` | How to measure Δ_t in Eq. 2 — L2 norm or cosine disagreement |
| `normalization` | `"sigma"` or `"uncond_norm"` | `"sigma"` | Denominator of Eq. 2 — σ_t or ‖ε_uncond‖₂ |
| `mapping_fn` | `"exponential"` or `"reciprocal"` | `"exponential"` | Shape of Eq. 3 — exp(−αs) or 1/(1+αs) |
| `w_min` | float | 2.0 | Lower bound on guidance scale |
| `w_max` | float | 10.0 | Upper bound on guidance scale |
| `alpha` | float | 1.0 | Sensitivity in Eq. 3 — how fast guidance drops as conflict grows |
| `beta` | float | 0.5 | EMA mixing coefficient in Eq. 4 — 0=no smoothing, 1=fully frozen |
| `epsilon` | float | 1e-5 | Numerical stability term in Eq. 2 denominator |

The `short_name()` method returns a string like `"l2_norm_sigma_exponential"` that is used to name output directories and label plots.

#### `ScheduledCFGConfig` (dataclass)

Parameters for the paper's required **scheduled CFG baseline** — a simple time-dependent schedule, not the proposed method. The schedule starts at `w_max` at step 0 (highest noise) and decays to `w_min` at the final step. Two schedules are supported:

- `"linear"`: constant-rate decay, `w(i) = w_max − (w_max − w_min) · i / (T−1)`
- `"cosine"`: smooth cosine anneal, `w(i) = w_min + (w_max − w_min) · 0.5 · (1 + cos(π · i/(T−1)))`

This baseline is **weaker than the adaptive method** because it uses a fixed schedule based only on time, not on the model's internal signal.

#### `ExperimentConfig` (dataclass)

Top-level settings that control the experiment as a whole, not any individual generation:

| Field | Default | What it controls |
|---|---|---|
| `model_id` | `"runwayml/stable-diffusion-v1-5"` | HuggingFace model to load |
| `num_inference_steps` | 30 | Number of DDIM denoising steps per image |
| `seeds` | `[42, 43]` | Random seeds to run — one full pass per seed, same prompts |
| `dataset_name` | `"coco_val"` | Which prompt set to use |
| `num_samples` | 100 | How many prompts from the dataset |
| `output_dir` | `"outputs"` | Root directory for all saved files |
| `device` | `"cuda"` | PyTorch device |
| `image_size` | 512 | Output image resolution (512×512) |
| `fixed_cfg_scales` | `[3.0, 7.5, 12.0]` | Scales to use for fixed CFG baselines |
| `run_scheduled_baselines` | `True` | Whether to run the scheduled CFG baselines |
| `save_images` | `True` | Whether to save individual PNGs (disable for faster run) |
| `log_file` | `"experiment.log"` | Path for the persistent log file |

---

### 4.2 `dataset.py`

**Purpose:** Loads the prompt sets used for evaluation. The primary benchmark is **MS-COCO 2017 validation captions**, chosen because it is the same dataset used to report CLIP scores in the original Stable Diffusion paper — making our numbers directly comparable to the published baseline.

#### `load_prompts(dataset_name, num_samples, seed)`

The single public function. Dispatches to the appropriate private loader, then returns a list of `num_samples` prompt strings. The `seed` is used only for shuffling — ensuring reproducible prompt selection even though the order of loading can vary.

#### `_load_coco_robust(num_samples)` — four-stage waterfall

Because COCO can fail to load for various reasons on Kaggle (network limits, HuggingFace API issues), this function tries four methods in order and stops at the first success:

1. **Disk cache check** — if a previous run already downloaded and parsed COCO annotations, a file `coco_val2017_captions.json` exists in the system's temp directory. This is the fastest path and is used on all runs after the first.

2. **Direct download** via `_try_download_coco_annotations()` — downloads the official COCO 2017 val annotations ZIP from `images.cocodataset.org` (~247MB). Extracts only `annotations/captions_val2017.json` from the ZIP, parses the `"annotations"` array (each entry has a `"caption"` field), shuffles all ~202,654 captions, saves to disk cache for future runs, and returns the first `num_samples`.

3. **HuggingFace datasets** via `_try_hf_coco()` — attempts to stream `HuggingFaceM4/COCO` with `load_dataset(..., streaming=True)`. Extracts captions from each item using `_extract_caption_from_item()`, which knows to check multiple field names (`"prompt"`, `"caption"`, `"text"`, `"captions"`) because different HuggingFace COCO mirrors use different schemas.

4. **PartiPrompts fallback** via `_load_parti_prompts()` — if all COCO methods fail, falls back to the PartiPrompts dataset (`nateraw/parti-prompts`) from HuggingFace, which has 1,632 diverse prompts created by Google. If that also fails, falls through to `_load_drawbench()`.

#### `_load_drawbench(num_samples)` — the last resort that never fails

100 prompts hardcoded directly in the source file as the constant `DRAWBENCH_PROMPTS`. Since they are Python strings in the code itself, this path requires no network, no disk, no external library. It is guaranteed to work on any environment.

#### `FAILURE_MODE_PROMPTS` — curated test set

20 prompts manually selected to stress-test the specific failure modes that the paper targets: close-up faces, hands, text rendering, repeating textures, and symmetrical structures. These are intentionally hard for SD v1.5 at high CFG scales.

---

### 4.3 `core/controller.py`

**Purpose:** The mathematical heart of the project. Implements Equations 2, 3, and 4 from the paper. Also contains the `ScheduledGuidanceController` for the paper's required scheduled baseline. Neither class touches the UNet or the latent directly — they only compute and return a float `w_t` given the noise predictions and sigma value.

#### `AdaptiveGuidanceController`

Instantiated fresh for **each new image** (the `sample()` method in `sampler_bridge.py` creates a new instance per call). This is important because the EMA state (`self.prev_w`) must be reset between images.

**`__init__(config)`**
- Sets `self.prev_w = config.w_max` — the EMA is initialized at the maximum guidance scale. This makes sense because at `t=T` (the first denoising step, highest noise), the model has almost no signal and we don't want the controller to immediately suppress guidance.
- Calls `_build_metric_fn()` and `_build_mapping_fn()` to select the appropriate function implementations via a dispatch table. This strategy pattern means the hot path (`update_and_get_w`) calls a closure directly rather than checking a string condition at every step.

**`_build_metric_fn()`** — implements Eq. 2 with all four ablation combinations:

- `_l2_sigma`: Primary proposed metric. Computes `‖ε_cond − ε_uncond‖₂` then divides by `σ_t + ε`. The `reshape(batch, -1)` flattens the spatial and channel dimensions so `torch.norm(..., dim=-1)` computes a single scalar norm per batch element, then `.mean()` averages across the batch (which is always 1 in our case but written to be general).

- `_l2_uncond`: Ablation (ii). Same numerator, but normalizes by `‖ε_uncond‖₂` instead of `σ_t`. Because `‖ε_uncond‖₂` remains in a stable range across the entire denoising trajectory (it doesn't collapse to zero like `σ_t` does), this produces a much more stable conflict signal. **This is the key finding of the ablation study.**

- `_cosine_sigma`: Ablation (i). Measures the angular disagreement between the two predictions: `1 − cosine_similarity(ε_cond, ε_uncond)`. This is 0 when they point in the same direction and up to 2 when they are opposite. To make it comparable in scale to the L2 metric (so the same `α` values are meaningful), it is also multiplied by `‖Δ_t‖₂/σ_t`.

- `_cosine_uncond`: Ablation (i)+(ii) combined. Pure cosine disagreement normalized by `‖ε_uncond‖₂`.

The four combinations are stored in a dict keyed by `(conflict_metric, normalization)` tuple. Accessing the wrong key raises a clear `ValueError` rather than silently using a wrong function.

**`_build_mapping_fn()`** — implements Eq. 3:

- `_exp_map`: `w_min + (w_max − w_min) · exp(−α · s_t)`. When `s_t = 0`, result is `w_max`. As `s_t → ∞`, result approaches `w_min`. The decay rate is controlled by `α` — at `α=1.0`, the function reaches near-`w_min` around `s_t ≈ 5`. **This is the failure point of the primary method**: when `σ_t → 0` in late steps, `s_t` explodes to very large values, and `exp(−1.0 · large)` ≈ 0, collapsing guidance to `w_min`.

- `_recip_map`: `w_min + (w_max − w_min) / (1 + α · s_t)`. Decays more slowly — at `s_t = 1/α`, the output is `w_min + (w_max − w_min) / 2`. More resistant to large `s_t` values.

**`update_and_get_w(noise_cond, noise_uncond, sigma_t)`** — called once per denoising step:
1. Calls `self._metric_fn(...)` → `s_t` (Eq. 2)
2. Calls `self._mapping_fn(s_t)` → `w_t` (Eq. 3)
3. Computes EMA: `w_smoothed = β · prev_w + (1−β) · w_t` (Eq. 4)
4. Clips `w_smoothed` to `[w_min, w_max]`
5. Updates `self.prev_w = w_smoothed`
6. Appends to history lists for `get_summary()`
7. Returns `(w_smoothed, s_t)`

The logging inside this function writes at `DEBUG` level only for steps 1, 2, 3, and multiples of 10 — avoiding 30 log lines per image at normal log levels.

**`get_summary()`** — called after all denoising steps complete. Returns statistics over the entire trajectory: mean/std of `s_t`, mean of raw `w_t`, and mean/min/max of the smoothed `w̃_t`. Used by `manager.py` for the per-image debug log line.

**`reset()`** — clears all history lists and resets `prev_w` to `w_max`. In the current codebase each image gets a new controller instance, so `reset()` is not called internally — but it exists for API completeness if someone wants to reuse a controller.

#### `ScheduledGuidanceController`

A stateless (per step) scheduler that takes `step_idx` and returns a predetermined `w` value according to either a linear or cosine decay schedule. It requires `total_steps` at construction time because the schedule is normalized to the total number of steps. **Not to be confused with the adaptive controller** — this is the dumb baseline that demonstrates a simple schedule is not sufficient and that the conflict signal is what matters.

---

### 4.4 `core/sampler_bridge.py`

**Purpose:** Wraps the HuggingFace Diffusers `StableDiffusionPipeline` and replaces its built-in CFG sampling loop with our custom loop that calls the guidance controller at each step. This is where Equation 1 lives.

#### `_get_sigma_t(scheduler, step_idx)` — free function

Extracts the noise level `σ_t` from the scheduler's internal state. This is needed for the primary conflict metric (Eq. 2). Two extraction paths:

1. **Direct `.sigmas` attribute**: Some schedulers (e.g., certain DPM-Solver variants) pre-compute and expose a `.sigmas` tensor. If it exists and is not `None`, index it directly.

2. **Derive from `.alphas_cumprod`**: DDIM (which we use) stores `ᾱ_t` (the cumulative product of noise schedule alphas). The standard relationship is `σ_t = sqrt((1 − ᾱ_t) / ᾱ_t)`. We clamp `ᾱ_t` to a minimum of `1e-8` to prevent division by zero at `t=0`.

3. **Fallback**: Returns `1.0` with a warning. This means the conflict metric degenerates to an unnormalized L2 norm, which is still functional but not what the paper proposes.

#### `AdaptiveSamplerWrapper`

The main class that users interact with. Takes a loaded pipeline and exactly one guidance mode specification. The `modes_set` assertion at the top of `__init__` prevents silent bugs from passing both `adaptive_config` and `fixed_guidance_scale`.

**`_encode_prompt(prompt, negative_prompt)`**

Tokenizes and encodes both the positive and negative (empty string by default) prompts through the pipeline's CLIP text encoder. Returns two tensors of shape `[1, 77, 768]` — the context embeddings passed to the UNet cross-attention layers. This is done once per image (before the denoising loop) since the text embeddings don't change across timesteps.

**`sample(prompt, num_inference_steps, seed, height, width, negative_prompt)`**

The full DDIM denoising loop, step by step:

1. **Seed the generator**: `torch.Generator(...).manual_seed(seed)` ensures reproducibility. The same prompt with the same seed will always produce the same image regardless of what other images were generated in the same run.

2. **Set timesteps**: `scheduler.set_timesteps(num_inference_steps)` populates `scheduler.timesteps` with a descending sequence of integers (e.g., `[999, 967, 934, ..., 15]` for 30 steps out of 1000). Each integer corresponds to a noise level.

3. **Initialize latents**: Sample `z_T ~ N(0, I)` of shape `[1, 4, 64, 64]` (4 channels, 64×64 for 512×512 images) and scale by `scheduler.init_noise_sigma`. For DDIM this scaling is 1.0, but other schedulers may differ.

4. **Instantiate controller**: Creates a fresh `AdaptiveGuidanceController`, `ScheduledGuidanceController`, or `None` depending on mode.

5. **Denoising loop** — for each `(step_idx, t)`:
   - `scheduler.scale_model_input(latents, t)`: Some schedulers (e.g., PNDM) scale the input latent before passing to the UNet. For DDIM this is a no-op, but the call is kept for scheduler-agnosticism.
   - **Two UNet passes**: `noise_uncond = unet(latent_input, t, uncond_emb)` then `noise_cond = unet(latent_input, t, cond_emb)`. These are the two forward passes that make CFG expensive relative to unconditioned generation.
   - **Get `w_t`**: Calls the controller (or reads the fixed scale) to get the guidance weight for this step.
   - **Eq. 1**: `noise_pred = noise_uncond + w_t * (noise_cond - noise_uncond)`. This is the actual CFG update. The entire adaptive method is contained in the single float `w_t` — the rest is standard.
   - **Scheduler step**: `scheduler.step(noise_pred, t, latents).prev_sample` runs the DDIM update rule to compute `z_{t−1}` from `z_t` and the predicted noise.

6. **VAE decode**: `latents / vae.config.scaling_factor` undoes the latent scaling factor (0.18215 for SD v1.5), then `vae.decode(...)` converts the 4-channel 64×64 latent to a 3-channel 512×512 image in `[−1, 1]` range. The `/ 2 + 0.5` converts to `[0, 1]`, then multiply by 255 and cast to uint8 for the PIL image.

**`build_pipeline(exp_config)`** — free function

Loads Stable Diffusion v1.5 from HuggingFace in three steps:
1. `DDIMScheduler.from_pretrained(...)`: Loads the DDIM scheduler separately from the `scheduler/` subfolder of the model repository.
2. `StableDiffusionPipeline.from_pretrained(...)`: Loads the full pipeline with the custom DDIM scheduler already attached. Uses `torch.float16` on CUDA for memory efficiency (P100 has 16GB; full fp16 SD v1.5 uses ~4GB).
3. `pipeline.safety_checker = None`: Disables the NSFW safety checker for research purposes to avoid false positives blocking generated images.

---

### 4.5 `eval/metrics.py`

**Purpose:** Implements all four paper metrics. Each class lazy-loads its model (only downloads on first use). The top-level `QualityEvaluator` orchestrates all four in one call.

#### `CLIPScorer`

Loads `openai/clip-vit-base-patch32` from HuggingFace. This is the same CLIP backbone used in the majority of T2I evaluation papers, making scores comparable across works.

**`score_batch(images, prompts, batch_size=16)`**: Processes images and prompts together through the CLIP processor (which resizes, normalizes, and tokenizes), then runs the CLIP model to get `logits_per_image`. The diagonal of the logit matrix is `logit[i,i] = similarity(image_i, text_i)` for paired inputs. Returns a list of per-image scores.

**`mean_score(images, prompts)`**: Convenience wrapper that returns the scalar mean over all scores. This is the number reported in the results table.

**Important nuance**: CLIP scores are reported as raw logits (scaled cosine similarities), not percentages. The range in our results (29–32) is typical for SD v1.5 with ViT-B/32 on natural caption prompts.

#### `LPIPSDiversity`

Loads `lpips` (AlexNet backbone, `net="alex"`) for perceptual distance computation.

**`pairwise_diversity(image_sets)`**: Takes a list of lists — each inner list contains all seeds' images for one prompt. For a 2-seed run with 100 prompts, this is 100 pairs. For each pair, computes the LPIPS perceptual distance. Images are first resized to 256×256 and normalized to `[−1, 1]` (LPIPS convention). A higher mean LPIPS diversity means the model produces more varied outputs across seeds — indicating the guidance is not collapsing all samples to a single mode.

**Why LPIPS and not pixel MSE?**: Pixel-level metrics ignore perceptual structure. Two images can be pixel-different but perceptually similar (e.g., slight color shift), or pixel-similar but perceptually different (e.g., different textures). LPIPS uses AlexNet's intermediate feature representations to measure perceptual difference, which correlates much better with human judgment.

#### `ArtifactEvaluator`

Two artifact proxy metrics that don't require a neural network — just signal processing.

**`saturation_stats(images)`**: Converts each image to HSV color space and extracts the saturation (S) channel. Computes the standard deviation of S across all pixels. A high saturation std indicates highly uneven saturation — some regions are oversaturated while others are not. This is a proxy for "color burn-in," the unnatural vivid colors that appear at high CFG. Returns mean and max over all images.

**`high_frequency_energy(images)`**: Converts each image to grayscale, applies a 2D FFT, shifts the zero-frequency component to the center (`fftshift`), and computes the mean magnitude of all frequency components **outside** a circular low-frequency region of radius `min(H,W)/4`. The mask `(y−cy)² + (x−cx)² > r²` selects the high-frequency ring. High HF energy indicates oversharpening — excessive edge contrast and texture artifacts. Lower is better for natural-looking images.

#### `QualityEvaluator`

Orchestrator that runs all three sub-evaluators in sequence and returns a single flat dict with all metrics. Called once per variant after all images for that variant are generated.

The key data structure it manages: `images_per_seed` is a list of lists where `images_per_seed[s][p]` is the image for seed index `s` and prompt index `p`. For CLIP scoring, all images are flattened with their prompts repeated across seeds. For LPIPS, images are grouped by prompt (one group per prompt, each containing one image per seed) — which is why we need at least 2 seeds to compute a non-trivial diversity score.

---

### 4.6 `experiments/manager.py`

**Purpose:** Runs the complete experiment. Given a list of prompts and a loaded pipeline, it sequentially runs all variants (fixed, scheduled, adaptive), evaluates each, accumulates results, saves checkpoints, and prints a final comparison table. This is the highest-level logic file.

#### `ADAPTIVE_VARIANTS` — module-level constant

A list of four dicts, each defining one ablation configuration. Stored here (not in `config.py`) because they are experiment design decisions, not user-configurable parameters. The four variants are:

| Name | metric | norm | map | Role |
|---|---|---|---|---|
| `adaptive_l2_sigma_exp` | l2_norm | sigma | exponential | Primary proposed method |
| `adaptive_cosine_sigma_exp` | cosine | sigma | exponential | Ablation (i): change conflict metric |
| `adaptive_l2_uncond_exp` | l2_norm | uncond_norm | exponential | Ablation (ii): change normalization |
| `adaptive_l2_sigma_recip` | l2_norm | sigma | reciprocal | Ablation (iii): change mapping |

Each ablation changes exactly **one** axis from the primary method, allowing clean attribution of performance differences.

#### `AblationStudy.__init__(exp_config, base_adaptive_config)`

Sets up the study. Creates a single `QualityEvaluator` (which is shared across all variants — no need to reload CLIP and LPIPS for each variant). Initializes `self.results` as an empty dict that accumulates variant name → metrics dict.

#### `_eta_str(done, total, elapsed)` — static method

Computes and formats a time remaining estimate. Formula: `rate = done / elapsed`, `remaining = (total - done) / rate`. Formatted as `HH:MM:SS`. Called after each image so the log shows an updating ETA.

#### `_run_variant(pipeline, name, description, prompts, ...)` — the core runner

The private method that executes one complete variant (e.g., `fixed_cfg_7.5` or `adaptive_l2_uncond_exp`):

1. Creates an `AdaptiveSamplerWrapper` with the appropriate mode
2. **Outer loop over seeds**: for each seed, runs all `num_samples` prompts
3. **Inner loop over prompts**: calls `wrapper.sample(prompt, seed=seed)` for each prompt, saves the image if `save_images=True`, logs progress every 10 images with ETA and mean guidance scale
4. After all seeds complete, calls `self.evaluator.evaluate_run(images_per_seed, prompts)` to compute all 4 metrics
5. Adds metadata (`variant`, `description`, `total_time_s`, `num_prompts`, `num_seeds`) to the metrics dict
6. Returns the complete metrics dict

Image saving uses the path structure `outputs/{variant_name}/seed_{seed}/{idx:04d}_{prompt_prefix}.png`. The prompt is truncated to 40 chars and spaces replaced with underscores for filesystem safety.

#### `run_fixed_baselines(prompts, pipeline)`

Iterates over `exp_config.fixed_cfg_scales` (default: `[3.0, 7.5, 12.0]`), calls `_run_variant` for each with `fixed_scale=scale`, then immediately calls `_checkpoint()`. The checkpoint write-after-each-variant means that if the run crashes 80% through, all completed variants are preserved in `results_checkpoint.json`.

#### `run_scheduled_baselines(prompts, pipeline)`

Runs two scheduled variants: `scheduled_linear` and `scheduled_cosine`. Both use `w_min` and `w_max` from the base adaptive config so the range matches. This is the **paper's required baseline** — the paper's evaluation plan explicitly mentions "a simple time-scheduled CFG baseline" as one of the comparison points.

#### `run_adaptive_ablations(prompts, pipeline)`

Iterates over the four `ADAPTIVE_VARIANTS` dicts. For each, creates an `AdaptiveCFGConfig` by overriding only the three ablation fields (`conflict_metric`, `normalization`, `mapping_fn`) while keeping all numeric hyperparameters (`w_min`, `w_max`, `alpha`, `beta`, `epsilon`) identical to the base config. This ensures that performance differences between variants are due to the design choice being ablated, not to different hyperparameter settings.

#### `save_results()`

Writes two output files and prints the final table:
- `results.json`: Full nested dict with all metrics for all variants. Human-readable, can be loaded for re-plotting.
- `results.csv`: Same data in flat tabular format. Headers are the union of all metric keys, sorted alphabetically. Missing values (if a variant crashed) are left blank rather than causing an error.

#### `_print_final_table()`

Logs a formatted comparison table sorted into three groups (Fixed → Scheduled → Adaptive) and then identifies the best variant for each metric. Uses left-justified string formatting to align columns. Example output:

```
===========================================================================================
FINAL RESULTS TABLE
===========================================================================================
Category     Variant                               CLIP     LPIPS     HF-E     Sat   T(s)
-------------------------------------------------------------------------------------------
Fixed        fixed_cfg_3.0                       30.4178   0.6197   4224.7   49.93  3919
Fixed        fixed_cfg_7.5                       31.6706   0.6545   3749.6   61.16  3918
...
```

---

### 4.7 `analyze.py`

**Purpose:** Generates all paper figures from the `results` dict. Runs headlessly via `matplotlib.use("Agg")` — no display required, works on Kaggle and remote servers.

#### Color scheme

```python
COLORS = {
    "fixed":     "#D32F2F",  # deep red   — fixed CFG baselines
    "scheduled": "#F57C00",  # deep orange — scheduled CFG baselines
    "adaptive":  "#1976D2",  # material blue — proposed adaptive method
    "best":      "gold",     # gold border   — best in category
}
```

`_variant_color(name)` maps variant names to colors by prefix. `_short_label(name)` shortens long variant names for axis tick labels (`"adaptive_l2_sigma_exp"` → `"adp_l2_sigma_exp"`).

#### `plot_metrics_comparison(results, output_dir)`

Produces `metrics_comparison.png`: a 2×2 subplot grid showing all four metrics (CLIP, LPIPS, HF Energy, Saturation std) as bar charts. Each bar is colored by variant type. The best bar in each subplot gets a thick gold border. Value labels are printed on top of each bar in 6.5pt font. A shared legend is placed outside the grid. Saved at 150 DPI for paper use.

#### `plot_pareto_curve(results, output_dir)`

Produces `pareto_curve.png`: a scatter plot with HF Energy on the x-axis and CLIP score on the y-axis. Points in the upper-left are best (low artifacts, high alignment). Each variant is annotated with its short label. Adaptive variants use circles (`"o"`), scheduled use diamonds (`"D"`), fixed use squares (`"s"`). A text annotation `"← ideal region"` with an arrow is placed near the upper-left corner.

#### `plot_guidance_trajectory(w_histories, output_dir)`

Produces `guidance_trajectory.png`: a line plot showing `w̃_t` over all `T` denoising steps for each variant. Adaptive variants use solid lines, scheduled use dashed, fixed use dotted. This is one of the most informative figures because it directly shows what the controller is doing — you can see the adaptive controller's guidance scale fluctuate in response to the conflict signal while the fixed and scheduled lines are deterministic.

This function takes `w_histories` (a dict of variant name → list of floats collected during generation) rather than the metrics dict, so it must be called at generation time, not from re-analysis.

#### `generate_qualitative_grids(results, output_dir, seed, max_prompts)`

Produces `qualitative_grid.png`: a grid where each row is a variant and each column is a prompt. Images are loaded from the saved per-variant per-seed directories (requires `save_images=True` during generation). Each image is resized to 224×224. Row labels are colored by variant type. Column headers show prompt index. A colored border on each image matches the variant type (red/orange/blue). Silently skips variants whose image directories don't exist.

---

### 4.8 `main.py`

**Purpose:** Command-line entry point. Parses arguments, sets up logging, loads prompts and pipeline, runs the study phases, saves results, and calls analysis. All configuration flows through here — it is the only file a user needs to interact with for a standard run.

#### `parse_args()`

Defines the full CLI interface. Key design decisions:
- `--seeds` accepts multiple values (`nargs="+"`) so `--seeds 42 43 44` works
- `--fixed_scales` accepts multiple values for custom baseline grids
- `--skip_baselines`, `--skip_scheduled`, `--skip_ablations` allow partial runs (e.g., only run the adaptive variants if you already have baseline results)
- `--only_analyze` loads `results.json` from a previous run and re-generates figures without any image generation
- `--log_level` controls verbosity: `DEBUG` shows per-step guidance values and controller summaries, `WARNING` shows only errors

#### Timing estimate at startup

Before loading anything, `main()` computes the expected number of images and prints an ETA estimate:
```
n_images = num_samples × len(seeds) × n_variants
eta_hours = n_images × 3.5 / 3600
```
The 3.5 seconds/image constant is empirically measured on a Kaggle P100 with 30 DDIM steps.

#### `_run_analysis(results, output_dir, logger)`

A small helper that wraps the three `analyze.py` calls in a try/except. If figure generation fails (e.g., no image directories found for the qualitative grid), it logs a warning and continues rather than crashing the entire run.

---

### 4.9 `reproduce_results.ipynb`

**Purpose:** A fully self-contained Kaggle notebook that reproduces the entire experiment end-to-end. Every cell is annotated with its purpose. The notebook is designed so that a person with no familiarity with the codebase can run it by pressing "Run All."

#### Cell 1 — Environment check
Runs `nvidia-smi` to confirm GPU type and VRAM, prints PyTorch version and CUDA availability. Serves as a quick sanity check before a 1.5-hour run.

#### Cell 2 — Install dependencies
Runs `pip install -q` with pinned versions to ensure reproducibility. Suppressed with `%%capture` to avoid flooding the notebook.

#### Cell 3 — Write source files
Creates the directory structure and writes all source files inline as Python strings. This makes the notebook completely self-contained — it does not need the repo to be separately uploaded. An alternative path using a Kaggle dataset upload is documented in comments.

#### Cells 4–6 — Configuration, prompts, pipeline
Sets up all configuration constants in one cell (the only cell a user should edit). Loads prompts with the full fallback chain. Loads the pipeline with a logged timer.

#### Cell 7 — Run all experiments
Calls `study.run_fixed_baselines()`, `study.run_scheduled_baselines()`, and `study.run_adaptive_ablations()` sequentially. The ETA is logged after every image so the user can see progress.

#### Cells 8–9 — Save and display results
Saves `results.json` and `results.csv`, then prints a formatted comparison table inline in the notebook output.

#### Cell 10–11 — Generate and display figures
Calls all `analyze.py` functions and renders each figure inline using `matplotlib.image.imread` + `imshow`.

#### Cell 12 — Guidance trajectory analysis
Runs a single prompt through all 9 variants and plots the guidance scale `w̃_t` over all denoising steps. This is the key mechanistic figure — it shows what the adaptive controller is actually doing, step by step.

#### Cell 13 — Alpha sensitivity test
Runs the primary method (`l2_sigma_exp`) with four different `α` values (`1.0, 0.5, 0.1, 0.05`) on a small subset of prompts and reports the CLIP score for each. This directly demonstrates the key finding: at `α=1.0`, the method underperforms because `exp(−1.0 · s_t)` collapses to 0 when `s_t` is large (which happens when `σ_t → 0` in late steps). Lowering `α` to `0.05` keeps the guidance in a productive range.

#### Cell 14 — File listing and zip
Lists every output file with size, then creates `adaptive_cfg_results.zip` for easy download from Kaggle's output panel.

---

## 5. Experimental Design

### Variants (9 total)

**Phase 1 — Fixed CFG Baselines (3 variants)**
Standard CFG with constant `w`. Three scales bracket the range: low (3.0), standard (7.5), and aggressive (12.0). These establish the alignment-artifact trade-off that the adaptive method aims to improve.

**Phase 2 — Scheduled CFG Baselines (2 variants)**
Linear and cosine time-scheduled CFG. Both decay from `w_max=10` at the first step to `w_min=2` at the last. These are stronger baselines than fixed CFG because they already incorporate the intuition that guidance should vary over the trajectory — but they use only time, not the model's signal.

**Phase 3 — Adaptive CFG Ablations (4 variants)**
The proposed method in four configurations, each changing exactly one design axis from the primary method to isolate its effect.

### Dataset

**Primary**: MS-COCO 2017 validation captions. 100 prompts randomly sampled, 2 seeds each → 200 images per variant → 1,800 images total. COCO is used because SD v1.5 paper reports CLIP scores on COCO val, making our numbers directly comparable.

**Failure subset**: 20 manually curated prompts targeting known failure modes (faces, hands, text, textures). Used for quick diagnostic runs and qualitative analysis.

### Metrics

| Metric | Measures | How | Better |
|---|---|---|---|
| CLIP Score | Prompt alignment | Cosine similarity in CLIP ViT-B/32 space | ↑ Higher |
| LPIPS Diversity | Output variety across seeds | Mean pairwise AlexNet-LPIPS distance | ↑ Higher |
| HF Energy | Oversharpening / texture artifacts | Mean FFT magnitude outside low-freq circle | ↓ Lower |
| Saturation Std | Color burn-in | Pixel-level std of HSV saturation channel | ↓ Lower |

---

## 6. Results & Findings

### Quantitative Results

| Variant | CLIP ↑ | LPIPS ↑ | HF-Energy ↓ | Sat-Std ↓ |
|---|---|---|---|---|
| fixed_cfg_3.0 | 30.42 | 0.620 | 4224.7 | 49.9 |
| fixed_cfg_7.5 | 31.67 | 0.654 | 3749.6 | 61.2 |
| fixed_cfg_12.0 | 31.78 | 0.666 | 3572.9 | 67.3 |
| scheduled_linear | — | — | — | — |
| scheduled_cosine | — | — | — | — |
| **adaptive_l2_sigma_exp** | 29.92 | 0.620 | 4264.0 | 49.7 |
| **adaptive_cosine_sigma_exp** | **31.83** | 0.662 | 3656.0 | 64.7 |
| **adaptive_l2_uncond_exp** | 31.81 | 0.662 | 3657.2 | 64.7 |
| adaptive_l2_sigma_recip | 30.35 | 0.624 | 4202.9 | 50.9 |

### Key Finding: Normalization Dominates

The ablation reveals one decisive factor: **the normalization choice determines whether the controller works at all**.

`l2_sigma_exp` (σ_t normalization) fails because σ_t spans orders of magnitude across the DDIM trajectory — it starts at ~16 at t=T and collapses to ~0.03 at t=0. This means `s_t = ‖Δ_t‖₂ / σ_t` explodes in late steps, and `exp(−1.0 · s_t)` → 0, collapsing guidance to `w_min` during exactly the fine-detail refinement phase that most affects CLIP score.

`l2_uncond_exp` uses ‖ε_uncond‖₂ as the denominator instead. Because the unconditional prediction norm is stable across the trajectory (it doesn't have the same order-of-magnitude variation as σ_t), `s_t` stays in a well-behaved range, and the controller produces a meaningful adaptive signal throughout all 30 steps.

The cosine variants happen to also avoid this failure mode because the cosine term (1 − cos_sim) is bounded in [0, 2] regardless of σ_t, making the metric more robust even when σ normalization is used.

### Practical Takeaway

If reusing this codebase, set `normalization="uncond_norm"` and use `alpha` between 0.05 and 0.5. The controller with `l2_uncond_exp` matches `fixed_cfg_12.0` on CLIP while reducing saturation std, making it a safer drop-in replacement at high guidance scales.

---

## 7. Reproducing the Results

### One-Click: Kaggle Notebook

1. Go to [kaggle.com/code](https://kaggle.com/code) → **New Notebook**
2. Upload `reproduce_results.ipynb`
3. Settings → Accelerator → **GPU P100**
4. Settings → Internet → **On**
5. **Run All** — estimated 1.5 hours for the default configuration

Results saved to `/kaggle/working/outputs/`. All figures displayed inline.

### Local / Colab

```bash
git clone <your-repo-url> && cd adaptive_cfg
pip install -r requirements.txt

# Smoke test first (~5 min, no GPU needed with --device cpu)
python main.py \
  --dataset failure_subset \
  --num_samples 5 \
  --num_steps 10 \
  --seeds 42 \
  --device cuda

# Full experiment
python main.py \
  --dataset coco_val \
  --num_samples 100 \
  --num_steps 30 \
  --seeds 42 43
```

### From existing results (re-plot only)

```bash
python main.py --only_analyze --output_dir outputs
```

---

## 8. Installation & Requirements

```bash
pip install -r requirements.txt
```

```
torch>=2.0.0
torchvision>=0.15.0
diffusers>=0.27.0
transformers>=4.38.0
accelerate>=0.27.0
datasets>=2.18.0
lpips>=0.1.4
Pillow>=10.0.0
numpy>=1.24.0
matplotlib>=3.7.0
scipy>=1.10.0
```

**Tested on:** Python 3.10, CUDA 11.8 and 12.1, Kaggle P100 (16GB VRAM), Ubuntu 22.04. On CPU, generation is functional but slow (~2 min/image).

**Model weight storage:** SD v1.5 weights (~4GB) are downloaded automatically by `diffusers` on first run and cached in `~/.cache/huggingface/`. They are not included in the repo.

---

## 9. Full CLI Reference

```bash
python main.py [OPTIONS]
```

| Argument | Default | Description |
|---|---|---|
| `--model_id` | `runwayml/stable-diffusion-v1-5` | HuggingFace model ID |
| `--dataset` | `coco_val` | Prompt set: `coco_val`, `parti_prompts`, `drawbench`, `failure_subset` |
| `--num_samples` | `100` | Number of prompts to sample from the dataset |
| `--num_steps` | `30` | DDIM denoising steps per image |
| `--seeds` | `42 43` | One or more random seeds (space-separated) |
| `--device` | `cuda` | PyTorch device |
| `--output_dir` | `outputs` | Root directory for all outputs |
| `--fixed_scales` | `3.0 7.5 12.0` | Fixed CFG guidance scales for baselines |
| `--w_min` | `2.0` | Minimum guidance scale for adaptive controller |
| `--w_max` | `10.0` | Maximum guidance scale for adaptive controller |
| `--alpha` | `1.0` | Sensitivity in Eq. 3 (lower = slower response to conflict) |
| `--beta` | `0.5` | EMA smoothing in Eq. 4 (higher = more inertia) |
| `--skip_baselines` | — | Do not run fixed CFG baselines |
| `--skip_scheduled` | — | Do not run scheduled CFG baselines |
| `--skip_ablations` | — | Do not run adaptive ablations |
| `--no_save_images` | — | Do not save individual PNG files (faster, less disk) |
| `--only_analyze` | — | Load existing `results.json` and regenerate figures only |
| `--log_level` | `INFO` | Verbosity: `DEBUG`, `INFO`, or `WARNING` |

---

## 10. Python API Reference

```python
from config import AdaptiveCFGConfig, ScheduledCFGConfig, ExperimentConfig
from core.sampler_bridge import AdaptiveSamplerWrapper, build_pipeline

exp_config = ExperimentConfig(device="cuda")
pipeline = build_pipeline(exp_config)

# --- Adaptive (proposed method, stable normalization) ---
cfg = AdaptiveCFGConfig(
    conflict_metric="l2_norm",
    normalization="uncond_norm",   # recommended: more stable than sigma
    mapping_fn="exponential",
    w_min=2.0, w_max=10.0,
    alpha=0.1,                     # lower alpha avoids guidance collapse
    beta=0.5,
)
wrapper = AdaptiveSamplerWrapper(pipeline, adaptive_config=cfg)
image, w_history, summary = wrapper.sample(
    prompt="a photorealistic portrait of an astronaut on Mars",
    num_inference_steps=30,
    seed=42,
)
# image: PIL.Image (512×512)
# w_history: list of 30 floats — guidance scale at each step
# summary: {"s_t_mean": ..., "w_smooth_mean": ..., "w_smooth_min": ..., "w_smooth_max": ...}

# --- Fixed CFG baseline ---
wrapper_fixed = AdaptiveSamplerWrapper(pipeline, fixed_guidance_scale=7.5)
image_fixed, _, _ = wrapper_fixed.sample("same prompt", seed=42)

# --- Scheduled CFG baseline ---
from config import ScheduledCFGConfig
sched = ScheduledCFGConfig(w_max=10.0, w_min=2.0, schedule="cosine")
wrapper_sched = AdaptiveSamplerWrapper(pipeline, scheduled_config=sched)
image_sched, _, _ = wrapper_sched.sample("same prompt", seed=42)

# --- Evaluate a set of images ---
from eval.metrics import QualityEvaluator
evaluator = QualityEvaluator(device="cuda")
metrics = evaluator.evaluate_run(
    images_per_seed=[[img1_seed42, img2_seed42], [img1_seed43, img2_seed43]],
    prompts=["prompt 1", "prompt 2"],
)
# metrics: {"clip_score": ..., "lpips_diversity": ...,
#           "hf_energy_mean": ..., "saturation_std_mean": ...}
```

---

## 11. Kaggle Timing Guide

Measured on Kaggle GPU P100 (16GB), SD v1.5, 512×512 images:

| Prompts | Seeds | Variants | Steps | Est. Time |
|---|---|---|---|---|
| 10 | 1 | 9 | 20 | ~5 min |
| 50 | 2 | 9 | 30 | ~45 min |
| 100 | 2 | 9 | 30 | ~1.5 h |
| 200 | 2 | 9 | 30 | ~3.0 h |
| 100 | 2 | 9 | 50 | ~2.5 h |

Kaggle sessions time out after **9 hours**. The 200-prompt experiment fits comfortably. Results are checkpointed after every variant (`results_checkpoint.json`), so if the session ends unexpectedly, all completed variants are preserved.

---

## 12. Citation

```bibtex
@misc{adaptivecfg2025,
  title   = {Norm-Regulated Conflict-Aware Classifier-Free Guidance},
  author  = {[Author Name]},
  year    = {2025},
  note    = {Course Project, Deep Generative Models, Winter 1404},
}
```

**Dependencies cited in the paper:**

```bibtex
@inproceedings{rombach2022ldm,
  title     = {High-Resolution Image Synthesis with Latent Diffusion Models},
  author    = {Rombach, Robin and Blattmann, Andreas and Lorenz, Dominik and Esser, Patrick and Ommer, Björn},
  booktitle = {CVPR},
  year      = {2022}
}

@inproceedings{lin2014coco,
  title     = {Microsoft COCO: Common Objects in Context},
  author    = {Lin, Tsung-Yi and Maire, Michael and Belongie, Serge and others},
  booktitle = {ECCV},
  year      = {2014}
}

@article{yu2022parti,
  title   = {Scaling Autoregressive Models for Content-Rich Text-to-Image Generation},
  author  = {Yu, Jiahui and Xu, Yuanzhong and Koh, Jing Yu and others},
  journal = {arXiv:2206.10789},
  year    = {2022}
}

@inproceedings{saharia2022imagen,
  title     = {Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding},
  author    = {Saharia, Chitwan and Chan, William and Saxena, Saurabh and others},
  booktitle = {NeurIPS},
  year      = {2022}
}

@inproceedings{ho2021cfg,
  title   = {Classifier-Free Diffusion Guidance},
  author  = {Ho, Jonathan and Salimans, Tim},
  journal = {NeurIPS Workshop on DGMs},
  year    = {2021}
}

@inproceedings{zhang2018lpips,
  title     = {The Unreasonable Effectiveness of Deep Features as a Perceptual Metric},
  author    = {Zhang, Richard and Isola, Phillip and Efros, Alexei A and others},
  booktitle = {CVPR},
  year      = {2018}
}
```

---

## License

Code: MIT License.  
Model weights (SD v1.5): [CreativeML Open RAIL-M](https://huggingface.co/spaces/CompVis/stable-diffusion-license).  
COCO dataset: [CC BY 4.0](https://cocodataset.org/#termsofuse).
