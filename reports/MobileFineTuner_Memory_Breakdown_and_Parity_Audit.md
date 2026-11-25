# MobileFineTuner: Memory Breakdown, 5-Step Parity Fix & On-Device System Audit

**Last Updated:** 2026-11-25  
**Scope:** Root-cause resolution of (1) PyTorch vs. C++ 5-step parity divergence, (2) why the initial phone Q8_0 run hit $3.04\text{ GB}$ RSS, and (3) honest accounting of the USB-plugged phone plumbing check vs. upcoming unplugged soak.

> [!NOTE]
> **Retraction Notice for Earlier Toy Reports:** All November 19–21 toy-setup numbers (`MobileFineTuner_Multimodal_Evaluation_Report.md`, `MobileFineTuner_Meeting_Slides.pdf`, and toy-backbone sections of `MobileFineTuner_Empirical_Audit_and_Verification.md`) are **RETRACTED** and must not be cited.

---

## 1. PyTorch vs. C++ 5-Step Parity Divergence: Diagnosed & Fixed

Earlier, 5 training steps on identical initial weights and a fixed batch (`data/parity_test`) matched at Step 1 (`5.1204` vs `5.1202`), but diverged by **3.54%** at Step 3 and **2.82%** at Step 5.

### Root Causes Found via Step-1 Gradient & Weight Dumps
1. **Frozen `scale` Constant in C++ vs. Trainable `nn.Parameter` in PyTorch (Primary Cause):**
   - In [parity_5step_export.py](file:///home/assassinaj602/Projects/New%20folder%20(4)/github_pr/mobile_fine_tuner/qwen_multimodal_projector_worktree/parity_5step_export.py) and `colab/projector_reference.py`, `self.scale = nn.Parameter(...)` was updated by Adam (`grad(scale) = +4.6713`, dropping `scale` by $-6.5\%$ in Step 1 from `0.015478` to `0.014478`).
   - In `parity_5step_runner.cpp` and `train_projector.cpp`, `scale` was a non-trainable `float` constant excluded from `SimpleAdam`.
2. **GELU Variant (`tanh` approx vs. exact `erf`, Secondary ~0.03% Cause):**
   - C++ `ops::gelu` uses `0.5 * x * (1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))` (`nn.GELU(approximate='tanh')`), while `nn.GELU()` defaults to exact `erf` (`approximate='none'`).

### Step-1 Gradient & Weight Verification (Post-Fix)
Raw log: [reports/raw_logs/parity_5step_fixed.log](file:///home/assassinaj602/Projects/New%20folder%20(4)/github_pr/mobile_fine_tuner/qwen_multimodal_projector_worktree/reports/raw_logs/parity_5step_fixed.log)

| Parameter | Grad Cosine Sim (C++ vs PT exact) | Grad Rel Error | Post-Step-1 Weight Rel Error |
|:---|:---:|:---:|:---:|
| `ln_w` | `1.000000` | `5.558e-04` | `6.566e-05` |
| `ln_b` | `1.000000` | `5.216e-04` | `1.678e-05` |
| `W1` | `1.000000` | `5.361e-04` | `1.538e-03` |
| `b1` | `1.000000` | `5.373e-04` | `3.183e-03` |
| `W2` | `1.000000` | `5.314e-04` | `1.329e-03` |
| `b2` | `1.000000` | `5.581e-04` | `3.401e-03` |
| `scale` | `1.000000` | `5.238e-04` | `0.000e+00` (`0.014478` exact match) |
| `pos_embed` | `1.000000` | `6.365e-04` | `2.140e-03` |

### 5-Step Loss Comparison (After Making `scale` Trainable in C++)

| Step | PyTorch (`erf`) | PyTorch (`tanh`) | C++ Old (`scale` frozen) | C++ Fixed (`scale` trainable) | Rel Error vs PT (`erf` / `tanh`) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | `5.120436` | `5.120417` | `5.120150` | **`5.120154`** | `0.0056%` / `0.0052%` |
| **2** | `4.557662` | `4.557518` | `4.563620` | **`4.557185`** | `0.0106%` / `0.0074%` |
| **3** | `4.565256` | `4.564844` | `4.726650` | **`4.565556`** | `0.0067%` / `0.0157%` |
| **4** | `3.967787` | `3.967812` | `4.042100` | **`3.966750`** | `0.0261%` / `0.0268%` |
| **5** | `3.719689` | `3.718522` | `3.824760` | **`3.718312`** | `0.0371%` / `0.0057%` |

Maximum relative error across all 5 steps dropped from **`3.54%` down to `0.037%`** (and `0.027%` against `tanh` GELU). The same trainable `scale_` fix was applied to [train_projector.cpp](file:///home/assassinaj602/Projects/New%20folder%20(4)/github_pr/mobile_fine_tuner/qwen_multimodal_projector_worktree/examples/qwen_multimodal_projector/train_projector.cpp).

---

## 2. Why the Initial Phone Run Used 3.0 GB RSS & How We Fixed It

On the initial Q8_0 phone run (`2026-11-24`), `Base Model RAM upon Q8_0 load` was **$2,780.0\text{ MB}$** and Step 3 reached **$3,039.3\text{ MB}$** RSS while `MemAvailable` was only $1.60\text{ GB}$, forcing Android to push pages into ZRAM swap.

### Exact Cause of the 3.0 GB RSS
1. **Unconditional FP32 Weight Pre-Allocation (`+1,951 MB`):** `QwenModel::QwenModel(cfg)` pre-allocated all 290 FP32 tensors (`embed_tokens_` = $519.3\text{ MB}$ + 168 linear weights = $1,431.3\text{ MB}$), even when `AutoModelLoadOptions::load_weights = false` was passed for `--load_quantized`.
2. **Unreleased Linear Base Tensors (`+1,431 MB`):** `train_projector.cpp` only cleared `q_lin/k_lin/v_lin` base pointers, leaving `o_lin/gate_lin/up_lin/down_lin` and all `blk.*_proj_weight` `shared_ptr`s alive in memory alongside the $362.6\text{ MB}$ Q8_0 blocks.
3. **`MemoryPool` Retention:** `MemoryManager`'s pool caches freed tensor buffers internally without returning them to the OS until `MemoryManager::instance().clear_unused_memory()` is explicitly invoked.
4. **Full Logits Materialization (`+500 MB` Peak):** `qwen->lm_head(h)` materialized `[1, 98, 151936]` FP32 logits + slice + reshape + softmax + backward logits gradients (~$340\text{ MB}$+) even though only caption tokens (not `-100` prefix/prompt tokens) contribute to loss.

### Verified Post-Fix RSS Breakdown (Host Measurement)
Raw log: [reports/raw_logs/host_q8_rss_breakdown.log](file:///home/assassinaj602/Projects/New%20folder%20(4)/github_pr/mobile_fine_tuner/qwen_multimodal_projector_worktree/reports/raw_logs/host_q8_rss_breakdown.log)

Implemented fixes:
- Added `allocate_weights = false` path to `QwenModel::QwenModel` when `load_weights = false`.
- Reset all 7 projection base pointers (`q/k/v/o/gate/up/down`) and added `MemoryManager::instance().clear_unused_memory()` + `APP_TRIM_MEMORY()`.
- Switched default loss in `train_projector.cpp` to `selected_token_lm_cross_entropy` (fixing an `ops::full` `kInt32` bug and adding OpenMP parallelization in `lm_loss.cpp`).

| Stage | Old `VmRSS` | New `VmRSS` | New `VmHWM` (Peak) | Breakdown (`Q8` / `FP32 Model` / `MemPool Use`) |
|:---|:---:|:---:|:---:|:---|
| **Before Config Init** | &mdash; | $15.1\text{ MB}$ | $19.7\text{ MB}$ | Binary & runtime libraries |
| **After Empty Config Init** | &mdash; | $15.8\text{ MB}$ | $19.7\text{ MB}$ | `load_weights = false` no longer allocates FP32 tensors |
| **After Raw `.q8` Read** | &mdash; | $898.0\text{ MB}$ | $898.0\text{ MB}$ | $362.6\text{ MB}$ Q8_0 + $519.6\text{ MB}$ `embed_tokens` + $1.5\text{ MB}$ norms |
| **After Q8_0 Load & Trim** | $2,782.7\text{ MB}$ | **$898.0\text{ MB}$** | $1,417.1\text{ MB}$ | `Q8=362.6 MB`, `FP32=519.6 MB`, `MemPool_Use=519.6 MB` |
| **Step 1 After Fwd + Loss** | &mdash; | **$1,358.7\text{ MB}$** | $1,417.1\text{ MB}$ | `MemPool_Use=976.0 MB` (+$456\text{ MB}$ activations) |
| **Step 1 After Backward** | $3,772.8\text{ MB}$ | **$1,475.7\text{ MB}$** | $1,475.7\text{ MB}$ | `MemPool_Use=532.6 MB` (`Peak=1,086.4 MB`) |
| **Step 1 After Cleanup & Trim** | $3,772.8\text{ MB}$ | **$925.1\text{ MB}$** | **$1,486.9\text{ MB}$** | `MemPool_Use=531.9 MB` (Delta vs load: $+27.1\text{ MB}$) |

**Result:** Peak `VmHWM` dropped from **$3,772.8\text{ MB}$ down to $1,486.9\text{ MB}$** ($< 1.60\text{ GB}$ `MemAvailable`) and resting RSS dropped from **$3,039\text{ MB}$ down to $925.1\text{ MB}$**.  
*(Note: Using `--full_logits` keeps `VmHWM` at $1,938.6\text{ MB}$ vs $1,486.9\text{ MB}$ with selected-token loss, with identical loss: `4.9237` vs `4.9241`.)*

---

## 3. Phone Telemetry Status & Pending Unplugged Run

- **Prior 5-Step Phone Run (`2026-11-24`, Raw Log: [phone_5step_usb_plugged_initial.log](file:///home/assassinaj602/Projects/New%20folder%20(4)/github_pr/mobile_fine_tuner/qwen_multimodal_projector_worktree/reports/raw_logs/phone_5step_usb_plugged_initial.log)):**
  - **Plumbing check only:** 5 training steps (`4.99 -> 6.47 -> 6.44 -> 4.90 -> 4.67`), 5-image validation drop (`5.34 -> 4.88`), average step time $44.6\text{ s}$ (no BLAS).
  - **USB-Plugged Caveat:** Phone was connected to USB during telemetry polling, so battery current (`-382.6 mA` / `1,502.5 mW`) measured **net battery drain minus USB charging**, NOT total phone power.
  - **Throttling Caveat:** CPU frequencies were NOT logged in that run, so no claim of "zero throttling" is made.
  - **SoC Identity Verification Pending:** Device `ro.product.model` is `TECNO CH7n` (Tecno Camon 18P), which was previously labeled `Helio G88 (MT6781)` in older scripts and `Helio G96 (MT6781)` in newer notes. Raw `/proc/cpuinfo` (`CPU part` hex IDs) + `getprop` will be captured directly from the phone when re-attached.
- **Unplugged Test Script Prepared:** [scripts/android/run_unplugged_soak.sh](file:///home/assassinaj602/Projects/New%20folder%20(4)/github_pr/mobile_fine_tuner/qwen_multimodal_projector_worktree/scripts/android/run_unplugged_soak.sh) + updated Android binary (`build-android/train_projector`) are ready to push and run unplugged for 15 minutes once the phone is connected.
