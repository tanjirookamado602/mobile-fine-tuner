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

## 3. Verified On-Device Hardware, RSS & 15-Minute Unplugged Soak (Tecno Camon 18P)

### 3.1 SoC & CPU Part-ID Verification (From Raw `/proc/cpuinfo` & `getprop`)
Raw logs: [cpuinfo_raw.txt](file:///home/assassinaj602/Projects/New%20folder%20(4)/github_pr/mobile_fine_tuner/qwen_multimodal_projector_worktree/reports/raw_logs/unplugged_soak_20261127/cpuinfo_raw.txt), [getprop_raw.txt](file:///home/assassinaj602/Projects/New%20folder%20(4)/github_pr/mobile_fine_tuner/qwen_multimodal_projector_worktree/reports/raw_logs/unplugged_soak_20261127/getprop_raw.txt)
- **Model / Device:** `TECNO CH7n` (`TECNO-CH7n`, **Tecno Camon 18P**), Android `12` (`arm64-v8a`)
- **Platform / SoC:** `ro.board.platform = mt6781`, `ro.soc.model = MT6781V/CD` (**MediaTek Helio G96**)
- **CPU Cores:** `processor 0–5` = `CPU part: 0xd05` (**6× Cortex-A55**, up to `2000 MHz`); `processor 6–7` = `CPU part: 0xd0b` (**2× Cortex-A76**, up to `2050 MHz`). Features include `asimddp`, `fphp`, `asimdhp`. *(Supersedes the erroneous `Helio G88` label in earlier scripts.)*

### 3.2 On-Device Post-Fix 1-Step RSS Breakdown (`2026-11-27`)
Raw log: [phone_post_fix_1step_rss_breakdown.log](file:///home/assassinaj602/Projects/New%20folder%20(4)/github_pr/mobile_fine_tuner/qwen_multimodal_projector_worktree/reports/raw_logs/phone_post_fix_1step_rss_breakdown.log)
- Device `MemTotal`: `7,995,456 kB` (`7.63 GiB`) | `MemAvailable` before run: `2,293,304 kB` (`2,239.6 MB`)
- Base Model upon Q8_0 load: **`896.55 MB`** (`362.6 MB` Q8_0 + `519.6 MB` FP32 `embed_tokens` + runtime)
- After Fwd + Selected-Token Loss: `1,359.4 MB` (`MemPool_Use = 976.0 MB`)
- After Backward: `1,477.9 MB` (`VmHWM = 1,489.6 MB`, `Swap = 0.0 MB`)
- Post-step cleanup & trim: **`922.8 MB`** (`Fwd = 7.38 s`, `Bwd = 19.45 s`, `Total = 27.12 s`)

### 3.3 15-Minute Unplugged Battery, Thermal & Frequency Soak (`2026-11-27`)
Raw logs: [unplugged_telemetry.csv](file:///home/assassinaj602/Projects/New%20folder%20(4)/github_pr/mobile_fine_tuner/qwen_multimodal_projector_worktree/reports/raw_logs/unplugged_soak_20261127/unplugged_telemetry.csv), [train_unplugged.log](file:///home/assassinaj602/Projects/New%20folder%20(4)/github_pr/mobile_fine_tuner/qwen_multimodal_projector_worktree/reports/raw_logs/unplugged_soak_20261127/train_unplugged.log), [soak_status.txt](file:///home/assassinaj602/Projects/New%20folder%20(4)/github_pr/mobile_fine_tuner/qwen_multimodal_projector_worktree/reports/raw_logs/unplugged_soak_20261127/soak_status.txt)

| Metric | 60 s Unplugged Idle Baseline (`n=12`) | 15 min Unplugged Training (`n=165`, 23 Steps) | Notes |
|:---|:---:|:---:|:---|
| **Charging Status** | **100% `Discharging`** (12/12) | **100% `Discharging`** (165/165) | Total 178/178 samples strictly unplugged |
| **Battery %** | `67% -> 67%` | **`67% -> 56%`** (`-11%`) | 900 s active training |
| **Whole-Phone Power** | **`1,853.7 ± 391.8 mW`** (`1.85 W`) | **`5,173.0 ± 720.1 mW`** (`5.17 W`) | Peak `7,405.6 mW`; **Incremental compute = `+3.32 W`** |
| **Battery Current / Volt** | `-473.2 mA` @ `3.917 V` | `-1,382.7 mA` @ `3.742 V` mean | Voltage sag `3.917 V -> 3.721 V` |
| **Battery Temperature** | `31.6 -> 31.8 °C` | **`31.5 -> 42.8 °C`** (`+11.3 °C`) | Crossed `~42 °C` near end of soak |
| **CPU0 (A55) Clock** | `825 MHz` mean (`500–1,075`) | `1,993 MHz` mean (`975–2,000`) | Little cluster held near max `2.0 GHz` |
| **CPU6 (A76) Clock** | `774 MHz` | **`1,761 MHz` mean** (max `2,050`, end `1,530`) | **Thermal throttling observed** (~25% A76 drop) |
| **Memory (`VmRSS` / `VmHWM`)** | &mdash; | **`922.7 -> 914.8 MB` resting**; **`1,531.0 MB` `VmHWM` peak** | Zero memory leak across 23 steps; `Swap = 0.0 MB` |

**Honest Takeaways:**
1. The earlier USB-plugged `1.50 W` reading was an underestimate due to USB charging offset; true unplugged whole-phone power during Q8_0 training is **`5.17 W`** (`+3.32 W` above idle).
2. **Thermal throttling IS real under 15-minute sustained 8-core load:** As battery temp reached `42.8 °C`, the big Cortex-A76 cores throttled from `2,050 MHz` to `1,530 MHz`, and step time rose from `~28–31 s` (Steps 1–10) to `~36–47 s` (Steps 16–23).
3. Memory remained completely bounded (`1,531.0 MB` peak `VmHWM`, `914.8 MB` resting RSS across 23 steps) with zero OOM or swap pressure.

