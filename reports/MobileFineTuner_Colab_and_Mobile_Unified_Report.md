# MobileFineTuner: Corrected Unified Colab & Mobile Audit Report

**Last Updated:** 2026-11-25  
**Status:** Interim Audit & Code Fixes Complete; **Pending (a) 5+ Epoch Batched Colab/Kaggle Rerun and (b) 15-Minute Unplugged Phone Test.**

> [!CAUTION]
> **Notice on Earlier Results & Retractions:**  
> 1. All November 19–21 toy-setup reports and slides (`MobileFineTuner_Multimodal_Evaluation_Report.md`, `MobileFineTuner_Meeting_Slides.pdf`) are **RETRACTED**.  
> 2. The previous Colab 10-seed MLP run was **NOT yet positive**: `zeroed` beat `real` in 9/10 seeds, training only ran **63 optimizer updates (0.50 epochs)**, and `learned_prefix`/`text_only` did not finish due to Colab GPU quota.

---

## 1. Exact Optimizer Updates & Epochs in Prior Colab Runs

Inspection of the previous [colab/projector_reference.py](file:///home/assassinaj602/Projects/New%20folder%20(4)/github_pr/mobile_fine_tuner/qwen_multimodal_projector_worktree/colab/projector_reference.py) and raw logs ([reports/raw_logs/colab_raw_transcript_extract.txt](file:///home/assassinaj602/Projects/New%20folder%20(4)/github_pr/mobile_fine_tuner/qwen_multimodal_projector_worktree/reports/raw_logs/colab_raw_transcript_extract.txt)):

| Colab Run | Commit | `steps` Flag | `accum_steps` | Actual Images / Micro-Steps | Actual Optimizer Updates | Epochs Completed | Runtime / Seed |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Overfit Sanity Check** | `0f3b991` | `300` | `1` | `300` (`B=1`, 20 images) | **300** | **15.0 epochs** | ~24 s |
| **Linear Projector Sweep** | `d3181ba` | `1000` | `1` | `1,000` (`B=1`, 2,000 train pool) | **1,000** | **0.50 epochs** | ~140 s |
| **2-Layer MLP 10-Seed Sweep** | `0f3b991` | `1000` | `16` | `1,000` (`B=1`, 2,000 train pool) | **63** (`floor(1000/16) + 1`) | **0.50 epochs** | ~129 s |

### Why the MLP Run Took Only ~129 s Despite `accum_steps=16`
In the previous code, `for step in range(1, num_steps + 1):` looped over `1..1000` single-image (`B=1`) forwards/backwards, and `optimizer.step()` only ran every 16 iterations. Thus it performed **only 62 full + 1 partial = 63 optimizer updates** over 0.50 epochs (and `warmup_steps=50` ended after just 3 optimizer updates).

---

## 2. Corrected Raw Numbers from the 0.5-Epoch Colab MLP Run (Seeds Fixed)

The prior summary table mislabeled seeds as `42, 101, 202...` and omitted `zeroed` in the unified report. Below are the exact raw test losses (all 5 captions per image across 200 COCO test images = 1,000 evaluations per seed) with corrected seed IDs (`LR = 2.0e-03`):

| Seed (Corrected) | `real` Test Loss | `shuffled` Test Loss | `zeroed` Test Loss | `learned_prefix` (Aborted Early) | `text_only` |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **42** | 3.5220 | 3.6206 | **3.4560** | 3.4736 | *did not run* |
| **1337** | 3.5906 | 3.6251 | **3.4635** | 3.4571 | *did not run* |
| **2026** | 3.5718 | 3.5740 | **3.4663** | ~3.458 *(truncated)* | *did not run* |
| **7** | 3.5462 | 3.5664 | **3.4869** | *did not finish* | *did not run* |
| **99** | 3.5159 | 3.6283 | **3.4540** | *did not finish* | *did not run* |
| **101** | 3.5719 | 3.5651 | **3.4701** | *did not finish* | *did not run* |
| **314** | 3.5966 | 3.5708 | **3.4565** | *did not finish* | *did not run* |
| **777** | **3.5707** | 3.5772 | 3.7739 | *did not finish* | *did not run* |
| **888** | 3.5115 | 3.5582 | **3.4645** | *did not finish* | *did not run* |
| **999** | 3.5837 | 3.6068 | **3.4499** | *did not finish* | *did not run* |
| **Mean** | **3.5581** | **3.5892** | **3.4942** | *(incomplete — GPU limit)* | *(did not run)* |
| **Median** | **3.5713** | **3.5756** | **3.4640** | *(incomplete)* | *(did not run)* |

**Honest Interpretation:**
- `real` beat `shuffled` in 8/10 seeds (+0.0312 mean, $p = 0.0535$ paired t-test, $p = 0.0645$ Wilcoxon, $p = 0.1094$ sign test — **not statistically significant**).
- **`zeroed` (no-image control) beat `real` in 9 of 10 seeds** ($3.4942$ vs $3.5581$ mean).
- **20-image Overfit Check (`--overfit`, 300 steps = 15 epochs):** `real` dropped $4.3353 \to 0.9110$ vs `shuffled` $2.3587$ vs `zeroed` $1.8156$, showing the gradient path works when trained for enough epochs, and motivating the batched 5+ epoch rewrite.

---

## 3. Rewritten Batched Colab / Kaggle Script (Ready to Run)

Rewrote [colab/projector_reference.py](file:///home/assassinaj602/Projects/New%20folder%20(4)/github_pr/mobile_fine_tuner/qwen_multimodal_projector_worktree/colab/projector_reference.py) with:
- **True Tensor Batching (`--batch_size 16` in parallel):** Right-padded sequences with explicit `attention_mask` + `position_ids` (verified locally on CPU against single-sample `B=1` loss to max abs diff **`2.86e-06`**).
- **Mixed-Precision FP16 Autocast + `GradScaler`:** Frozen Qwen2.5-0.5B in `fp16` on CUDA under `torch.amp.autocast('cuda', dtype=torch.float16)`, trainable projector parameters in `fp32`.
- **True Multi-Epoch Training (`--epochs 5` default):** `125` updates/epoch $\times 5 = \mathbf{625\text{ optimizer updates}}$ over $10,000$ image presentations (**10x** the previous 63 updates and 0.5 epochs).
- **Incremental Checkpointing (`--resume`) + Kaggle/Drive Support.**

---

## 4. C++ Parity & On-Device Memory Summary (See Detailed Audit)

Full details in [MobileFineTuner_Memory_Breakdown_and_Parity_Audit.md](file:///home/assassinaj602/.gemini/antigravity/brain/b2257d23-c4b6-4bf8-9c46-5ffb01155d8d/MobileFineTuner_Memory_Breakdown_and_Parity_Audit.md):
- **Parity Fixed:** Trainable `scale_` parameter + Step-1 gradient/weight dump verified (`grad_cos = 1.000000`, 5-step loss error reduced from **`3.54%` to `0.006% – 0.037%`**).
- **Phone 3.0 GB RSS Mystery Fixed:** Eliminated `1,431 MB` retained FP32 linear weights, `MemoryPool` leaks, and full logits bloat; host `VmHWM` peak dropped from **`3,772.8 MB` to `1,486.9 MB`** (`925.1 MB` steady RSS).
- **Next On-Device Step:** Reconnect the phone briefly to push updated `train_projector` + [scripts/android/run_unplugged_soak.sh](file:///home/assassinaj602/Projects/New%20folder%20(4)/github_pr/mobile_fine_tuner/qwen_multimodal_projector_worktree/scripts/android/run_unplugged_soak.sh), unplug USB, run 15 minutes on battery, and pull raw telemetry + `/proc/cpuinfo` CPU part IDs.
