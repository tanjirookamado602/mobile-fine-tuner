#!/usr/bin/env python3
"""
colab/projector_reference.py
================================================================================
MobileFineTuner: Rigorous Multimodal Projector Training & Parity Reference
Designed for Google Colab (Free T4 GPU / 16GB VRAM) and Local Execution.

Key Scientifically Rigorous Upgrades:
1. Scale Fix: LayerNorm(768) + Linear(768 -> 896) + learnable output scale
   calibrated so initial projector output mean norm matches Qwen's embedding norm (~0.4633).
2. True Control Baseline: 64 freely learnable soft prompt tokens (mode: learned_prefix),
   calibrated to norm ~0.4633, to isolate image grounding from soft prompt tuning.
3. Extended Learning Rate Grid (up to 1e-2), swept on validation set using 2 seeds.
4. 1,000 training steps (0.5 epochs over 2,000 images, batch size = 1).
5. 10 seeds per mode with non-parametric statistics:
   - Median and IQR (Interquartile Range)
   - Per-seed paired differences: (real - shuffled) and (real - learned_prefix)
   - Paired Wilcoxon signed-rank test and 10,000-resample paired bootstrap 95% CI.
6. Comprehensive test evaluation on ALL 5 reference captions per test image (1,000 captions).
7. Greedy generation and BLEU-1/2/4 evaluation on 10 sample images per mode.
8. Single-batch C++ verification export to explain and reproduce the step-0 loss parity.
================================================================================
"""

import os
import sys
import gc
import json
import time
import math
import hashlib
import random
import zipfile
import argparse
import urllib.request
from collections import Counter
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    CLIPVisionModel,
    CLIPImageProcessor
)

try:
    from scipy.stats import wilcoxon
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ==============================================================================
# 1. Colab & Environment Setup
# ==============================================================================
def setup_environment(use_drive=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Runtime Device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM Total: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

    output_base = "./colab_artifacts"
    if use_drive:
        drive_mount_point = "/content/drive"
        if os.path.exists(os.path.join(drive_mount_point, "MyDrive")):
            output_base = os.path.join(drive_mount_point, "MyDrive", "MobileFineTuner_Colab")
            print(f"Using existing Google Drive mount: {output_base}")
        else:
            try:
                from google.colab import drive
                print("Attempting Google Drive mount...")
                drive.mount(drive_mount_point)
                output_base = os.path.join(drive_mount_point, "MyDrive", "MobileFineTuner_Colab")
                print(f"Google Drive mounted: {output_base}")
            except Exception as e:
                print(f"Note: Google Drive mount skipped ({e}). Saving artifacts locally to {output_base}")
                output_base = "./colab_artifacts"

    os.makedirs(output_base, exist_ok=True)
    return device, output_base

# ==============================================================================
# 2. Official COCO 2017 Dataset Acquisition
# ==============================================================================
def download_file(url, dest_path):
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=30) as response, open(dest_path, 'wb') as out_file:
            out_file.write(response.read())
        return True
    except Exception as e:
        return False

def acquire_coco_subset(data_dir, num_train=2000, num_val=200, num_test=200):
    os.makedirs(data_dir, exist_ok=True)
    total_needed = num_train + num_val + num_test

    ann_zip_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    ann_zip_path = os.path.join(data_dir, "annotations_trainval2017.zip")
    val_ann_file = os.path.join(data_dir, "annotations", "captions_val2017.json")
    train_ann_file = os.path.join(data_dir, "annotations", "captions_train2017.json")

    if not os.path.exists(val_ann_file):
        if not os.path.exists(ann_zip_path):
            print(f"Downloading official COCO 2017 annotations from {ann_zip_url} (241 MB)...")
            success = download_file(ann_zip_url, ann_zip_path)
            if not success:
                raise RuntimeError("Failed to download COCO 2017 annotations.")
        print("Extracting annotation files...")
        with zipfile.ZipFile(ann_zip_path, 'r') as zip_ref:
            for member in zip_ref.namelist():
                if "captions_" in member:
                    zip_ref.extract(member, data_dir)

    print("Loading captions_val2017.json...")
    with open(val_ann_file, "r") as f:
        val_data = json.load(f)

    images_pool = val_data["images"]
    annotations_pool = val_data["annotations"]

    if total_needed > len(images_pool) and os.path.exists(train_ann_file):
        print("Loading captions_train2017.json to supplement image pool...")
        with open(train_ann_file, "r") as f:
            tr_data = json.load(f)
            images_pool = tr_data["images"] + images_pool
            annotations_pool = tr_data["annotations"] + annotations_pool

    img_id_to_captions = {}
    for ann in annotations_pool:
        iid = ann["image_id"]
        if iid not in img_id_to_captions:
            img_id_to_captions[iid] = []
        img_id_to_captions[iid].append({
            "caption_id": ann["id"],
            "caption": ann["caption"].strip()
        })

    valid_images = [img for img in images_pool if img["id"] in img_id_to_captions and "coco_url" in img]
    print(f"Data Source: COCO 2017 val2017 pool ({len(valid_images)} annotated images available).")

    if len(valid_images) < total_needed:
        raise ValueError(f"Requested {total_needed} images, but pool only contains {len(valid_images)}.")

    rng = random.Random(42)
    rng.shuffle(valid_images)
    selected_images = valid_images[:total_needed]

    train_imgs = selected_images[:num_train]
    val_imgs = selected_images[num_train : num_train + num_val]
    test_imgs = selected_images[num_train + num_val : num_train + num_val + num_test]

    train_ids = set(img["id"] for img in train_imgs)
    val_ids = set(img["id"] for img in val_imgs)
    test_ids = set(img["id"] for img in test_imgs)

    assert len(train_ids & val_ids) == 0, "FATAL: Overlap between Train and Val sets!"
    assert len(train_ids & test_ids) == 0, "FATAL: Overlap between Train and Test sets!"
    assert len(val_ids & test_ids) == 0, "FATAL: Overlap between Val and Test sets!"
    print(f"Strict Zero-Overlap Assertions PASSED: Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}.")

    img_dir = os.path.join(data_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    print(f"Verifying {len(selected_images)} selected images...")

    failed_downloads = 0
    verified_records = []

    def fetch_image(item, split_name):
        url = item["coco_url"]
        file_name = item["file_name"]
        dest = os.path.join(img_dir, file_name)
        if not os.path.exists(dest):
            ok = download_file(url, dest)
            if not ok:
                return None
        try:
            with Image.open(dest) as img:
                img.verify()
            with open(dest, "rb") as f:
                sha256 = hashlib.sha256(f.read()).hexdigest()
            return {
                "image_id": item["id"],
                "file_name": file_name,
                "local_path": dest,
                "url": url,
                "sha256": sha256,
                "split": split_name,
                "captions": img_id_to_captions[item["id"]]
            }
        except Exception:
            if os.path.exists(dest):
                os.remove(dest)
            return None

    tasks = []
    with ThreadPoolExecutor(max_workers=16) as executor:
        for it in train_imgs: tasks.append(executor.submit(fetch_image, it, "train"))
        for it in val_imgs: tasks.append(executor.submit(fetch_image, it, "val"))
        for it in test_imgs: tasks.append(executor.submit(fetch_image, it, "test"))

        for future in as_completed(tasks):
            res = future.result()
            if res:
                verified_records.append(res)
            else:
                failed_downloads += 1

    print(f"Image verification complete: {len(verified_records)} images verified, {failed_downloads} failed.")

    manifest_path = os.path.join(data_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump({
            "source_dataset": "COCO 2017 Captions (val2017 pool)",
            "license": "Creative Commons Attribution 4.0 License",
            "num_train": num_train,
            "num_val": num_val,
            "num_test": num_test,
            "train_image_ids": list(train_ids),
            "val_image_ids": list(val_ids),
            "test_image_ids": list(test_ids),
            "records": verified_records
        }, f, indent=2)

    return verified_records

# ==============================================================================
# 3. CLIP ViT-B/16 Feature Extraction
# ==============================================================================
def extract_clip_features(records, data_dir, device):
    cache_path = os.path.join(data_dir, "clip_vit_b16_features_64tok.pt")
    if os.path.exists(cache_path):
        print(f"Loading cached CLIP features from {cache_path}...")
        return torch.load(cache_path, map_location="cpu", weights_only=True)

    print("Loading CLIP ViT-B/16 vision model (openai/clip-vit-base-patch16)...")
    clip_model_name = "openai/clip-vit-base-patch16"
    processor = CLIPImageProcessor.from_pretrained(clip_model_name)
    vision_model = CLIPVisionModel.from_pretrained(clip_model_name).to(device).eval()

    pool_2d = nn.AdaptiveAvgPool2d((8, 8))
    features_dict = {}

    batch_size = 32
    print(f"Extracting pooled 64-token features for {len(records)} images on {device}...")
    t0 = time.time()

    for i in range(0, len(records), batch_size):
        batch_recs = records[i : i + batch_size]
        pil_images = []
        valid_recs = []
        for r in batch_recs:
            try:
                img = Image.open(r["local_path"]).convert("RGB")
                pil_images.append(img)
                valid_recs.append(r)
            except Exception:
                pass

        if not pil_images:
            continue

        inputs = processor(images=pil_images, return_tensors="pt").to(device)
        with torch.no_grad():
            out = vision_model(**inputs)
            patch_tokens = out.last_hidden_state[:, 1:, :] # [B, 196, 768]
            B, P, D = patch_tokens.shape
            grid = patch_tokens.permute(0, 2, 1).view(B, D, 14, 14)
            pooled_grid = pool_2d(grid) # [B, D, 8, 8]
            tokens_64 = pooled_grid.view(B, D, 64).permute(0, 2, 1) # [B, 64, D]

        for idx, r in enumerate(valid_recs):
            features_dict[r["image_id"]] = tokens_64[idx].detach().cpu().half()

    elapsed = time.time() - t0
    print(f"Extraction complete in {elapsed:.1f} s ({len(records) / elapsed:.1f} imgs/s).")
    torch.save(features_dict, cache_path)
    return features_dict

# ==============================================================================
# 4. Projector Architectures (Scale-Fixed Projector & Soft-Prompt Baseline)
# ==============================================================================
class ScaledProjector(nn.Module):
    """
    Projector with LayerNorm on visual tokens + Xavier Linear + learnable output scale
    calibrated so initial projector output mean norm is ~0.4633 (matching Qwen embedding norm).
    """
    def __init__(self, in_features=768, out_features=896, target_norm=0.4633, use_layer_norm=True):
        super().__init__()
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.ln = nn.LayerNorm(in_features)
        self.linear = nn.Linear(in_features, out_features)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

        # Scale factor calibrated so initial output norm is target_norm
        init_scale = target_norm / math.sqrt(out_features)
        self.scale = nn.Parameter(torch.tensor(init_scale, dtype=torch.float32))

    def forward(self, x):
        if self.use_layer_norm:
            x = self.ln(x)
        return self.linear(x) * self.scale

class LearnedPrefix(nn.Module):
    """
    Soft-prompt baseline: 64 freely learnable tokens (no image), initialized to target norm.
    Isolates whether image features help beyond generic prefix prompt tuning.
    """
    def __init__(self, num_tokens=64, dim=896, target_norm=0.4633):
        super().__init__()
        init_std = target_norm / math.sqrt(dim)
        self.prefix = nn.Parameter(torch.randn(1, num_tokens, dim) * init_std)

    def forward(self, batch_size=1):
        return self.prefix.expand(batch_size, -1, -1)

# ==============================================================================
# 5. BLEU & Generation Metrics
# ==============================================================================
def compute_sentence_bleu(reference_tokens_list, candidate_tokens, max_n=4):
    cand_len = len(candidate_tokens)
    if cand_len == 0:
        return 0.0
    ref_lens = [len(r) for r in reference_tokens_list]
    closest_ref_len = min(ref_lens, key=lambda r: (abs(r - cand_len), r))
    bp = 1.0 if cand_len > closest_ref_len else math.exp(1.0 - closest_ref_len / cand_len)

    precisions = []
    for n in range(1, max_n + 1):
        cand_ngrams = Counter([tuple(candidate_tokens[i:i+n]) for i in range(cand_len - n + 1)])
        if sum(cand_ngrams.values()) == 0:
            precisions.append(0.0)
            continue
        max_ref_ngrams = Counter()
        for ref in reference_tokens_list:
            ref_ngrams = Counter([tuple(ref[i:i+n]) for i in range(len(ref) - n + 1)])
            for ng in ref_ngrams:
                max_ref_ngrams[ng] = max(max_ref_ngrams[ng], ref_ngrams[ng])
        clipped_count = sum(min(count, max_ref_ngrams[ng]) for ng, count in cand_ngrams.items())
        total_count = sum(cand_ngrams.values())
        precisions.append((clipped_count + 1e-7) / (total_count + 1e-7))

    p_log_sum = sum(math.log(p) for p in precisions) / max_n
    return bp * math.exp(p_log_sum)

# ==============================================================================
# 6. Norm Audit
# ==============================================================================
def conduct_norm_audit(features_dict, qwen_model):
    device = qwen_model.device
    clip_sample = next(iter(features_dict.values())).float().to(device)
    clip_norms = torch.norm(clip_sample, p=2, dim=-1)
    mean_clip_norm = clip_norms.mean().item()

    proj = ScaledProjector(768, 896).to(device)
    with torch.no_grad():
        proj_out = proj(clip_sample.unsqueeze(0)).squeeze(0)
        proj_norms = torch.norm(proj_out, p=2, dim=-1)
        mean_proj_norm = proj_norms.mean().item()

    w_embed = qwen_model.model.embed_tokens.weight
    embed_norms = torch.norm(w_embed, p=2, dim=-1)
    mean_embed_norm = embed_norms.mean().item()

    ratio = mean_proj_norm / mean_embed_norm

    print("\n" + "=" * 90)
    print("  MATHEMATICAL NORM & NUMERICAL SCALE AUDIT (WITH SCALE FIX)")
    print("=" * 90)
    print(f"1. CLIP ViT-B/16 Visual Tokens Mean L2 Norm:     {mean_clip_norm:.4f}")
    print(f"2. Scaled Projector Initial Output Mean Norm:    {mean_proj_norm:.4f}")
    print(f"3. Qwen2.5-0.5B Token Embedding Mean L2 Norm:    {mean_embed_norm:.4f}")
    print(f"4. Scale Ratio (Scaled Projector / Qwen Embed):  {ratio:.4f}x (Calibrated to 1.0x)")
    print("=" * 90)

    return {
        "clip_mean_norm": mean_clip_norm,
        "projector_output_mean_norm": mean_proj_norm,
        "qwen_embed_mean_norm": mean_embed_norm,
        "scale_ratio": ratio
    }

# ==============================================================================
# 7. Training & Evaluation Engine
# ==============================================================================
def train_and_evaluate(
    mode, seed, lr, records, features_dict, qwen_model, tokenizer, device,
    num_steps=1000, val_every=250, use_amp=True
):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    projector = None
    learned_prefix = None
    optimizer = None

    if mode in ["real", "shuffled", "zeroed"]:
        projector = ScaledProjector(768, 896).to(device).float()
        optimizer = torch.optim.AdamW(projector.parameters(), lr=lr, weight_decay=0.01)
    elif mode == "learned_prefix":
        learned_prefix = LearnedPrefix(64, 896).to(device).float()
        optimizer = torch.optim.AdamW(learned_prefix.parameters(), lr=lr, weight_decay=0.01)

    train_recs = [r for r in records if r["split"] == "train"]
    val_recs = [r for r in records if r["split"] == "val"]
    test_recs = [r for r in records if r["split"] == "test"]

    def format_pair(r, cap_idx=0):
        caps = r["captions"]
        caption = caps[cap_idx % len(caps)]["caption"]
        prompt = "<|im_start|>user\nDescribe the image in detail.<|im_end|>\n<|im_start|>assistant\n"
        response = f"{caption}<|im_end|>\n"
        return prompt, response

    def compute_loss_for_pair(prompt_text, response_text, v_prefix):
        p_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        r_ids = tokenizer.encode(response_text, add_special_tokens=False)
        tot_ids = torch.tensor([p_ids + r_ids], device=device, dtype=torch.long)
        text_embeds = qwen_model.model.embed_tokens(tot_ids)

        if v_prefix is not None:
            spliced = torch.cat([v_prefix, text_embeds], dim=1)
            prefix_len = v_prefix.shape[1]
        else:
            spliced = text_embeds
            prefix_len = 0

        out = qwen_model(inputs_embeds=spliced)
        logits = out.logits[:, :-1, :]

        pred_len = logits.shape[1]
        labels = torch.full((1, pred_len), -100, device=device, dtype=torch.long)
        start_resp = prefix_len + len(p_ids) - 1
        end_resp = start_resp + len(r_ids)
        labels[:, start_resp:end_resp] = tot_ids[:, len(p_ids):]

        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100)
        return loss

    def get_prefix_for_eval(r, eval_mode):
        if eval_mode == "text_only":
            return None
        elif eval_mode == "learned_prefix":
            return learned_prefix()
        elif eval_mode == "zeroed":
            v_raw = torch.zeros(1, 64, 768, device=device, dtype=torch.float32)
            return projector(v_raw)
        elif eval_mode == "shuffled":
            rand_id = random.choice(list(features_dict.keys()))
            v_raw = features_dict[rand_id].unsqueeze(0).to(device).float()
            return projector(v_raw)
        else: # real
            v_raw = features_dict[r["image_id"]].unsqueeze(0).to(device).float()
            return projector(v_raw)

    def evaluate_split_all_captions(split_records, eval_mode, max_images=None):
        if projector is not None: projector.eval()
        if learned_prefix is not None: learned_prefix.eval()

        total_loss = 0.0
        count = 0
        recs = split_records[:max_images] if max_images else split_records

        with torch.no_grad():
            for r in recs:
                v_pref = get_prefix_for_eval(r, eval_mode)
                for cap_idx in range(len(r["captions"])):
                    p_txt, r_txt = format_pair(r, cap_idx)
                    loss = compute_loss_for_pair(p_txt, r_txt, v_pref)
                    if not torch.isnan(loss):
                        total_loss += loss.item()
                        count += 1
        return total_loss / max(1, count)

    # Initial Step-0 Validation Loss
    step0_val_loss = evaluate_split_all_captions(val_recs, mode, max_images=50)

    # Training Loop
    if projector is not None: projector.train()
    if learned_prefix is not None: learned_prefix.train()

    step_losses = []
    val_losses = [(0, step0_val_loss)]
    t_start = time.time()

    initial_weights = {}
    if projector is not None:
        initial_weights = {
            "W": projector.linear.weight.detach().cpu().numpy().tolist(),
            "b": projector.linear.bias.detach().cpu().numpy().tolist(),
            "scale": projector.scale.detach().cpu().item()
        }

    for step in range(1, num_steps + 1):
        r = train_recs[(step - 1) % len(train_recs)]
        p_text, r_text = format_pair(r, cap_idx=(step - 1))

        if mode == "text_only":
            v_pref = None
        elif mode == "learned_prefix":
            v_pref = learned_prefix()
        elif mode == "zeroed":
            v_raw = torch.zeros(1, 64, 768, device=device, dtype=torch.float32)
            v_pref = projector(v_raw)
        elif mode == "shuffled":
            rand_id = random.choice(list(features_dict.keys()))
            v_raw = features_dict[rand_id].unsqueeze(0).to(device).float()
            v_pref = projector(v_raw)
        else: # real
            v_raw = features_dict[r["image_id"]].unsqueeze(0).to(device).float()
            v_pref = projector(v_raw)

        if optimizer is not None:
            optimizer.zero_grad()

        loss = compute_loss_for_pair(p_text, r_text, v_pref)

        if mode != "text_only" and optimizer is not None:
            loss.backward()
            params = projector.parameters() if projector else learned_prefix.parameters()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

        loss_val = loss.item()
        step_losses.append(loss_val)

        if step % val_every == 0:
            vl = evaluate_split_all_captions(val_recs, mode, max_images=50)
            val_losses.append((step, vl))

    elapsed = time.time() - t_start

    # Comprehensive Final Evaluation on ALL 5 Captions across all 200 Test Images
    final_test_loss = evaluate_split_all_captions(test_recs, mode, max_images=None)

    # Qualitative Greedy Generations on 10 Sample Images
    generated_samples = []
    if projector is not None: projector.eval()
    if learned_prefix is not None: learned_prefix.eval()

    sample_test_recs = test_recs[:10]
    with torch.no_grad():
        for r in sample_test_recs:
            v_pref = get_prefix_for_eval(r, mode)
            p_text = "<|im_start|>user\nDescribe the image in detail.<|im_end|>\n<|im_start|>assistant\n"
            p_ids = tokenizer.encode(p_text, add_special_tokens=False)
            tot_ids = torch.tensor([p_ids], device=device, dtype=torch.long)
            text_embeds = qwen_model.model.embed_tokens(tot_ids)

            curr_embeds = torch.cat([v_pref, text_embeds], dim=1) if v_pref is not None else text_embeds
            gen_toks = []
            for _ in range(24):
                out = qwen_model(inputs_embeds=curr_embeds)
                next_tok = torch.argmax(out.logits[:, -1, :], dim=-1)
                tok_id = next_tok.item()
                if tok_id in [tokenizer.eos_token_id, 151645]: # <|im_end|>
                    break
                gen_toks.append(tok_id)
                next_embed = qwen_model.model.embed_tokens(next_tok.unsqueeze(0))
                curr_embeds = torch.cat([curr_embeds, next_embed], dim=1)

            gen_text = tokenizer.decode(gen_toks, skip_special_tokens=True).strip()
            ref_texts = [c["caption"] for c in r["captions"]]

            ref_toks_list = [c.lower().split() for c in ref_texts]
            cand_toks = gen_text.lower().split()
            bleu4 = compute_sentence_bleu(ref_toks_list, cand_toks, max_n=4)

            generated_samples.append({
                "image_id": r["image_id"],
                "generated": gen_text,
                "references": ref_texts,
                "bleu4": bleu4
            })

    mean_bleu4 = float(np.mean([s["bleu4"] for s in generated_samples]))

    return {
        "mode": mode,
        "seed": seed,
        "lr": lr,
        "step0_val_loss": step0_val_loss,
        "final_val_loss": val_losses[-1][1],
        "final_test_loss": final_test_loss,
        "mean_bleu4": mean_bleu4,
        "generated_samples": generated_samples,
        "val_losses": val_losses,
        "first_20_losses": step_losses[:20],
        "all_step_losses": step_losses,
        "initial_weights": initial_weights,
        "training_time_s": elapsed,
        "mean_step_ms": (elapsed / max(1, num_steps)) * 1000
    }

# ==============================================================================
# 8. Non-Parametric Bootstrap & Paired Statistics
# ==============================================================================
def paired_bootstrap_test(diffs, n_resamples=10000, alpha=0.05):
    """Computes paired bootstrap mean difference and 95% Confidence Interval."""
    means = []
    n = len(diffs)
    for _ in range(n_resamples):
        sample = np.random.choice(diffs, size=n, replace=True)
        means.append(np.mean(sample))
    mean_diff = float(np.mean(diffs))
    lower_ci = float(np.percentile(means, 100 * (alpha / 2)))
    upper_ci = float(np.percentile(means, 100 * (1.0 - alpha / 2)))
    return mean_diff, lower_ci, upper_ci

# ==============================================================================
# 9. Main Orchestrator
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Rigorous Multimodal Projector Benchmark")
    parser.add_argument("--data_dir", type=str, default="./coco_data", help="Directory for COCO data")
    parser.add_argument("--num_train", type=int, default=2000, help="Train image count")
    parser.add_argument("--num_val", type=int, default=200, help="Val image count")
    parser.add_argument("--num_test", type=int, default=200, help="Test image count")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B", help="Base LLM")
    parser.add_argument("--steps", type=int, default=1000, help="Training steps (default: 1000 = 0.5 epochs)")
    parser.add_argument("--num_seeds", type=int, default=10, help="Seeds per mode (default: 10)")
    parser.add_argument("--sweep", action="store_true", default=True, help="Sweep learning rates")
    parser.add_argument("--no_drive", action="store_true", help="Disable Google Drive saving")
    args = parser.parse_args()

    device, output_dir = setup_environment(use_drive=not args.no_drive)

    # 1. Dataset Acquisition
    records = acquire_coco_subset(
        data_dir=args.data_dir,
        num_train=args.num_train,
        num_val=args.num_val,
        num_test=args.num_test
    )

    # 2. Vision Features
    features_dict = extract_clip_features(records, args.data_dir, device)

    # 3. Base Language Model
    print(f"\nLoading {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    qwen_model = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=torch.float32).to(device)

    # Freeze LLM backbone completely
    for p in qwen_model.parameters():
        p.requires_grad = False
    qwen_model.eval()

    # 4. Norm Audit
    norm_report = conduct_norm_audit(features_dict, qwen_model)

    # 5. Learning Rate Sweep (2 Seeds on Validation Set)
    candidate_lrs = [1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2]
    best_lr = 2e-3
    lr_sweep_results = {}

    if args.sweep:
        print("\n" + "=" * 90)
        print("  LEARNING RATE SWEEP (Validation Set, Averaged Over 2 Seeds: 42 and 1337)")
        print("=" * 90)
        best_val = float("inf")
        for lr in candidate_lrs:
            val_scores = []
            for s in [42, 1337]:
                res = train_and_evaluate(
                    mode="real", seed=s, lr=lr, records=records,
                    features_dict=features_dict, qwen_model=qwen_model, tokenizer=tokenizer,
                    device=device, num_steps=min(200, args.steps), val_every=100
                )
                val_scores.append(res["final_val_loss"])
            mean_vl = float(np.mean(val_scores))
            lr_sweep_results[str(lr)] = mean_vl
            print(f"  LR = {lr:.1e} -> Mean Val Loss = {mean_vl:.4f} (Seed 42: {val_scores[0]:.4f}, Seed 1337: {val_scores[1]:.4f})")
            if mean_vl < best_val:
                best_val = mean_vl
                best_lr = lr
        print(f"\nSelected Optimal Learning Rate: {best_lr:.1e} (Mean Val Loss: {best_val:.4f})\n")

    # 6. Comprehensive 10-Seed Multimodal Benchmark Across 5 Modes
    all_seeds = [42, 1337, 2026, 7, 99, 101, 314, 777, 888, 999][:args.num_seeds]
    modes = ["real", "shuffled", "zeroed", "learned_prefix", "text_only"]
    benchmark_results = {m: [] for m in modes}

    print("\n" + "=" * 90)
    epochs = args.steps / args.num_train
    print(f"  10-SEED RIGOROUS MULTIMODAL BENCHMARK (LR = {best_lr:.1e}, Steps = {args.steps}, Epochs = {epochs:.2f})")
    print(f"  Evaluation on ALL 5 reference captions per test image (1,000 captions total)")
    print("=" * 90)

    for mode in modes:
        for seed in all_seeds:
            print(f"Running Mode: [{mode:<14}] | Seed: {seed:<4} ...", end=" ", flush=True)
            res = train_and_evaluate(
                mode=mode, seed=seed, lr=best_lr, records=records,
                features_dict=features_dict, qwen_model=qwen_model, tokenizer=tokenizer,
                device=device, num_steps=args.steps, val_every=250
            )
            benchmark_results[mode].append(res)
            print(f"Done in {res['training_time_s']:.1f} s | Test Loss: {res['final_test_loss']:.4f} | BLEU-4: {res['mean_bleu4']:.4f}")

    # 7. Non-Parametric Statistical Summary Table
    print("\n" + "=" * 105)
    print("  FINAL 10-SEED HELD-OUT TEST EVALUATION SUMMARY (ALL 5 CAPTIONS PER IMAGE)")
    print("=" * 105)
    print(f"{'Mode':<16} | {'Median Loss':<12} | {'IQR':<10} | {'Mean ± Std':<18} | {'Min Loss':<10} | {'Max Loss':<10} | {'Mean BLEU-4':<12}")
    print("-" * 105)

    mode_test_losses = {}
    for mode in modes:
        losses = [r["final_test_loss"] for r in benchmark_results[mode]]
        bleus = [r["mean_bleu4"] for r in benchmark_results[mode]]
        mode_test_losses[mode] = losses

        med = np.median(losses)
        iqr = np.percentile(losses, 75) - np.percentile(losses, 25)
        mean_l = np.mean(losses)
        std_l = np.std(losses)
        min_l = np.min(losses)
        max_l = np.max(losses)
        mean_b = np.mean(bleus)

        print(f"{mode:<16} | {med:<12.4f} | {iqr:<10.4f} | {mean_l:.4f} ± {std_l:.4f}   | {min_l:<10.4f} | {max_l:<10.4f} | {mean_b:<12.4f}")
    print("=" * 105)

    # 8. Paired Hypothesis Testing & Non-Parametric Bootstrap
    print("\n" + "=" * 90)
    print("  PAIRED HYPOTHESIS TESTING (PER-SEED DIFFERENCES)")
    print("=" * 90)

    real_losses = np.array(mode_test_losses["real"])
    shuffled_losses = np.array(mode_test_losses["shuffled"])
    diff_shuffled = shuffled_losses - real_losses
    mean_diff_s, ci_low_s, ci_high_s = paired_bootstrap_test(diff_shuffled)

    print(f"1. Real vs. Shuffled Visual Features:")
    print(f"   - Per-seed paired difference (Shuffled - Real): {np.round(diff_shuffled, 4).tolist()}")
    print(f"   - Mean Advantage of Real: {mean_diff_s:+.4f} loss points")
    print(f"   - 95% Bootstrap Confidence Interval: [{ci_low_s:+.4f}, {ci_high_s:+.4f}]")
    if HAS_SCIPY and len(diff_shuffled) >= 5:
        w_stat, p_val_s = wilcoxon(shuffled_losses, real_losses, alternative='greater')
        print(f"   - Paired Wilcoxon Signed-Rank Test: W = {w_stat:.1f}, p = {p_val_s:.4e}")

    prefix_losses = np.array(mode_test_losses["learned_prefix"])
    diff_prefix = prefix_losses - real_losses
    mean_diff_p, ci_low_p, ci_high_p = paired_bootstrap_test(diff_prefix)

    print(f"\n2. Real Visual Features vs. Freely Learned Prefix (Soft Prompt Control):")
    print(f"   - Per-seed paired difference (LearnedPrefix - Real): {np.round(diff_prefix, 4).tolist()}")
    print(f"   - Mean Advantage of Real: {mean_diff_p:+.4f} loss points")
    print(f"   - 95% Bootstrap Confidence Interval: [{ci_low_p:+.4f}, {ci_high_p:+.4f}]")
    if HAS_SCIPY and len(diff_prefix) >= 5:
        w_stat_p, p_val_p = wilcoxon(prefix_losses, real_losses, alternative='greater')
        print(f"   - Paired Wilcoxon Signed-Rank Test: W = {w_stat_p:.1f}, p = {p_val_p:.4e}")
    print("=" * 90)

    # 9. Qualitative Caption Generations
    print("\n" + "=" * 90)
    print("  QUALITATIVE EVALUATION: 10 SAMPLE GENERATED CAPTIONS ACROSS MODES")
    print("=" * 90)
    sample_recs = benchmark_results["real"][0]["generated_samples"]
    for i, s in enumerate(sample_recs):
        print(f"\n[Image ID: {s['image_id']}]")
        print(f"  Ground Truth: \"{s['references'][0]}\"")
        for m in modes:
            gen_txt = benchmark_results[m][0]["generated_samples"][i]["generated"]
            b4 = benchmark_results[m][0]["generated_samples"][i]["bleu4"]
            print(f"  - [{m:<14}]: \"{gen_txt}\" (BLEU-4: {b4:.4f})")
    print("=" * 90)

    # 10. Single-Batch C++ Parity Check Export
    test_r = records[0]
    p_text = "<|im_start|>user\nDescribe the image in detail.<|im_end|>\n<|im_start|>assistant\n"
    r_text = f"{test_r['captions'][0]['caption']}<|im_end|>\n"
    p_ids = tokenizer.encode(p_text, add_special_tokens=False)
    r_ids = tokenizer.encode(r_text, add_special_tokens=False)
    tot_ids = p_ids + r_ids

    export_payload = {
        "norm_report": norm_report,
        "best_learning_rate": best_lr,
        "lr_sweep_results": lr_sweep_results,
        "summary_table": {
            mode: {
                "median_loss": float(np.median(mode_test_losses[mode])),
                "iqr": float(np.percentile(mode_test_losses[mode], 75) - np.percentile(mode_test_losses[mode], 25)),
                "mean_loss": float(np.mean(mode_test_losses[mode])),
                "std_loss": float(np.std(mode_test_losses[mode])),
                "losses": [float(x) for x in mode_test_losses[mode]],
                "mean_bleu4": float(np.mean([r["mean_bleu4"] for r in benchmark_results[mode]]))
            } for mode in modes
        },
        "paired_comparisons": {
            "real_vs_shuffled": {
                "mean_diff": mean_diff_s,
                "ci_95": [ci_low_s, ci_high_s],
                "diffs": [float(x) for x in diff_shuffled]
            },
            "real_vs_learned_prefix": {
                "mean_diff": mean_diff_p,
                "ci_95": [ci_low_p, ci_high_p],
                "diffs": [float(x) for x in diff_prefix]
            }
        },
        "fixed_batch_for_cpp": {
            "image_id": test_r["image_id"],
            "visual_features_shape": [64, 768],
            "input_ids": tot_ids,
            "prompt_len": len(p_ids),
            "response_len": len(r_ids)
        },
        "initial_projector_weights": benchmark_results["real"][0]["initial_weights"],
        "created_at": datetime.now().isoformat()
    }

    export_path = os.path.join(output_dir, "pytorch_projector_reference_results.json")
    with open(export_path, "w") as f:
        json.dump(export_payload, f, indent=2)
    print(f"\nSaved complete benchmark artifacts to: {export_path}")

if __name__ == "__main__":
    main()
