#!/usr/bin/env python3
"""
colab/projector_reference.py
================================================================================
MobileFineTuner: Batched & Mixed-Precision Multimodal Projector Reference Suite
Designed for Google Colab / Kaggle (T4 / P100 GPU) and Local CPU Verification.

Key Upgrades (Batched Multi-Epoch Engine):
1. True Tensor Batching (--batch_size 16 in parallel, right-padded with explicit
   attention_mask + position_ids) instead of B=1 micro-step loops.
2. Mixed-Precision FP16 Autocast + GradScaler on CUDA:
   Frozen Qwen2.5-0.5B runs in float16 under torch.amp.autocast('cuda', float16);
   trainable Projector / LearnedPrefix stay in float32 master weights.
3. Explicit Epoch & Optimizer-Update Accounting (--epochs 5 default -> 625 real
   optimizer updates over 10,000 image-caption pairs on 2,000 train images).
4. Deterministic Split-Scoped Controls:
   real, shuffled (deterministic per-split derangement), zeroed, learned_prefix,
   and text_only (evaluated once deterministically).
5. Incremental Artifact Checkpointing (--resume support) + Raw Log Export.
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

PROMPT_TEXT = "<|im_start|>user\nDescribe the image in detail.<|im_end|>\n<|im_start|>assistant\n"
DEFAULT_SEEDS = [42, 1337, 2026, 7, 99]


# ==============================================================================
# 1. Runtime & Environment Setup (Colab / Kaggle / Local)
# ==============================================================================
def setup_environment(use_drive=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Runtime Device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM Total: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

    output_base = "./colab_artifacts"
    if os.path.exists("/kaggle/working"):
        output_base = "/kaggle/working/colab_artifacts"
        print(f"Using Kaggle working directory: {output_base}")
    elif use_drive:
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
    except Exception:
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
            if not download_file(ann_zip_url, ann_zip_path):
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
        img_id_to_captions.setdefault(iid, []).append({
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
    val_imgs = selected_images[num_train:num_train + num_val]
    test_imgs = selected_images[num_train + num_val:num_train + num_val + num_test]

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
            if not download_file(url, dest):
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

    with ThreadPoolExecutor(max_workers=16) as executor:
        tasks = (
            [executor.submit(fetch_image, it, "train") for it in train_imgs]
            + [executor.submit(fetch_image, it, "val") for it in val_imgs]
            + [executor.submit(fetch_image, it, "test") for it in test_imgs]
        )
        for future in as_completed(tasks):
            res = future.result()
            if res:
                verified_records.append(res)
            else:
                failed_downloads += 1

    # Preserve deterministic ordering by split and image_id
    verified_records.sort(key=lambda r: ({"train": 0, "val": 1, "test": 2}[r["split"]], r["image_id"]))
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
def extract_clip_features(records, data_dir, device, no_pool=False, clip_layer="penultimate"):
    tag = f"{clip_layer}_{'196tok' if no_pool else '64tok'}"
    cache_path = os.path.join(data_dir, f"clip_vit_b16_features_{tag}.pt")
    if os.path.exists(cache_path):
        print(f"Loading cached CLIP features from {cache_path}...")
        return torch.load(cache_path, map_location="cpu", weights_only=True)

    print(f"Loading CLIP ViT-B/16 (openai/clip-vit-base-patch16), layer={clip_layer}, no_pool={no_pool}...")
    clip_model_name = "openai/clip-vit-base-patch16"
    processor = CLIPImageProcessor.from_pretrained(clip_model_name)
    vision_model = CLIPVisionModel.from_pretrained(clip_model_name).to(device).eval()

    pool_2d = nn.AdaptiveAvgPool2d((8, 8)) if not no_pool else None
    features_dict = {}

    batch_size = 32
    print(f"Extracting features for {len(records)} images on {device}...")
    t0 = time.time()

    for i in range(0, len(records), batch_size):
        batch_recs = records[i:i + batch_size]
        pil_images, valid_recs = [], []
        for r in batch_recs:
            try:
                pil_images.append(Image.open(r["local_path"]).convert("RGB"))
                valid_recs.append(r)
            except Exception:
                pass
        if not pil_images:
            continue

        inputs = processor(images=pil_images, return_tensors="pt").to(device)
        with torch.no_grad():
            out = vision_model(**inputs, output_hidden_states=True)
            hidden = out.hidden_states[-2] if clip_layer == "penultimate" else out.last_hidden_state
            patch_tokens = hidden[:, 1:, :]
            B, _, D = patch_tokens.shape
            if no_pool:
                tokens_out = patch_tokens
            else:
                grid = patch_tokens.permute(0, 2, 1).view(B, D, 14, 14)
                tokens_out = pool_2d(grid).view(B, D, 64).permute(0, 2, 1)

        for idx, r in enumerate(valid_recs):
            features_dict[r["image_id"]] = tokens_out[idx].detach().cpu().half()

    elapsed = time.time() - t0
    print(f"Extraction complete in {elapsed:.1f} s ({len(records) / max(elapsed, 1e-6):.1f} imgs/s). Saved to {cache_path}")
    torch.save(features_dict, cache_path)
    del vision_model
    if device == "cuda":
        torch.cuda.empty_cache()
    return features_dict


# ==============================================================================
# 4. Projector Architectures
# ==============================================================================
class Projector(nn.Module):
    """
    2-layer GELU MLP or Linear projector with LayerNorm + scale + pos_embed.
    """
    def __init__(
        self, in_features=768, out_features=896, arch="mlp", num_tokens=64,
        target_norm=0.4633, use_layer_norm=True, use_pos_embed=True, gelu_approx="none"
    ):
        super().__init__()
        self.use_layer_norm = use_layer_norm
        self.num_tokens = num_tokens

        if use_layer_norm:
            self.ln = nn.LayerNorm(in_features)

        if arch == "mlp":
            self.net = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.GELU(approximate=gelu_approx),
                nn.Linear(out_features, out_features)
            )
            nn.init.xavier_uniform_(self.net[0].weight)
            nn.init.zeros_(self.net[0].bias)
            nn.init.xavier_uniform_(self.net[2].weight)
            nn.init.zeros_(self.net[2].bias)
        else:
            self.net = nn.Linear(in_features, out_features)
            nn.init.xavier_uniform_(self.net.weight)
            nn.init.zeros_(self.net.bias)

        init_scale = target_norm / math.sqrt(out_features)
        self.scale = nn.Parameter(torch.tensor(init_scale, dtype=torch.float32))

        if use_pos_embed:
            init_std = target_norm / math.sqrt(out_features)
            self.pos_embed = nn.Parameter(torch.randn(1, num_tokens, out_features) * init_std)
        else:
            self.pos_embed = None

    def forward(self, x):
        if self.use_layer_norm:
            x = self.ln(x)
        out = self.net(x) * self.scale
        if self.pos_embed is not None:
            out = out + self.pos_embed
        return out


class LearnedPrefix(nn.Module):
    """Freely learnable soft prompt control (no image input)."""
    def __init__(self, num_tokens=64, dim=896, target_norm=0.4633):
        super().__init__()
        init_std = target_norm / math.sqrt(dim)
        self.prefix = nn.Parameter(torch.randn(1, num_tokens, dim) * init_std)

    def forward(self, batch_size=1):
        return self.prefix.expand(batch_size, -1, -1)


# ==============================================================================
# 5. Sentence BLEU Metric & Diagnostics
# ==============================================================================
def compute_sentence_bleu(reference_tokens_list, candidate_tokens, max_n=4):
    cand_len = len(candidate_tokens)
    if cand_len == 0 or any(len(r) == 0 for r in reference_tokens_list):
        return 0.0
    ref_lens = [len(r) for r in reference_tokens_list]
    closest_ref_len = min(ref_lens, key=lambda r: (abs(r - cand_len), r))
    bp = 1.0 if cand_len > closest_ref_len else math.exp(1.0 - closest_ref_len / cand_len)

    precisions = []
    for n in range(1, max_n + 1):
        if cand_len < n:
            return 0.0
        cand_ngrams = Counter([tuple(candidate_tokens[i:i + n]) for i in range(cand_len - n + 1)])
        max_ref_ngrams = Counter()
        for ref in reference_tokens_list:
            ref_ngrams = Counter([tuple(ref[i:i + n]) for i in range(len(ref) - n + 1)])
            for ng in ref_ngrams:
                max_ref_ngrams[ng] = max(max_ref_ngrams[ng], ref_ngrams[ng])
        clipped_count = sum(min(count, max_ref_ngrams[ng]) for ng, count in cand_ngrams.items())
        total_count = sum(cand_ngrams.values())
        if clipped_count == 0:
            return 0.0
        precisions.append(clipped_count / total_count)

    return bp * math.exp(sum(math.log(p) for p in precisions) / max_n)


def build_split_shuffle_map(records, seed=1337):
    """Build deterministic derangement within each split (train/val/test) so no cross-split leakage occurs."""
    shuffle_map = {}
    rng = random.Random(seed)
    for split in ("train", "val", "test"):
        ids = [r["image_id"] for r in records if r["split"] == split]
        if len(ids) <= 1:
            for iid in ids:
                shuffle_map[iid] = iid
            continue
        perm = ids.copy()
        rng.shuffle(perm)
        for i in range(len(ids)):
            if perm[i] == ids[i]:
                swap_j = (i + 1) % len(ids)
                perm[i], perm[swap_j] = perm[swap_j], perm[i]
        for orig, shuf in zip(ids, perm):
            shuffle_map[orig] = shuf
    return shuffle_map


def verify_data_alignment(records, features_dict):
    print("\n" + "=" * 90)
    print("  ALIGNMENT DIAGNOSTIC: 5 VERIFIED SAMPLES (Image ID -> Captions -> CLIP Features)")
    print("=" * 90)
    for i, r in enumerate(records[:5]):
        iid = r["image_id"]
        feat = features_dict[iid].float()
        norm_val = torch.norm(feat, p=2, dim=-1).mean().item()
        head_vals = np.round(feat[0, :5].numpy(), 4).tolist()
        print(f"Sample {i+1}:")
        print(f"  Image ID:    {iid} ({r['file_name']})")
        print(f"  Caption:     \"{r['captions'][0]['caption']}\"")
        print(f"  Feature Dim: {list(feat.shape)} | Mean Token Norm: {norm_val:.4f}")
        print(f"  First 5 Dims: {head_vals}")

    shuf_map = build_split_shuffle_map(records)
    mismatches = sum(1 for k, v in shuf_map.items() if k != v)
    sample_id = records[0]["image_id"]
    sample_cos = F.cosine_similarity(
        features_dict[sample_id][0].float(),
        features_dict[shuf_map[sample_id]][0].float(),
        dim=0
    ).item()
    print("\n[Split-Scoped Shuffle Permutation Check]:")
    print(f"  Keys Permuted: {mismatches} / {len(shuf_map)} keys displaced ({mismatches / max(1, len(shuf_map)) * 100:.1f}%)")
    print(f"  Cosine Similarity Between Sample Real vs. Shuffled Feature: {sample_cos:.4f} (Confirmed permuted)")
    print("=" * 90)


# ==============================================================================
# 6. Parallel Batched Collation & Forward/Loss Helpers
# ==============================================================================
def pretokenize_splits(records, tokenizer):
    """Pre-tokenize all captions once so batching has zero tokenizer overhead."""
    prompt_ids = tokenizer.encode(PROMPT_TEXT, add_special_tokens=False)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    tokenized = {}
    for r in records:
        iid = r["image_id"]
        tokenized[iid] = [
            tokenizer.encode(f"{c['caption']}<|im_end|>\n", add_special_tokens=False)
            for c in r["captions"]
        ]
    return prompt_ids, pad_id, tokenized


def build_batch_tensors(batch_items, prompt_ids, pad_id, device):
    """
    batch_items: list of (image_id, r_ids)
    Returns right-padded tot_ids [B, T_max], text_attn [B, T_max], and labels [B, T_max - 1]
    (labels before prefix are prepended with -100 once prefix_len is known).
    """
    B = len(batch_items)
    p_len = len(prompt_ids)
    seqs = [prompt_ids + r_ids for _, r_ids in batch_items]
    max_t = max(len(s) for s in seqs)

    tot_ids = torch.full((B, max_t), pad_id, dtype=torch.long)
    text_attn = torch.zeros((B, max_t), dtype=torch.long)
    resp_lens = []

    for b, ((_, r_ids), seq) in enumerate(zip(batch_items, seqs)):
        L = len(seq)
        tot_ids[b, :L] = torch.tensor(seq, dtype=torch.long)
        text_attn[b, :L] = 1
        resp_lens.append(len(r_ids))

    return tot_ids.to(device), text_attn.to(device), p_len, resp_lens


def batched_forward_loss(
    mode, batch_items, v_prefix_fn, qwen_model, embed_layer,
    prompt_ids, pad_id, device, features_dict, shuffle_map,
    num_tokens, per_sample=False
):
    tot_ids, text_attn, p_len, resp_lens = build_batch_tensors(batch_items, prompt_ids, pad_id, device)
    B, max_t = tot_ids.shape
    amp_enabled = (device == "cuda")

    with torch.no_grad():
        text_embeds = embed_layer(tot_ids)

    v_pref = v_prefix_fn(batch_items, B)  # [B, prefix_len, H] or None
    if v_pref is not None:
        prefix_len = v_pref.shape[1]
        spliced = torch.cat([v_pref.to(text_embeds.dtype), text_embeds], dim=1)
        pref_attn = torch.ones((B, prefix_len), device=device, dtype=torch.long)
        attn_mask = torch.cat([pref_attn, text_attn], dim=1)
    else:
        prefix_len = 0
        spliced = text_embeds
        attn_mask = text_attn

    # Explicit position_ids matched to valid tokens (invariant under right-padding)
    position_ids = attn_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attn_mask == 0, 0)

    # Shift labels: predict token at pos t+1 from logits at pos t
    pred_len = spliced.shape[1] - 1
    labels = torch.full((B, pred_len), -100, device=device, dtype=torch.long)
    for b, r_len in enumerate(resp_lens):
        start_resp = prefix_len + p_len - 1
        end_resp = start_resp + r_len
        labels[b, start_resp:end_resp] = tot_ids[b, p_len:p_len + r_len]

    with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp_enabled):
        out = qwen_model(inputs_embeds=spliced, attention_mask=attn_mask, position_ids=position_ids)
        logits = out.logits[:, :-1, :].float()

    if per_sample:
        # Compute per-sample mean CE loss identically to unbatched B=1 evaluation
        flat_losses = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            ignore_index=-100,
            reduction="none"
        ).view(B, pred_len)
        valid_mask = (labels != -100).float()
        per_item = (flat_losses * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1.0)
        return per_item

    loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=-100
    )
    return loss


# ==============================================================================
# 7. Overfit Diagnostic (20 Images, 300 Steps)
# ==============================================================================
def run_overfit_test(records, features_dict, qwen_model, tokenizer, device, arch="mlp", num_tokens=64):
    print("\n" + "=" * 90)
    print("  OVERFIT TEST: 20 Images, 300 Steps (Real vs. Shuffled vs. Zeroed)")
    print("=" * 90)

    overfit_recs = [r for r in records if r["split"] == "train"][:20]
    prompt_ids, pad_id, tokenized = pretokenize_splits(overfit_recs, tokenizer)
    shuffle_map = build_split_shuffle_map(overfit_recs, seed=1337)
    embed_layer = qwen_model.model.embed_tokens
    amp_enabled = (device == "cuda")
    history = {}

    for mode in ["real", "shuffled", "zeroed"]:
        torch.manual_seed(42)
        random.seed(42)
        projector = Projector(768, 896, arch=arch, num_tokens=num_tokens, use_pos_embed=True).to(device).float()
        optimizer = torch.optim.AdamW(projector.parameters(), lr=1e-3, weight_decay=0.0)
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

        def v_fn(items, B):
            if mode == "zeroed":
                v_raw = torch.zeros(B, num_tokens, 768, device=device, dtype=torch.float32)
            elif mode == "shuffled":
                v_raw = torch.stack([features_dict[shuffle_map[iid]] for iid, _ in items]).to(device).float()
            else:
                v_raw = torch.stack([features_dict[iid] for iid, _ in items]).to(device).float()
            return projector(v_raw)

        step_losses = []
        t0 = time.time()
        for step in range(1, 301):
            r = overfit_recs[(step - 1) % len(overfit_recs)]
            iid = r["image_id"]
            batch_items = [(iid, tokenized[iid][0])]
            optimizer.zero_grad()
            loss = batched_forward_loss(
                mode, batch_items, v_fn, qwen_model, embed_layer,
                prompt_ids, pad_id, device, features_dict, shuffle_map, num_tokens
            )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            step_losses.append(loss.item())
            if step in [1, 50, 100, 150, 200, 250, 300]:
                print(f"  Mode [{mode:<8}] | Step {step:<3} | Loss: {loss.item():.4f}")

        elapsed = time.time() - t0
        history[mode] = step_losses
        print(f"  Mode [{mode:<8}] Done in {elapsed:.1f} s | Start Loss: {step_losses[0]:.4f} -> Final Loss: {step_losses[-1]:.4f}\n")

    return history


# ==============================================================================
# 8. Batched Multi-Epoch Training & Evaluation Engine
# ==============================================================================
class LoRALinear(nn.Module):
    """Rank-r LoRA wrapper around a frozen nn.Linear layer (train A, B in FP32)."""

    def __init__(self, base_linear: nn.Linear, rank: int = 16, alpha: float = 32.0):
        super().__init__()
        self.base = base_linear
        self.rank = rank
        self.scaling = alpha / float(rank)
        in_f = base_linear.in_features
        out_f = base_linear.out_features
        dev = base_linear.weight.device
        self.lora_A = nn.Parameter(torch.randn(in_f, rank, device=dev, dtype=torch.float32) * (1.0 / math.sqrt(rank)))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_f, device=dev, dtype=torch.float32))

    def forward(self, x):
        base_out = self.base(x)
        delta = (x.float() @ self.lora_A) @ self.lora_B
        return base_out + (delta * self.scaling).to(base_out.dtype)


def attach_lora_to_attention(qwen_model, rank: int = 16, alpha: float = 32.0):
    """Attach LoRA adapters to q_proj, k_proj, v_proj, o_proj across all decoder layers."""
    lora_params = []
    targets = ("q_proj", "k_proj", "v_proj", "o_proj")
    for layer in qwen_model.model.layers:
        attn = layer.self_attn
        for name in targets:
            mod = getattr(attn, name)
            if isinstance(mod, nn.Linear):
                wrapped = LoRALinear(mod, rank=rank, alpha=alpha)
                setattr(attn, name, wrapped)
                lora_params.extend([wrapped.lora_A, wrapped.lora_B])
    return lora_params


def detach_lora_from_attention(qwen_model):
    """Restore original frozen nn.Linear attention projections."""
    targets = ("q_proj", "k_proj", "v_proj", "o_proj")
    for layer in qwen_model.model.layers:
        attn = layer.self_attn
        for name in targets:
            mod = getattr(attn, name)
            if isinstance(mod, LoRALinear):
                setattr(attn, name, mod.base)


def train_and_evaluate(
    mode, seed, lr, records, features_dict, qwen_model, tokenizer, device,
    prompt_ids, pad_id, tokenized, shuffle_map,
    epochs=5, batch_size=16, grad_accum=1, arch="mlp", num_tokens=64,
    eval_batch_size=32, eval_test=True, val_every_epochs=1,
    print_epoch_curve=False, lora_stage=False, lora_rank=16, lora_lr=1e-5, lora_epochs=5
):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    projector = None
    learned_prefix = None
    optimizer = None

    if mode in ["real", "shuffled", "zeroed"]:
        projector = Projector(768, 896, arch=arch, num_tokens=num_tokens, use_pos_embed=True).to(device).float()
        optimizer = torch.optim.AdamW(projector.parameters(), lr=lr, weight_decay=0.01)
    elif mode == "learned_prefix":
        learned_prefix = LearnedPrefix(num_tokens, 896).to(device).float()
        optimizer = torch.optim.AdamW(learned_prefix.parameters(), lr=lr, weight_decay=0.01)

    train_recs = [r for r in records if r["split"] == "train"]
    val_recs = [r for r in records if r["split"] == "val"]
    test_recs = [r for r in records if r["split"] == "test"]
    embed_layer = qwen_model.model.embed_tokens
    amp_enabled = (device == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    micro_batches_per_epoch = math.ceil(len(train_recs) / batch_size)
    updates_per_epoch = math.ceil(micro_batches_per_epoch / grad_accum)
    total_updates = epochs * updates_per_epoch if mode != "text_only" else 0
    warmup_updates = max(10, int(0.05 * total_updates)) if total_updates > 0 else 1

    def get_scheduled_lr(update_idx):
        if update_idx < warmup_updates:
            return lr * (max(1, update_idx) / max(1, warmup_updates))
        progress = (update_idx - warmup_updates) / max(1, total_updates - warmup_updates)
        return lr * 0.5 * (1.0 + math.cos(math.pi * progress))

    def make_prefix_fn():
        def v_fn(items, B):
            if mode == "text_only":
                return None
            elif mode == "learned_prefix":
                return learned_prefix(batch_size=B)
            elif mode == "zeroed":
                v_raw = torch.zeros(B, num_tokens, 768, device=device, dtype=torch.float32)
                return projector(v_raw)
            elif mode == "shuffled":
                v_raw = torch.stack([features_dict[shuffle_map[iid]] for iid, _ in items]).to(device).float()
                return projector(v_raw)
            else:  # real
                v_raw = torch.stack([features_dict[iid] for iid, _ in items]).to(device).float()
                return projector(v_raw)
        return v_fn

    v_prefix_fn = make_prefix_fn()

    def evaluate_split_all_captions(split_records, return_per_image=False):
        if projector is not None:
            projector.eval()
        if learned_prefix is not None:
            learned_prefix.eval()

        all_pairs = [
            (r["image_id"], cap_toks)
            for r in split_records
            for cap_toks in tokenized[r["image_id"]]
        ]
        all_losses = []
        per_image_buckets = {}
        with torch.no_grad():
            for i in range(0, len(all_pairs), eval_batch_size):
                chunk = all_pairs[i:i + eval_batch_size]
                per_item = batched_forward_loss(
                    mode, chunk, v_prefix_fn, qwen_model, embed_layer,
                    prompt_ids, pad_id, device, features_dict, shuffle_map,
                    num_tokens, per_sample=True
                )
                vals = per_item.detach().cpu().tolist()
                for (iid, _), v in zip(chunk, vals):
                    if not math.isnan(v):
                        all_losses.append(v)
                        per_image_buckets.setdefault(iid, []).append(v)
        mean_loss = float(np.mean(all_losses)) if all_losses else float("nan")
        if return_per_image:
            per_image = [
                {"image_id": r["image_id"], "loss": float(np.mean(per_image_buckets[r["image_id"]]))}
                for r in split_records if r["image_id"] in per_image_buckets
            ]
            return mean_loss, per_image
        return mean_loss

    step0_val_loss = evaluate_split_all_captions(val_recs)
    val_losses = [(0, step0_val_loss)]
    if print_epoch_curve:
        print(f"\n  [Epoch 0/{epochs}] Val Loss: {step0_val_loss:.4f}", flush=True)

    if projector is not None:
        projector.train()
    if learned_prefix is not None:
        learned_prefix.train()

    update_losses = []
    t_start = time.time()
    rng_order = random.Random(seed)
    optimizer_updates = 0
    images_seen = 0

    if mode != "text_only" and optimizer is not None:
        for epoch in range(1, epochs + 1):
            t_ep = time.time()
            ep_losses = []
            epoch_recs = train_recs.copy()
            rng_order.shuffle(epoch_recs)

            optimizer.zero_grad()
            accum_count = 0
            accum_loss_sum = 0.0

            for mb_idx, i in enumerate(range(0, len(epoch_recs), batch_size)):
                chunk_recs = epoch_recs[i:i + batch_size]
                # Cycle across all 5 captions across epochs so training sees diverse phrasing
                batch_items = []
                for idx_in_b, r in enumerate(chunk_recs):
                    iid = r["image_id"]
                    cap_list = tokenized[iid]
                    cap_toks = cap_list[(epoch - 1 + idx_in_b) % len(cap_list)]
                    batch_items.append((iid, cap_toks))

                images_seen += len(batch_items)
                loss = batched_forward_loss(
                    mode, batch_items, v_prefix_fn, qwen_model, embed_layer,
                    prompt_ids, pad_id, device, features_dict, shuffle_map,
                    num_tokens, per_sample=False
                )
                scaled_loss = loss / grad_accum
                scaler.scale(scaled_loss).backward()
                accum_loss_sum += loss.item()
                accum_count += 1

                is_last_mb = (i + batch_size >= len(epoch_recs))
                if accum_count == grad_accum or is_last_mb:
                    optimizer_updates += 1
                    cur_lr = get_scheduled_lr(optimizer_updates)
                    for g in optimizer.param_groups:
                        g["lr"] = cur_lr

                    params = projector.parameters() if projector is not None else learned_prefix.parameters()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                    step_l = accum_loss_sum / accum_count
                    update_losses.append(step_l)
                    ep_losses.append(step_l)
                    accum_count = 0
                    accum_loss_sum = 0.0

            if epoch % val_every_epochs == 0 or epoch == epochs:
                vl = evaluate_split_all_captions(val_recs)
                val_losses.append((epoch, vl))
                if print_epoch_curve:
                    ep_dt = time.time() - t_ep
                    tr_mean = float(np.mean(ep_losses)) if ep_losses else float("nan")
                    print(f"  [Epoch {epoch:2d}/{epochs}] Time: {ep_dt:5.1f} s | Train Loss: {tr_mean:.4f} | Val Loss: {vl:.4f}", flush=True)
                if projector is not None:
                    projector.train()
                if learned_prefix is not None:
                    learned_prefix.train()

    elapsed = time.time() - t_start
    best_ep, best_vl = min(val_losses, key=lambda x: x[1])
    if print_epoch_curve and mode != "text_only":
        print(f"  [Plateau Check] Best Val Loss: {best_vl:.4f} at Epoch {best_ep}/{epochs} | Avg Epoch Time: {elapsed / max(1, epochs):.1f} s", flush=True)

    lora_info = None
    if lora_stage and mode != "text_only" and projector is not None:
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        lora_params = attach_lora_to_attention(qwen_model, rank=lora_rank, alpha=float(lora_rank * 2))
        proj_params = list(projector.parameters())
        n_proj = sum(p.numel() for p in proj_params)
        n_lora = sum(p.numel() for p in lora_params)
        n_total = sum(p.numel() for p in qwen_model.parameters()) + n_proj
        lora_opt = torch.optim.AdamW(proj_params + lora_params, lr=lora_lr, weight_decay=0.01)
        lora_scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        t_lora = time.time()
        lora_val_curve = []
        for le in range(1, lora_epochs + 1):
            t_lep = time.time()
            projector.train()
            qwen_model.train()
            epoch_recs = train_recs.copy()
            rng_order.shuffle(epoch_recs)
            lora_opt.zero_grad()
            ac = 0
            for i in range(0, len(epoch_recs), batch_size):
                chunk_recs = epoch_recs[i:i + batch_size]
                batch_items = [
                    (r["image_id"], tokenized[r["image_id"]][(le - 1 + idx_b) % len(tokenized[r["image_id"]])])
                    for idx_b, r in enumerate(chunk_recs)
                ]
                loss = batched_forward_loss(
                    mode, batch_items, v_prefix_fn, qwen_model, embed_layer,
                    prompt_ids, pad_id, device, features_dict, shuffle_map,
                    num_tokens, per_sample=False
                )
                lora_scaler.scale(loss / grad_accum).backward()
                ac += 1
                if ac == grad_accum or (i + batch_size >= len(epoch_recs)):
                    lora_scaler.unscale_(lora_opt)
                    torch.nn.utils.clip_grad_norm_(proj_params + lora_params, max_norm=1.0)
                    lora_scaler.step(lora_opt)
                    lora_scaler.update()
                    lora_opt.zero_grad()
                    ac = 0
            qwen_model.eval()
            lvl = evaluate_split_all_captions(val_recs)
            lora_val_curve.append((le, lvl))
            if print_epoch_curve:
                print(f"  [LoRA Epoch {le:2d}/{lora_epochs}] Time: {time.time() - t_lep:5.1f} s | Val Loss: {lvl:.4f}", flush=True)
        peak_mb = (torch.cuda.max_memory_allocated() / (1024 ** 2)) if device == "cuda" else 0.0
        lora_info = {
            "lora_rank": lora_rank,
            "lora_lr": lora_lr,
            "lora_epochs": lora_epochs,
            "projector_params": n_proj,
            "lora_params": n_lora,
            "trainable_params": n_proj + n_lora,
            "total_params": n_total,
            "trainable_pct": 100.0 * (n_proj + n_lora) / max(1, n_total),
            "peak_cuda_mb": peak_mb,
            "lora_time_s": time.time() - t_lora,
            "lora_val_curve": lora_val_curve,
        }
        print(
            f"  [LoRA Summary] r={lora_rank}, lr={lora_lr:.1e} | Trainable: {n_proj + n_lora:,} "
            f"({lora_info['trainable_pct']:.3f}%, LoRA={n_lora:,}) | Peak CUDA: {peak_mb:.1f} MB | Final Val: {lora_val_curve[-1][1]:.4f}",
            flush=True,
        )

    final_val_loss = val_losses[-1][1]
    final_test_loss = final_val_loss
    per_image_test_losses = []
    mean_bleu4 = 0.0
    generated_samples = []

    if eval_test:
        final_test_loss, per_image_test_losses = evaluate_split_all_captions(test_recs, return_per_image=True)

        if projector is not None:
            projector.eval()
        if learned_prefix is not None:
            learned_prefix.eval()

        im_end_ids = tokenizer.encode("<|im_end|>", add_special_tokens=False)
        eos_ids = set(x for x in [tokenizer.eos_token_id, tokenizer.pad_token_id, 151643, 151645, 198] + im_end_ids if x is not None)
        sample_test_recs = test_recs[:10]
        p_tensor = torch.tensor([prompt_ids], device=device, dtype=torch.long)
        with torch.no_grad():
            text_embeds_prompt = embed_layer(p_tensor)
            for r in sample_test_recs:
                iid = r["image_id"]
                v_pref = v_prefix_fn([(iid, [])], 1)
                curr_embeds = (
                    torch.cat([v_pref.to(text_embeds_prompt.dtype), text_embeds_prompt], dim=1)
                    if v_pref is not None else text_embeds_prompt
                )
                gen_toks = []
                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=amp_enabled):
                    for _ in range(32):
                        out = qwen_model(inputs_embeds=curr_embeds)
                        next_tok = torch.argmax(out.logits[:, -1, :], dim=-1)
                        tok_id = next_tok.item()
                        if tok_id in eos_ids:
                            break
                        tok_piece = tokenizer.decode([tok_id], skip_special_tokens=False)
                        if "<|im_end|>" in tok_piece or "<|endoftext|>" in tok_piece or "\n" in tok_piece:
                            break
                        gen_toks.append(tok_id)
                        next_embed = embed_layer(next_tok.unsqueeze(0))
                        curr_embeds = torch.cat([curr_embeds, next_embed], dim=1)

                gen_text = tokenizer.decode(gen_toks, skip_special_tokens=True).strip()
                if "\n" in gen_text:
                    gen_text = gen_text.split("\n")[0].strip()

                ref_texts = [c["caption"] for c in r["captions"]]
                ref_toks_list = [c.lower().split() for c in ref_texts]
                bleu4 = compute_sentence_bleu(ref_toks_list, gen_text.lower().split(), max_n=4)
                generated_samples.append({
                    "image_id": iid,
                    "generated": gen_text,
                    "references": ref_texts,
                    "bleu4": bleu4
                })

        mean_bleu4 = float(np.mean([s["bleu4"] for s in generated_samples]))

    if lora_stage:
        detach_lora_from_attention(qwen_model)
        qwen_model.eval()

    return {
        "mode": mode,
        "seed": seed,
        "lr": lr,
        "epochs": epochs if mode != "text_only" else 0,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "optimizer_updates": optimizer_updates,
        "images_seen": images_seen,
        "step0_val_loss": step0_val_loss,
        "final_val_loss": final_val_loss,
        "best_val_epoch": best_ep,
        "best_val_loss": best_vl,
        "final_test_loss": final_test_loss,
        "per_image_test_losses": per_image_test_losses,
        "mean_bleu4": mean_bleu4,
        "generated_samples": generated_samples,
        "val_losses": val_losses,
        "lora_info": lora_info,
        "first_20_update_losses": update_losses[:20],
        "last_20_update_losses": update_losses[-20:],
        "training_time_s": elapsed,
        "ms_per_update": (elapsed / max(1, optimizer_updates)) * 1000 if optimizer_updates > 0 else 0.0,
    }


# ==============================================================================
# 9. Statistical Summary & Hypothesis Testing
# ==============================================================================
def paired_bootstrap_test(diffs, n_resamples=10000, alpha=0.05):
    rng = np.random.default_rng(42)
    n = len(diffs)
    means = [float(np.mean(rng.choice(diffs, size=n, replace=True))) for _ in range(n_resamples)]
    return float(np.mean(diffs)), float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1.0 - alpha / 2)))


# ==============================================================================
# 10. Main Orchestrator
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Batched & FP16 Multimodal Projector Benchmark")
    parser.add_argument("--data_dir", type=str, default="./coco_data", help="Directory for COCO data")
    parser.add_argument("--num_train", type=int, default=2000, help="Train image count")
    parser.add_argument("--num_val", type=int, default=200, help="Val image count")
    parser.add_argument("--num_test", type=int, default=200, help="Test image count")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B", help="Base LLM")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs (default: 5)")
    parser.add_argument("--batch_size", type=int, default=16, help="Parallel mini-batch size (default: 16)")
    parser.add_argument("--grad_accum", type=int, default=1, help="Gradient accumulation steps (default: 1)")
    parser.add_argument("--num_seeds", type=int, default=5, help="Seeds per mode (default: 5)")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Explicit seed list")
    parser.add_argument("--modes", type=str, nargs="+",
                        default=["real", "shuffled", "zeroed", "learned_prefix", "text_only"],
                        help="Modes to evaluate")
    parser.add_argument("--arch", type=str, default="mlp", choices=["mlp", "linear"], help="Projector arch")
    parser.add_argument("--no_pool", action="store_true", help="Disable 2D pooling (use 196 tokens)")
    parser.add_argument("--clip_layer", type=str, default="penultimate", choices=["penultimate", "last"])
    parser.add_argument("--lr", type=float, default=None, help="Fixed LR (skips sweep when provided)")
    parser.add_argument("--sweep", action="store_true", help="Perform validation LR sweep")
    parser.add_argument("--sweep_epochs", type=int, default=3, help="Epochs per LR sweep trial")
    parser.add_argument("--overfit", action="store_true", help="Run 20-image overfit test only")
    parser.add_argument("--verify_data", action="store_true", help="Run alignment diagnostics only")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint JSON if present")
    parser.add_argument("--no_drive", action="store_true", help="Disable Google Drive saving")
    parser.add_argument("--print_epoch_curve", action="store_true", help="Print per-epoch time, train loss, and val loss")
    parser.add_argument("--lora_stage", action="store_true", help="Run optional rank-r LoRA stage on attention projections after projector training")
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA rank (default: 16)")
    parser.add_argument("--lora_lr", type=float, default=1e-5, help="LoRA stage learning rate (default: 1e-5)")
    parser.add_argument("--lora_epochs", type=int, default=5, help="LoRA stage epochs (default: 5)")
    parser.add_argument("--check_fp16", action="store_true", help="Compare FP16 autocast vs FP32 on one fixed batch (loss, grad cosine, NaNs)")
    args = parser.parse_args()

    device, output_dir = setup_environment(use_drive=not args.no_drive)
    ckpt_path = os.path.join(output_dir, "benchmark_checkpoint.json")

    records = acquire_coco_subset(
        data_dir=args.data_dir,
        num_train=args.num_train,
        num_val=args.num_val,
        num_test=args.num_test
    )

    num_tokens = 196 if args.no_pool else 64
    features_dict = extract_clip_features(
        records, args.data_dir, device, no_pool=args.no_pool, clip_layer=args.clip_layer
    )

    print(f"\nLoading {args.model_name} (dtype={'float16' if device == 'cuda' else 'float32'})...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model_dtype = torch.float16 if device == "cuda" else torch.float32
    qwen_model = AutoModelForCausalLM.from_pretrained(args.model_name, dtype=model_dtype).to(device)
    for p in qwen_model.parameters():
        p.requires_grad = False
    qwen_model.eval()

    if args.verify_data:
        verify_data_alignment(records, features_dict)
        return

    if args.check_fp16:
        prompt_ids, pad_id, tokenized = pretokenize_splits(records, tokenizer)
        train_recs = [r for r in records if r["split"] == "train"][:args.batch_size]
        batch_items = [(r["image_id"], tokenized[r["image_id"]][0]) for r in train_recs]
        embed_layer = qwen_model.model.embed_tokens
        def run_one(use_fp16: bool):
            torch.manual_seed(42)
            proj = Projector(768, 896, arch=args.arch, num_tokens=num_tokens, use_pos_embed=True).to(device).float()
            proj.train()
            qwen_dtype = torch.float16 if (use_fp16 and device == "cuda") else torch.float32
            qwen_model.to(dtype=qwen_dtype)
            def v_fn(items, B):
                v_raw = torch.stack([features_dict[iid] for iid, _ in items]).to(device).float()
                return proj(v_raw)
            tot_ids, text_attn, p_len, resp_lens = build_batch_tensors(batch_items, prompt_ids, pad_id, device)
            B = len(batch_items)
            with torch.no_grad():
                text_embeds = embed_layer(tot_ids)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(use_fp16 and device == "cuda")):
                v_pref = v_fn(batch_items, B)
                spliced = torch.cat([v_pref.to(text_embeds.dtype), text_embeds], dim=1)
                attn_mask = torch.cat([torch.ones((B, v_pref.shape[1]), device=device, dtype=torch.long), text_attn], dim=1)
                pos_ids = attn_mask.long().cumsum(-1) - 1
                pos_ids.masked_fill_(attn_mask == 0, 0)
                pred_len = spliced.shape[1] - 1
                labels = torch.full((B, pred_len), -100, device=device, dtype=torch.long)
                for b, r_len in enumerate(resp_lens):
                    start = v_pref.shape[1] + p_len - 1
                    labels[b, start:start + r_len] = tot_ids[b, p_len:p_len + r_len]
                out = qwen_model(inputs_embeds=spliced, attention_mask=attn_mask, position_ids=pos_ids)
                logits = out.logits[:, :pred_len, :].float()
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100)
            nan_cnt = int(torch.isnan(logits).sum().item() + torch.isinf(logits).sum().item() + int(torch.isnan(loss).item()))
            loss.backward()
            grads = {n: p.grad.detach().float().clone() for n, p in proj.named_parameters()}
            nan_cnt += sum(int(torch.isnan(g).sum().item() + torch.isinf(g).sum().item()) for g in grads.values())
            return float(loss.item()), nan_cnt, grads
        l32, n32, g32 = run_one(False)
        l16, n16, g16 = run_one(True)
        print(f"\n[FP16 Autocast vs FP32 Check | Batch={args.batch_size}]")
        print(f"  FP32: loss={l32:.6f}, NaN/Inf={n32}")
        print(f"  FP16: loss={l16:.6f}, NaN/Inf={n16} | Rel Diff={abs(l16 - l32) / max(1e-12, abs(l32)) * 100:.4f}%")
        for n in g32:
            cos = F.cosine_similarity(g32[n].flatten(), g16[n].flatten(), dim=0).item()
            print(f"    grad {n:<16}: cos={cos:.6f}")
        return

    if args.overfit:
        verify_data_alignment(records, features_dict)
        run_overfit_test(records, features_dict, qwen_model, tokenizer, device, arch=args.arch, num_tokens=num_tokens)
        return

    verify_data_alignment(records, features_dict)
    prompt_ids, pad_id, tokenized = pretokenize_splits(records, tokenizer)
    shuffle_map = build_split_shuffle_map(records, seed=1337)

    candidate_lrs = [5e-4, 1e-3, 2e-3, 3e-3]
    best_lr = args.lr if args.lr is not None else 2e-3
    lr_sweep_results = {}

    if args.sweep and args.lr is None:
        print("\n" + "=" * 90)
        print(f"  LEARNING RATE SWEEP (Val All-Captions, 2 Seeds [42, 1337], {args.sweep_epochs} Epochs, Batch {args.batch_size})")
        print("=" * 90)
        best_val = float("inf")
        for lr in candidate_lrs:
            val_scores = []
            for s in [42, 1337]:
                res = train_and_evaluate(
                    mode="real", seed=s, lr=lr, records=records,
                    features_dict=features_dict, qwen_model=qwen_model, tokenizer=tokenizer,
                    device=device, prompt_ids=prompt_ids, pad_id=pad_id,
                    tokenized=tokenized, shuffle_map=shuffle_map,
                    epochs=args.sweep_epochs, batch_size=args.batch_size, grad_accum=args.grad_accum,
                    arch=args.arch, num_tokens=num_tokens, eval_test=False
                )
                val_scores.append(res["final_val_loss"])
            mean_vl = float(np.mean(val_scores))
            lr_sweep_results[str(lr)] = mean_vl
            print(f"  LR = {lr:.1e} -> Mean Val Loss = {mean_vl:.4f} (Seeds: {[round(x, 4) for x in val_scores]})")
            if mean_vl < best_val:
                best_val = mean_vl
                best_lr = lr
        print(f"\nSelected Optimal Learning Rate: {best_lr:.1e} (Mean Val Loss: {best_val:.4f})\n")

    all_seeds = (args.seeds if args.seeds else DEFAULT_SEEDS)[:args.num_seeds]
    modes = args.modes

    benchmark_results = {m: [] for m in modes}
    if args.resume and os.path.exists(ckpt_path):
        with open(ckpt_path, "r") as f:
            saved = json.load(f)
        for m in modes:
            if m in saved.get("runs", {}):
                benchmark_results[m] = saved["runs"][m]
        print(f"[Resume] Loaded partial checkpoint from {ckpt_path}")

    updates_per_epoch = math.ceil(math.ceil(args.num_train / args.batch_size) / args.grad_accum)
    total_updates = args.epochs * updates_per_epoch

    print("\n" + "=" * 96)
    print(f"  BATCHED MULTI-EPOCH MULTIMODAL BENCHMARK ({len(all_seeds)} Seeds: {all_seeds})")
    print(f"  Arch: {args.arch.upper()} | CLIP: {args.clip_layer} | Tokens: {num_tokens} | LR: {best_lr:.1e}")
    print(f"  Epochs: {args.epochs} | Batch Size: {args.batch_size} | Grad Accum: {args.grad_accum} | "
          f"Optimizer Updates/Run: {total_updates} | Images Seen/Run: {args.epochs * args.num_train}")
    print(f"  Precision: {'FP16 Autocast + FP32 Master Weights' if device == 'cuda' else 'FP32 CPU'}")
    print("=" * 96)

    for mode in modes:
        existing_by_seed = {r["seed"]: r for r in benchmark_results[mode]}
        if mode == "text_only":
            if existing_by_seed:
                continue
            print(f"Running Mode: [{mode:<14}] (deterministic, no trainable params) ...", end=" ", flush=True)
            res = train_and_evaluate(
                mode=mode, seed=all_seeds[0], lr=best_lr, records=records,
                features_dict=features_dict, qwen_model=qwen_model, tokenizer=tokenizer,
                device=device, prompt_ids=prompt_ids, pad_id=pad_id,
                tokenized=tokenized, shuffle_map=shuffle_map,
                epochs=args.epochs, batch_size=args.batch_size, grad_accum=args.grad_accum,
                arch=args.arch, num_tokens=num_tokens, eval_test=True
            )
            benchmark_results[mode] = [res]
            print(f"Done in {res['training_time_s']:.1f} s | Test Loss: {res['final_test_loss']:.4f} | BLEU-4: {res['mean_bleu4']:.4f}")
        else:
            for seed in all_seeds:
                if seed in existing_by_seed:
                    print(f"Skipping Mode: [{mode:<14}] | Seed: {seed:<4} (already in checkpoint)")
                    continue
                print(f"Running Mode: [{mode:<14}] | Seed: {seed:<4} ...", end=" ", flush=True)
                res = train_and_evaluate(
                    mode=mode, seed=seed, lr=best_lr, records=records,
                    features_dict=features_dict, qwen_model=qwen_model, tokenizer=tokenizer,
                    device=device, prompt_ids=prompt_ids, pad_id=pad_id,
                    tokenized=tokenized, shuffle_map=shuffle_map,
                    epochs=args.epochs, batch_size=args.batch_size, grad_accum=args.grad_accum,
                    arch=args.arch, num_tokens=num_tokens, eval_test=True,
                    print_epoch_curve=args.print_epoch_curve,
                    lora_stage=args.lora_stage, lora_rank=args.lora_rank,
                    lora_lr=args.lora_lr, lora_epochs=args.lora_epochs,
                )
                benchmark_results[mode].append(res)
                print(
                    f"Done in {res['training_time_s']:.1f} s | Updates: {res['optimizer_updates']} | "
                    f"Epochs: {res['epochs']} | Test Loss: {res['final_test_loss']:.4f} | BLEU-4: {res['mean_bleu4']:.4f}"
                )
                with open(ckpt_path, "w") as f:
                    json.dump({"lr": best_lr, "lr_sweep": lr_sweep_results, "runs": benchmark_results}, f, indent=2)

    print("\n" + "=" * 110)
    print("  FINAL HELD-OUT TEST EVALUATION SUMMARY (ALL 5 CAPTIONS PER IMAGE = 1,000 EVALUATIONS)")
    print("=" * 110)
    print(f"{'Mode':<16} | {'Seeds':<5} | {'Updates':<7} | {'Median Loss':<11} | {'IQR':<8} | {'Mean ± Std':<18} | {'Mean BLEU-4':<11}")
    print("-" * 110)

    mode_test_losses = {}
    for mode in modes:
        runs = benchmark_results[mode]
        losses = [r["final_test_loss"] for r in runs]
        bleus = [r["mean_bleu4"] for r in runs]
        mode_test_losses[mode] = losses
        upds = runs[0]["optimizer_updates"] if runs else 0
        med = float(np.median(losses))
        iqr = float(np.percentile(losses, 75) - np.percentile(losses, 25))
        mean_l = float(np.mean(losses))
        std_l = float(np.std(losses))
        mean_b = float(np.mean(bleus))
        print(f"{mode:<16} | {len(runs):<5} | {upds:<7} | {med:<11.4f} | {iqr:<8.4f} | {mean_l:.4f} ± {std_l:.4f}   | {mean_b:<11.4f}")
    print("=" * 110)

    # Paired comparisons against every control mode present
    if "real" in mode_test_losses and len(mode_test_losses["real"]) >= 2:
        print("\n" + "=" * 90)
        print("  PAIRED HYPOTHESIS TESTING vs. CONTROLS (TWO-SIDED)")
        print("=" * 90)
        real_arr = np.array(mode_test_losses["real"])
        for ctrl in ["shuffled", "zeroed", "learned_prefix"]:
            if ctrl in mode_test_losses and len(mode_test_losses[ctrl]) == len(real_arr):
                ctrl_arr = np.array(mode_test_losses[ctrl])
                diff = ctrl_arr - real_arr
                mean_d, ci_lo, ci_hi = paired_bootstrap_test(diff)
                wins = int(np.sum(real_arr < ctrl_arr))
                print(f"Real vs. {ctrl:<14} (Wins: {wins}/{len(real_arr)}):")
                print(f"  Per-seed ({ctrl} - real): {np.round(diff, 4).tolist()}")
                print(f"  Mean advantage of real: {mean_d:+.4f} | 95% Bootstrap CI: [{ci_lo:+.4f}, {ci_hi:+.4f}]")
                if HAS_SCIPY and len(diff) >= 5 and np.any(diff != 0):
                    w_stat, p_val = wilcoxon(ctrl_arr, real_arr, alternative="two-sided")
                    print(f"  Two-sided Wilcoxon signed-rank: W = {w_stat:.1f}, p = {p_val:.4e}")
        print("=" * 90)

    # Sample Qualitative Generations (Seed 0)
    if benchmark_results.get("real"):
        print("\n" + "=" * 90)
        print("  QUALITATIVE EVALUATION: 10 TEST IMAGES (SEED 1)")
        print("=" * 90)
        sample_recs = benchmark_results["real"][0]["generated_samples"]
        for i, s in enumerate(sample_recs):
            print(f"\n[Image ID: {s['image_id']}]")
            print(f"  Ground Truth: \"{s['references'][0]}\"")
            for m in modes:
                if benchmark_results[m]:
                    g = benchmark_results[m][0]["generated_samples"][i]
                    print(f"  - [{m:<14}]: \"{g['generated']}\" (BLEU-4: {g['bleu4']:.4f})")
        print("=" * 90)

    export_payload = {
        "arch": args.arch,
        "clip_layer": args.clip_layer,
        "num_tokens": num_tokens,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "optimizer_updates_per_run": total_updates,
        "images_seen_per_run": args.epochs * args.num_train,
        "seeds": all_seeds,
        "best_learning_rate": best_lr,
        "lr_sweep_results": lr_sweep_results,
        "summary_table": {
            mode: {
                "num_runs": len(benchmark_results[mode]),
                "optimizer_updates": benchmark_results[mode][0]["optimizer_updates"] if benchmark_results[mode] else 0,
                "median_loss": float(np.median(mode_test_losses[mode])),
                "iqr": float(np.percentile(mode_test_losses[mode], 75) - np.percentile(mode_test_losses[mode], 25)),
                "mean_loss": float(np.mean(mode_test_losses[mode])),
                "std_loss": float(np.std(mode_test_losses[mode])),
                "losses": [float(x) for x in mode_test_losses[mode]],
                "mean_bleu4": float(np.mean([r["mean_bleu4"] for r in benchmark_results[mode]])),
            }
            for mode in modes
        },
        "runs": benchmark_results,
        "created_at": datetime.now().isoformat(),
    }

    export_path = os.path.join(output_dir, "pytorch_projector_reference_results.json")
    with open(export_path, "w") as f:
        json.dump(export_payload, f, indent=2)
    print(f"\nSaved complete benchmark artifacts to: {export_path}")


if __name__ == "__main__":
    main()
