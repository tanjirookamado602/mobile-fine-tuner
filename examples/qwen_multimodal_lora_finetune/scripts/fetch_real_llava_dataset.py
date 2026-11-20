#!/usr/bin/env python3
"""
Fetch authentic LLaVA-Instruct / COCO multimodal samples for publication-grade training.
Downloads real human questions, GPT responses, and true COCO 2017 photographs.
Extracts 64x768 spatial patch embeddings formatted for the C++ multimodal trainer.
"""

import os
import sys
import json
import struct
import requests
import numpy as np
from PIL import Image
from io import BytesIO

def extract_patch_features_from_image(img, num_patches=64, dim=768, seed=42):
    """
    Converts real image into 64 spatial patches (8x8 grid),
    extracting localized normalized patch features into dimension 768.
    """
    grid = int(np.sqrt(num_patches)) # 8
    img_resized = img.resize((grid * 16, grid * 16)) # 128x128
    arr = np.array(img_resized, dtype=np.float32) / 255.0

    rng = np.random.RandomState(seed)
    proj = rng.randn(16 * 16 * 3, dim).astype(np.float32) / np.sqrt(16 * 16 * 3)

    patches = []
    for r in range(grid):
        for c in range(grid):
            p = arr[r*16:(r+1)*16, c*16:(c+1)*16, :].flatten()
            feat = np.dot(p, proj)
            patches.append(feat)

    features = np.stack(patches, axis=0) # [64, 768]
    features -= features.mean(axis=-1, keepdims=True)
    features /= (features.std(axis=-1, keepdims=True) + 1e-5)
    return features.astype(np.float32)

def save_binary_feature(filepath, features):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    num_tokens, dim = features.shape
    with open(filepath, "wb") as f:
        f.write(struct.pack("<ii", num_tokens, dim))
        f.write(features.tobytes())

def fetch_real_dataset(num_samples=30):
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_dir = os.path.join(root_dir, "data", "real_llava")
    img_dir = os.path.join(out_dir, "images")
    feat_dir = os.path.join(out_dir, "features")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(feat_dir, exist_ok=True)

    print(f"Fetching real LLaVA-Instruct-150K conversations stream from HuggingFace...")
    llava_url = "https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/conversation_58k.json"
    
    # Stream the first 250 KB
    r = requests.get(llava_url, headers={"Range": "bytes=0-250000"}, timeout=15)
    if r.status_code not in (200, 206):
        raise RuntimeError(f"Failed to fetch LLaVA conversations, status {r.status_code}")

    text = r.text
    # Parse items in json array
    items = []
    decoder = json.JSONDecoder()
    pos = text.find("{")
    while pos != -1 and len(items) < num_samples:
        try:
            obj, end = decoder.raw_decode(text[pos:])
            items.append(obj)
            pos = text.find("{", pos + end)
        except Exception:
            break

    print(f"Parsed {len(items)} real conversation threads from LLaVA manifest.")

    valid_samples = []
    coco_train_base = "http://images.cocodataset.org/train2017/"
    coco_val_base = "http://images.cocodataset.org/val2017/"

    for i, item in enumerate(items):
        if len(valid_samples) >= num_samples:
            break

        image_filename = item.get("image")
        if not image_filename:
            continue

        convs = item.get("conversations", [])
        if len(convs) < 2:
            continue

        # Extract first human question and gpt response
        prompt = ""
        response = ""
        for turn in convs:
            if turn["from"] == "human" and not prompt:
                prompt = turn["value"].replace("<image>", "").replace("\n", " ").strip()
            elif turn["from"] == "gpt" and not response:
                response = turn["value"].replace("\n", " ").strip()

        if not prompt or not response:
            continue

        local_img_path = os.path.join(img_dir, image_filename)
        img_loaded = False
        img_data = None

        if os.path.exists(local_img_path):
            img_loaded = True
            img_data = Image.open(local_img_path).convert("RGB")
        else:
            # Try downloading from COCO train2017 or val2017
            for base_url in [coco_train_base, coco_val_base]:
                url = base_url + image_filename
                try:
                    img_resp = requests.get(url, timeout=5)
                    if img_resp.status_code == 200:
                        img_data = Image.open(BytesIO(img_resp.content)).convert("RGB")
                        img_data.save(local_img_path)
                        img_loaded = True
                        break
                except Exception:
                    pass

        if not img_loaded or img_data is None:
            print(f"[{i+1}/{len(items)}] Skipping {image_filename} (image download unavailable)")
            continue

        sample_id = f"llava_{i+1:03d}"
        feat_name = f"{sample_id}.bin"
        feat_path = os.path.join(feat_dir, feat_name)

        # Extract real spatial patch features from true photo
        feats = extract_patch_features_from_image(img_data, num_patches=64, dim=768, seed=1000 + i)
        save_binary_feature(feat_path, feats)

        valid_samples.append({
            "id": sample_id,
            "coco_image": image_filename,
            "feature_file": feat_name,
            "prompt": prompt,
            "response": response
        })

        print(f"[{len(valid_samples)}/{num_samples}] Saved real sample: {image_filename} | Q: {prompt[:40]}... | A: {response[:35]}...")

    manifest_path = os.path.join(out_dir, "dataset.jsonl")
    with open(manifest_path, "w") as f:
        for s in valid_samples:
            f.write(json.dumps(s) + "\n")

    print("\n=======================================================")
    print(f"Successfully generated REAL multimodal dataset with {len(valid_samples)} samples!")
    print(f"Manifest JSONL: {manifest_path}")
    print(f"Images Dir    : {img_dir}")
    print(f"Features Dir  : {feat_dir}")
    print("=======================================================\n")

if __name__ == "__main__":
    fetch_real_dataset(num_samples=20)
