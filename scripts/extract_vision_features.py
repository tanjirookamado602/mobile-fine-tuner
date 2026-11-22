#!/usr/bin/env python3
"""
CLIP Vision Feature Extraction Script for Google Colab and Local Execution:
- Model: CLIP ViT-B/16 (openai/clip-vit-base-patch16)
- Input: 224x224 RGB images
- Spatial Tokens: 14x14 = 196 patch tokens -> 2D Adaptive Average Pooled to 8x8 = 64 tokens
- Output Dtype: float16
- Artifacts: visual_features.npy, image_ids.json, manifest.json
"""

import os
import sys
import json
import argparse
import time
from datetime import datetime
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPVisionModel, CLIPImageProcessor

def extract_clip_features(image_paths_or_images, model, processor, device="cuda" if torch.cuda.is_available() else "cpu", pool_grid=8):
    """
    Extracts 64 visual tokens (8x8) of dim 768 from 224x224 images.
    1. Preprocesses image to 224x224 normalized RGB tensor.
    2. Runs CLIP ViT-B/16 forward pass to obtain last_hidden_state [B, 197, 768].
    3. Drops the CLS token (index 0) leaving 196 patch tokens [B, 196, 768].
    4. Reshapes to [B, 768, 14, 14] and applies 2D Adaptive Average Pooling to [B, 768, 8, 8].
    5. Reshapes to [B, 64, 768] and casts to float16.
    """
    model.eval()
    inputs = processor(images=image_paths_or_images, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        # last_hidden_state: [Batch, 197, 768] (index 0 is CLS token)
        hidden_states = outputs.last_hidden_state
        # Drop CLS token -> [Batch, 196, 768]
        patch_tokens = hidden_states[:, 1:, :]
        
        # Reshape to 2D feature map: [Batch, 14, 14, 768] -> [Batch, 768, 14, 14]
        B, num_patches, D = patch_tokens.shape
        grid_h = grid_w = int(num_patches ** 0.5) # 14
        feature_map = patch_tokens.reshape(B, grid_h, grid_w, D).permute(0, 3, 1, 2)
        
        # 2D Adaptive Average Pooling: 14x14 -> 8x8 (64 tokens)
        pooled = F.adaptive_avg_pool2d(feature_map, (pool_grid, pool_grid)) # [B, 768, 8, 8]
        # Reshape back to [B, 64, 768]
        visual_tokens = pooled.permute(0, 2, 3, 1).reshape(B, pool_grid * pool_grid, D)
        
        return visual_tokens.cpu().to(torch.float16).numpy()

def main():
    parser = argparse.ArgumentParser(description="Extract CLIP ViT-B/16 pooled vision features for MobileFineTuner")
    parser.add_argument("--image_dir", type=str, default="data/sample_images", help="Directory containing images")
    parser.add_argument("--output_dir", type=str, default="data/vision_features", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for feature extraction")
    parser.add_argument("--model_name", type=str, default="openai/clip-vit-base-patch16", help="Hugging Face CLIP vision model")
    parser.add_argument("--pool_grid", type=int, default=8, help="Target spatial grid size (8 gives 64 tokens)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Device: {device}")
    print(f"Loading CLIP Vision Model '{args.model_name}'...")
    
    processor = CLIPImageProcessor.from_pretrained(args.model_name)
    model = CLIPVisionModel.from_pretrained(args.model_name).to(device)

    # Collect images
    supported_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    image_files = []
    if os.path.exists(args.image_dir):
        for root, _, files in os.walk(args.image_dir):
            for f in sorted(files):
                if f.lower().endswith(supported_exts):
                    image_files.append(os.path.join(root, f))
    
    # If image dir is empty or does not exist, generate synthetic/sample data for smoke verification
    is_synthetic = False
    if len(image_files) == 0:
        print(f"No images found in {args.image_dir}. Generating 50 benchmark synthetic images for smoke validation...")
        os.makedirs(args.image_dir, exist_ok=True)
        is_synthetic = True
        for i in range(50):
            # Create synthetic patterned RGB images (color ramps / geometric noise)
            arr = np.zeros((224, 224, 3), dtype=np.uint8)
            arr[:, :, 0] = (i * 5) % 256
            arr[:, :, 1] = np.linspace(0, 255, 224, dtype=np.uint8)
            arr[:, :, 2] = np.linspace(255, 0, 224, dtype=np.uint8)[:, None]
            img = Image.fromarray(arr)
            img_path = os.path.join(args.image_dir, f"sample_{i:04d}.jpg")
            img.save(img_path)
            image_files.append(img_path)

    print(f"Total images to process: {len(image_files)}")
    
    all_features = []
    image_ids = []
    
    t0 = time.time()
    for idx in range(0, len(image_files), args.batch_size):
        batch_paths = image_files[idx : idx + args.batch_size]
        batch_imgs = [Image.open(p).convert("RGB") for p in batch_paths]
        feats = extract_clip_features(batch_imgs, model, processor, device=device, pool_grid=args.pool_grid)
        all_features.append(feats)
        for p in batch_paths:
            img_id = os.path.splitext(os.path.basename(p))[0]
            image_ids.append(img_id)
        print(f"  Processed {min(idx + args.batch_size, len(image_files))} / {len(image_files)} images...")
    
    all_features = np.concatenate(all_features, axis=0) # [N, 64, 768]
    elapsed = time.time() - t0
    
    # Save artifacts
    features_path = os.path.join(args.output_dir, "visual_features.npy")
    ids_path = os.path.join(args.output_dir, "image_ids.json")
    manifest_path = os.path.join(args.output_dir, "manifest.json")
    
    np.save(features_path, all_features)
    with open(ids_path, "w") as f:
        json.dump(image_ids, f, indent=2)
        
    manifest = {
        "model_name": args.model_name,
        "input_resolution": [224, 224],
        "patch_size": 16,
        "raw_patches": 196,
        "pooling_method": f"spatial_2d_adaptive_avg_pool_14x14_to_{args.pool_grid}x{args.pool_grid}",
        "output_tokens_per_image": args.pool_grid * args.pool_grid,
        "feature_dim": 768,
        "dtype": "float16",
        "num_images": len(image_ids),
        "feature_shape": list(all_features.shape),
        "extraction_time_seconds": round(elapsed, 2),
        "is_synthetic": is_synthetic,
        "created_at": datetime.now().isoformat()
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
        
    print("\n" + "=" * 80)
    print("EXTRACTION COMPLETE:")
    print(f"  Features Shape: {all_features.shape} ({all_features.dtype})")
    print(f"  Output File:    {features_path} ({os.path.getsize(features_path) / 1024 / 1024:.2f} MB)")
    print(f"  Manifest File:  {manifest_path}")
    print(f"  Throughput:     {len(image_files) / elapsed:.1f} images/second")
    print("=" * 80)

if __name__ == "__main__":
    main()
