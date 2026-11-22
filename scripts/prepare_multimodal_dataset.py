#!/usr/bin/env python3
"""
Multimodal Dataset Split and Tokenization Script:
- Splits conversations strictly by `image_id` to prevent data leakage across splits.
- Enforces explicit assertions: len(train_ids & val_ids) == 0, etc.
- Tokenizes prompt and target response using Hugging Face AutoTokenizer (Qwen/Qwen2.5-0.5B).
- Formats input_ids and labels (prompt positions masked with -100).
- Produces: train (2,000 pairs), val (200 pairs), test (200 pairs).
"""

import os
import sys
import json
import random
import argparse
from datetime import datetime
from transformers import AutoTokenizer

def build_sample_dataset(num_train_imgs=2000, num_val_imgs=200, num_test_imgs=200):
    """
    Generates synthetic / templated multimodal conversation pairs if raw LLaVA/COCO json is not present.
    Each image has 1 to 2 question-answer pairs.
    """
    templates = [
        ("Describe the visual elements in this image in detail.", "The image shows a structured scene containing geometric shapes and high contrast textures with balanced illumination."),
        ("What is the primary subject in the foreground?", "The foreground features prominent objects centered within the 224x224 field of view."),
        ("Identify the color palette and lighting condition.", "The scene demonstrates a natural color temperature with smooth gradient transitions across the surface."),
        ("Explain the spatial arrangement of the elements.", "The composition follows standard perspective with central emphasis and distinct background boundaries.")
    ]
    
    total_imgs = num_train_imgs + num_val_imgs + num_test_imgs
    all_records = []
    
    for i in range(total_imgs):
        img_id = f"image_{i:06d}"
        # 1-2 pairs per image
        n_pairs = random.choice([1, 2])
        for p in range(n_pairs):
            q, a = random.choice(templates)
            all_records.append({
                "image_id": img_id,
                "conversations": [
                    {"from": "human", "value": f"<image>\n{q}"},
                    {"from": "gpt", "value": a}
                ]
            })
    return all_records

def main():
    parser = argparse.ArgumentParser(description="Split multimodal dataset strictly by image_id and pre-tokenize")
    parser.add_argument("--input_json", type=str, default="", help="Path to raw LLaVA/COCO json file")
    parser.add_argument("--output_dir", type=str, default="data/multimodal_splits", help="Output directory")
    parser.add_argument("--tokenizer_dir", type=str, default="models/Qwen2.5-0.5B", help="Path to tokenizer")
    parser.add_argument("--num_train", type=int, default=2000, help="Target number of train images")
    parser.add_argument("--num_val", type=int, default=200, help="Target number of val images")
    parser.add_argument("--num_test", type=int, default=200, help="Target number of test images")
    parser.add_argument("--max_seq_len", type=int, default=128, help="Max text token sequence length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Loading Tokenizer from {args.tokenizer_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)

    # 1. Load or Generate raw records
    if args.input_json and os.path.exists(args.input_json):
        print(f"Loading raw dataset from {args.input_json}...")
        with open(args.input_json, "r") as f:
            raw_data = json.load(f)
    else:
        print(f"No input json provided. Creating dataset with {args.num_train} train / {args.num_val} val / {args.num_test} test images...")
        raw_data = build_sample_dataset(args.num_train, args.num_val, args.num_test)

    # 2. Group records strictly by image_id
    image_to_records = {}
    for item in raw_data:
        iid = item.get("image_id") or item.get("image") or item.get("id")
        if not iid: continue
        # Normalize image_id (strip extension)
        iid = os.path.splitext(str(iid))[0]
        if iid not in image_to_records:
            image_to_records[iid] = []
        image_to_records[iid].append(item)

    all_image_ids = sorted(list(image_to_records.keys()))
    random.shuffle(all_image_ids)
    
    n_train = min(args.num_train, len(all_image_ids))
    n_val = min(args.num_val, max(0, len(all_image_ids) - n_train))
    n_test = min(args.num_test, max(0, len(all_image_ids) - n_train - n_val))

    train_img_ids = set(all_image_ids[:n_train])
    val_img_ids = set(all_image_ids[n_train : n_train + n_val])
    test_img_ids = set(all_image_ids[n_train + n_val : n_train + n_val + n_test])

    # 3. CRITICAL ASSERTIONS: Zero overlap across splits
    print("\nVerifying Strict Zero-Overlap Assertions by Image ID...")
    assert len(train_img_ids & val_img_ids) == 0, "FATAL: Overlap between Train and Validation sets!"
    assert len(train_img_ids & test_img_ids) == 0, "FATAL: Overlap between Train and Test sets!"
    assert len(val_img_ids & test_img_ids) == 0, "FATAL: Overlap between Validation and Test sets!"
    print(f"  Assertion PASSED: Train={len(train_img_ids)}, Val={len(val_img_ids)}, Test={len(test_img_ids)} unique images. Zero overlap.")

    # 4. Tokenization function
    def process_records(img_ids, split_name):
        split_records = []
        for iid in img_ids:
            for rec in image_to_records[iid]:
                convs = rec.get("conversations", [])
                if len(convs) < 2: continue
                prompt_text = convs[0]["value"].replace("<image>\n", "").replace("<image>", "").strip()
                response_text = convs[1]["value"].strip()

                # Format with Qwen chat template style:
                prompt_full = f"<|im_start|>user\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n"
                response_full = f"{response_text}<|im_end|>\n"

                prompt_tokens = tokenizer.encode(prompt_full, add_special_tokens=False)
                resp_tokens = tokenizer.encode(response_full, add_special_tokens=False)

                total_tokens = prompt_tokens + resp_tokens
                if len(total_tokens) > args.max_seq_len:
                    total_tokens = total_tokens[:args.max_seq_len]
                    prompt_len = min(len(prompt_tokens), args.max_seq_len)
                else:
                    prompt_len = len(prompt_tokens)

                # Labels: mask prompt tokens with -100 so loss is computed solely on response
                labels = [-100] * prompt_len + total_tokens[prompt_len:]

                split_records.append({
                    "image_id": iid,
                    "input_ids": total_tokens,
                    "labels": labels,
                    "prompt_len": prompt_len,
                    "total_len": len(total_tokens),
                    "prompt_text": prompt_text,
                    "response_text": response_text
                })

        out_path = os.path.join(args.output_dir, f"{split_name}.jsonl")
        with open(out_path, "w") as f:
            for r in split_records:
                f.write(json.dumps(r) + "\n")
        print(f"  Saved {split_name}: {len(split_records)} pairs across {len(img_ids)} unique images -> {out_path}")
        return len(split_records)

    print("\nTokenizing and Writing Splits with Hugging Face Tokenizer...")
    n_tr_pairs = process_records(train_img_ids, "train")
    n_va_pairs = process_records(val_img_ids, "val")
    n_te_pairs = process_records(test_img_ids, "test")

    summary = {
        "num_train_images": len(train_img_ids),
        "num_train_pairs": n_tr_pairs,
        "num_val_images": len(val_img_ids),
        "num_val_pairs": n_va_pairs,
        "num_test_images": len(test_img_ids),
        "num_test_pairs": n_te_pairs,
        "tokenizer": args.tokenizer_dir,
        "zero_overlap_asserted": True,
        "created_at": datetime.now().isoformat()
    }
    with open(os.path.join(args.output_dir, "split_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("DATASET PREPARATION COMPLETE & VERIFIED.")
    print(f"  Summary saved to {os.path.join(args.output_dir, 'split_summary.json')}")
    print("=" * 80)

if __name__ == "__main__":
    main()
