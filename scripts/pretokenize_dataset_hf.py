#!/usr/bin/env python3
"""
Pre-tokenize multimodal or text datasets using Hugging Face AutoTokenizer directly.
Outputs pre-tokenized JSONL files containing input_ids and labels for MobileFineTuner,
completely avoiding runtime tokenizer edge-case discrepancies while keeping C++ upstream pristine.
"""

import os
import sys
import json
import argparse
from transformers import AutoTokenizer

def pretokenize_dataset(model_dir: str, input_jsonl: str, output_jsonl: str, max_seq_len: int = 512):
    print(f"Loading tokenizer from {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    print(f"Processing {input_jsonl} -> {output_jsonl} (max_seq_len={max_seq_len})...")
    num_samples = 0

    with open(input_jsonl, "r", encoding="utf-8") as in_f, open(output_jsonl, "w", encoding="utf-8") as out_f:
        for line_idx, line in enumerate(in_f):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)

            sample_id = item.get("id", f"sample_{line_idx}")
            image_path = item.get("image", None)
            
            # Support text or conversations format
            if "conversations" in item:
                # Format conversational turns
                text_parts = []
                for conv in item["conversations"]:
                    role = conv.get("from", "user")
                    val = conv.get("value", "")
                    text_parts.append(f"<|im_start|>{role}\n{val}<|im_end|>\n")
                full_text = "".join(text_parts)
            elif "text" in item:
                full_text = item["text"]
            else:
                print(f"Warning: sample {sample_id} has neither 'conversations' nor 'text'. Skipping.")
                continue

            # Tokenize with exact Hugging Face BPE
            tokens = tokenizer.encode(full_text, add_special_tokens=False)
            if len(tokens) > max_seq_len:
                tokens = tokens[:max_seq_len]

            # Autoregressive labels (target = input_ids, masked prefix handled separately)
            labels = list(tokens)

            record = {
                "id": sample_id,
                "image": image_path,
                "input_ids": tokens,
                "labels": labels,
                "seq_len": len(tokens)
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            num_samples += 1

    print(f"Successfully pre-tokenized {num_samples} samples into {output_jsonl}")

def create_sample_dataset(output_path: str):
    samples = [
        {"id": "doc_001", "text": "The quick brown fox jumps over the lazy dog."},
        {"id": "doc_002", "text": "Deep learning models require careful optimization."},
        {"id": "conv_003", "image": "images/test1.jpg", "conversations": [
            {"from": "user", "value": "What is depicted in the image?"},
            {"from": "assistant", "value": "A red sports car parked outside a garage."}
        ]}
    ]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    print(f"Created sample raw dataset at {output_path}")

def main():
    parser = argparse.ArgumentParser(description="HF Pre-tokenization for MobileFineTuner")
    parser.add_argument("--model-dir", default="models/Qwen2.5-0.5B", help="Path to HF model dir")
    parser.add_argument("--input", default="data/sample_raw.jsonl", help="Input raw JSONL")
    parser.add_argument("--output", default="data/pretokenized_hf.jsonl", help="Output tokenized JSONL")
    parser.add_argument("--max-len", type=int, default=512, help="Max sequence length")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Input file {args.input} does not exist. Generating sample...")
        create_sample_dataset(args.input)

    pretokenize_dataset(args.model_dir, args.input, args.output, args.max_len)

if __name__ == "__main__":
    main()
