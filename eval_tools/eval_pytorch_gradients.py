#!/usr/bin/env python3
"""
PyTorch gradient and inputs_embeds forward parity evaluator against MobileFineTuner C++
"""

import os
import struct
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_bin_floats(path, count):
    with open(path, "rb") as f:
        data = np.frombuffer(f.read(count * 4), dtype=np.float32)
    return data

def main():
    print("=" * 80)
    print("PyTorch Reference Forward & Gradient Evaluator (inputs_embeds & ConcatBackward)")
    print("=" * 80)

    model_dir = "models/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32)

    # 1. Load exact binary prefix [1, 8, 896] exported by C++
    prefix_np = load_bin_floats("eval_tools/fixed_prefix_8x896.bin", 8 * 896).reshape(1, 8, 896)
    prefix_torch = torch.tensor(prefix_np, dtype=torch.float32, requires_grad=True)

    # 2. Tokenize prompt to 16 tokens
    prompt = "The quick brown fox jumps over the lazy dog. Artificial intelligence on mobile devices enables fast"
    tokens = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = tokens["input_ids"][:, :16]

    # 3. Embedding lookup for text tokens
    text_embeds = model.model.embed_tokens(input_ids) # [1, 16, 896]

    # 4. Concatenate prefix [1, 8, 896] and text [1, 16, 896] -> [1, 24, 896]
    spliced_embeds = torch.cat([prefix_torch, text_embeds], dim=1)

    # 5. Forward pass via inputs_embeds
    outputs = model(inputs_embeds=spliced_embeds)
    hf_logits = outputs.logits # [1, 24, 151936]
    np_hf_logits = hf_logits.detach().cpu().numpy().astype(np.float32)

    # 6. Load targets [23]
    with open("eval_tools/targets_23.bin", "rb") as f:
        target_np = np.frombuffer(f.read(23 * 4), dtype=np.int32)
    targets_torch = torch.tensor(target_np, dtype=torch.long)

    # 7. Compute loss
    flat_logits = hf_logits[:, :23, :].reshape(23, 151936)
    loss = F.cross_entropy(flat_logits, targets_torch, ignore_index=-100)
    loss_val = loss.item()
    print(f"PyTorch Cross-Entropy Loss: {loss_val:.6f}")

    # 8. Backward pass
    loss.backward()

    # Extract PyTorch Gradients
    hf_grad_prefix = prefix_torch.grad.cpu().numpy() # [1, 8, 896]
    hf_grad_qproj = model.model.layers[23].self_attn.q_proj.weight.grad.cpu().numpy() # [896, 896]
    hf_grad_downproj = model.model.layers[23].mlp.down_proj.weight.grad.cpu().numpy() # [896, 4864]
    hf_grad_norm = model.model.norm.weight.grad.cpu().numpy() # [896]

    # Load C++ outputs
    cpp_logits = load_bin_floats("eval_tools/cpp_spliced_logits_fp32.bin", 1 * 24 * 151936).reshape(1, 24, 151936)
    cpp_grad_prefix = load_bin_floats("eval_tools/cpp_grad_prefix.bin", 1 * 8 * 896).reshape(1, 8, 896)
    cpp_grad_qproj = load_bin_floats("eval_tools/cpp_grad_qproj23.bin", 896 * 896).reshape(896, 896)
    cpp_grad_downproj = load_bin_floats("eval_tools/cpp_grad_downproj23.bin", 4864 * 896).reshape(4864, 896)
    cpp_grad_norm = load_bin_floats("eval_tools/cpp_grad_norm.bin", 896)

    print("\n" + "=" * 80)
    print("PARITY ANALYSIS 1: forward_hidden_from_embeds (Logits Parity on [1, 24, 896])")
    print("=" * 80)
    diff_logits = np.abs(cpp_logits - np_hf_logits)
    max_abs_logits = np.max(diff_logits)
    mean_abs_logits = np.mean(diff_logits)

    cpp_top1 = np.argmax(cpp_logits, axis=-1)[0]
    hf_top1 = np.argmax(np_hf_logits, axis=-1)[0]
    top1_matches = (cpp_top1 == hf_top1)
    top1_acc = np.mean(top1_matches) * 100.0

    print(f"  Max Absolute Logit Diff:  {max_abs_logits:.6e}")
    print(f"  Mean Absolute Logit Diff: {mean_abs_logits:.6e}")
    print(f"  Top-1 Agreement:          {np.sum(top1_matches)}/24 ({top1_acc:.1f}%)")

    print("\n" + "=" * 80)
    print("PARITY ANALYSIS 2: Gradient Correctness (ConcatBackward & Model Backward)")
    print("=" * 80)

    def print_grad_metrics(name, cpp_g, hf_g):
        diff = np.abs(cpp_g - hf_g)
        max_abs = np.max(diff)
        mean_abs = np.mean(diff)
        denom = np.abs(hf_g) + 1e-7
        rel = diff / denom
        max_rel = np.max(rel)
        mean_rel = np.mean(rel)
        # Cosine similarity
        cos_sim = np.dot(cpp_g.flatten(), hf_g.flatten()) / (np.linalg.norm(cpp_g) * np.linalg.norm(hf_g) + 1e-9)
        print(f"[{name}]")
        print(f"  Shapes: C++ {cpp_g.shape} vs HF {hf_g.shape}")
        print(f"  Cosine Similarity:       {cos_sim:.6f}")
        print(f"  Max Absolute Difference: {max_abs:.6e}")
        print(f"  Mean Absolute Error:     {mean_abs:.6e}")
        print(f"  Mean Relative Error:     {mean_rel:.6e}")
        return cos_sim, max_abs, mean_abs

    # 1. Prefix Embedding Gradient (exercises ConcatBackward!)
    print_grad_metrics("Prefix Embeddings [1, 8, 896] (ConcatBackward Test)", cpp_grad_prefix, hf_grad_prefix)

    # 2. Final Layer Norm Weight Gradient
    print_grad_metrics("Final Layer Norm Weight [896]", cpp_grad_norm, hf_grad_norm)

    # 3. Layer 23 Q-Proj Weight Gradient
    # In C++, W is [in, out] = [896, 896]. In PyTorch, W is [out, in] = [896, 896]. Transpose PyTorch for [in, out] comparison
    print_grad_metrics("Layer 23 Q-Proj Weight [896, 896]", cpp_grad_qproj, hf_grad_qproj.T)

    # 4. Layer 23 MLP Down-Proj Weight Gradient
    # In C++, W is [4864, 896]. In PyTorch, W is [896, 4864].
    print_grad_metrics("Layer 23 Down-Proj Weight [4864, 896]", cpp_grad_downproj, hf_grad_downproj.T)

if __name__ == "__main__":
    main()
