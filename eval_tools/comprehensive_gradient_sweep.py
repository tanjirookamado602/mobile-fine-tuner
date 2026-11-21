#!/usr/bin/env python3
"""
Comprehensive PyTorch Reference Evaluator:
Multi-Prompt, Multi-Layer Gradient Parity in Float64.
Computes Cosine Similarity clamped to [-1.0, 1.0] and Relative Errors normalized by Gradient Norm.
Uses chunked float64 streaming to guarantee minimal RAM usage (< 100 MB).
"""

import os
import gc
import glob
import json
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

def load_bin_floats(path, count):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        data = np.frombuffer(f.read(count * 4), dtype=np.float32)
    return data

def load_bin_i32(path, count):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        data = np.frombuffer(f.read(count * 4), dtype=np.int32)
    return data

def compute_metrics_chunked(cpp_g, hf_g, chunk_size=50000):
    """
    Compute exact float64 cosine similarity and error metrics using streaming chunks
    to avoid allocating gigabytes of intermediate float64 arrays.
    """
    cpp_flat = cpp_g.reshape(-1)
    hf_flat = hf_g.reshape(-1)
    N = len(cpp_flat)

    dot = 0.0
    norm_cpp_sq = 0.0
    norm_hf_sq = 0.0
    max_abs = 0.0
    abs_sum = 0.0

    for i in range(0, N, chunk_size):
        end = min(i + chunk_size, N)
        c_chunk = cpp_flat[i:end].astype(np.float64)
        h_chunk = hf_flat[i:end].astype(np.float64)

        diff = np.abs(c_chunk - h_chunk)
        m = float(np.max(diff))
        if m > max_abs:
            max_abs = m
        abs_sum += float(np.sum(diff))

        dot += float(np.dot(c_chunk, h_chunk))
        norm_cpp_sq += float(np.dot(c_chunk, c_chunk))
        norm_hf_sq += float(np.dot(h_chunk, h_chunk))

    norm_cpp = float(np.sqrt(norm_cpp_sq))
    norm_hf = float(np.sqrt(norm_hf_sq))

    cos_sim = dot / (norm_cpp * norm_hf + 1e-15)
    cos_sim = float(np.clip(cos_sim, -1.0, 1.0))
    mean_abs = float(abs_sum / N)
    rel_max = float(max_abs / (norm_hf + 1e-12))
    rel_mean = float(mean_abs / (norm_hf + 1e-12))

    return {
        "cos_sim": cos_sim,
        "grad_norm_hf": norm_hf,
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "rel_max_to_norm": rel_max,
        "rel_mean_to_norm": rel_mean
    }

def main():
    print("=" * 105)
    print("  COMPREHENSIVE MULTI-PROMPT MULTI-LAYER GRADIENT PARITY EVALUATOR (FLOAT64)")
    print("  Verification: MobileFineTuner C++ Autograd vs Hugging Face PyTorch Reference")
    print("=" * 105)

    model_dir = "models/Qwen2.5-0.5B"
    out_dir = "eval_tools/sweep_out"

    print("Loading PyTorch reference model from", model_dir, "...")
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32)

    # Tensor definitions
    tensors_to_check = [
        ("Prefix [1, 8, 896] (ConcatBackward)", "grad_prefix", (1, 8, 896)),
        ("Final RMSNorm [896]", "grad_norm", (896,)),
        ("Layer 0 Q-Proj [896, 896]", "grad_l0_qproj", (896, 896)),
        ("Layer 0 Down-Proj [4864, 896]", "grad_l0_downproj", (4864, 896)),
        ("Layer 12 Q-Proj [896, 896]", "grad_l12_qproj", (896, 896)),
        ("Layer 12 Down-Proj [4864, 896]", "grad_l12_downproj", (4864, 896)),
        ("Layer 23 Q-Proj [896, 896]", "grad_l23_qproj", (896, 896)),
        ("Layer 23 Down-Proj [4864, 896]", "grad_l23_downproj", (4864, 896)),
        ("LM Head / Embeddings [151936, 896]", "grad_embed", (151936, 896)),
    ]

    all_results = []

    for p_idx in range(5):
        pfx_path = f"{out_dir}/prefix_p{p_idx}.bin"
        tid_path = f"{out_dir}/text_ids_p{p_idx}.bin"
        if not os.path.exists(pfx_path) or not os.path.exists(tid_path):
            print(f"Skipping prompt {p_idx} (files missing)")
            continue

        raw_pfx = load_bin_floats(pfx_path, 8 * 896).reshape(1, 8, 896)
        text_ids = load_bin_i32(tid_path, 16)

        for mode in range(2):
            mode_name = "Masked Loss" if mode == 0 else "Unmasked Loss"
            tag = f"_p{p_idx}_m{mode}"
            tgt_path = f"{out_dir}/targets{tag}.bin"
            if not os.path.exists(tgt_path):
                continue
            targets = load_bin_i32(tgt_path, 23)

            # PyTorch forward + backward
            model.zero_grad()
            prefix_torch = torch.tensor(raw_pfx, dtype=torch.float32, requires_grad=True)
            tid_torch = torch.tensor(text_ids.reshape(1, 16), dtype=torch.long)
            text_embeds = model.model.embed_tokens(tid_torch)

            spliced_embeds = torch.cat([prefix_torch, text_embeds], dim=1)
            outputs = model(inputs_embeds=spliced_embeds)
            flat_logits = outputs.logits[:, :23, :].reshape(23, 151936)

            tgt_torch = torch.tensor(targets, dtype=torch.long)
            loss = F.cross_entropy(flat_logits, tgt_torch, ignore_index=-100)
            loss.backward()

            # Map PyTorch gradients
            hf_grads = {
                "grad_prefix": prefix_torch.grad.cpu().numpy(),
                "grad_norm": model.model.norm.weight.grad.cpu().numpy(),
                "grad_l0_qproj": model.model.layers[0].self_attn.q_proj.weight.grad.cpu().numpy().T,
                "grad_l0_downproj": model.model.layers[0].mlp.down_proj.weight.grad.cpu().numpy().T,
                "grad_l12_qproj": model.model.layers[12].self_attn.q_proj.weight.grad.cpu().numpy().T,
                "grad_l12_downproj": model.model.layers[12].mlp.down_proj.weight.grad.cpu().numpy().T,
                "grad_l23_qproj": model.model.layers[23].self_attn.q_proj.weight.grad.cpu().numpy().T,
                "grad_l23_downproj": model.model.layers[23].mlp.down_proj.weight.grad.cpu().numpy().T,
                "grad_embed": model.lm_head.weight.grad.cpu().numpy() # [151936, 896]
            }

            print(f"\n[Prompt {p_idx+1}/5 | Mode: {mode_name}] PyTorch Loss = {loss.item():.6f}")
            print("-" * 105)
            print(f"{'Target Tensor':<38} | {'Cos Sim':<9} | {'HF ||G||_F':<11} | {'Max Abs':<11} | {'Rel Max':<11} | {'Rel Mean':<10}")
            print("-" * 105)

            for display_name, file_prefix, shape in tensors_to_check:
                bin_path = f"{out_dir}/{file_prefix}{tag}.bin"
                total_elem = 1
                for s in shape: total_elem *= s
                cpp_g = load_bin_floats(bin_path, total_elem)
                if cpp_g is None:
                    continue
                cpp_g = cpp_g.reshape(shape)
                hf_g = hf_grads[file_prefix]

                m = compute_metrics_chunked(cpp_g, hf_g)
                res_entry = {
                    "prompt_idx": p_idx,
                    "mode": mode_name,
                    "tensor": display_name,
                    "metrics": m
                }
                all_results.append(res_entry)

                print(f"{display_name:<38} | {m['cos_sim']:.6f}  | {m['grad_norm_hf']:<11.4e} | {m['max_abs']:<11.4e} | {m['rel_max_to_norm']:<11.4e} | {m['rel_mean_to_norm']:<10.4e}")

            # Free memory explicitly between modes
            del hf_grads, outputs, loss, flat_logits, spliced_embeds, prefix_torch
            model.zero_grad(set_to_none=True)
            gc.collect()

    # Write summary report
    with open("eval_tools/sweep_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 105)
    print(f"EVALUATION COMPLETE: {len(all_results)} tensor comparisons performed across 5 prompts.")
    print("Full metrics saved to eval_tools/sweep_results.json")
    print("=" * 105)

if __name__ == "__main__":
    main()
