#!/usr/bin/env python3
"""
Evaluate parity between MobileFineTuner C++ logits and Hugging Face PyTorch logits
"""

import sys
import struct
import numpy as np

def load_cpp_logits(path):
    with open(path, "rb") as f:
        B, S, V = struct.unpack("<iii", f.read(12))
        total_floats = B * S * V
        data = np.frombuffer(f.read(total_floats * 4), dtype=np.float32)
        return data.reshape(B, S, V)

def main():
    cpp_path = sys.argv[1] if len(sys.argv) > 1 else "models/Qwen2.5-0.5B/qwen_cpp_logits_fp32.bin"
    hf_path = sys.argv[2] if len(sys.argv) > 2 else "qwen_hf_reference_outputs/qwen_ref_logits_fp32.npy"

    print("=" * 70)
    print("MobileFineTuner C++ vs Hugging Face PyTorch Parity Comparison")
    print("=" * 70)
    print(f"C++ logits file: {cpp_path}")
    print(f"HF  logits file: {hf_path}")

    cpp_logits = load_cpp_logits(cpp_path)
    hf_logits = np.load(hf_path)

    print(f"C++ shape: {cpp_logits.shape}, dtype: {cpp_logits.dtype}")
    print(f"HF  shape: {hf_logits.shape}, dtype: {hf_logits.dtype}")

    assert cpp_logits.shape == hf_logits.shape, f"Shape mismatch: {cpp_logits.shape} vs {hf_logits.shape}"

    diff = np.abs(cpp_logits - hf_logits)
    max_abs = np.max(diff)
    mean_abs = np.mean(diff)

    denom = np.abs(hf_logits) + 1e-7
    rel_diff = diff / denom
    max_rel = np.max(rel_diff)
    mean_rel = np.mean(rel_diff)

    print("\nNumerical Error Metrics across all positions and vocab:")
    print(f"  Max Absolute Difference: {max_abs:.6e}")
    print(f"  Mean Absolute Error:     {mean_abs:.6e}")
    print(f"  Max Relative Difference: {max_rel:.6e}")
    print(f"  Mean Relative Error:     {mean_rel:.6e}")

    # Top-1 agreement per position
    B, S, V = cpp_logits.shape
    cpp_top1 = np.argmax(cpp_logits, axis=-1)[0]
    hf_top1 = np.argmax(hf_logits, axis=-1)[0]
    matches = (cpp_top1 == hf_top1)
    acc = np.mean(matches) * 100.0

    print(f"\nTop-1 Next-Token Agreement across {S} positions:")
    print(f"  Agreement: {np.sum(matches)}/{S} ({acc:.1f}%)")

    print("\nPer-Position Breakdown:")
    for pos in range(S):
        c_tok = int(cpp_top1[pos])
        h_tok = int(hf_top1[pos])
        c_val = float(cpp_logits[0, pos, c_tok])
        h_val = float(hf_logits[0, pos, h_tok])
        match_str = "MATCH" if c_tok == h_tok else "MISMATCH"
        print(f"  Pos {pos:02d}: {match_str} | C++ top1 ID {c_tok} (val={c_val:.2f}) vs HF top1 ID {h_tok} (val={h_val:.2f})")

    if max_abs < 1e-3 and acc == 100.0:
        print("\n>>> VERDICT: NUMERICAL PARITY CONFIRMED (within FP32 tolerance) <<<")
    else:
        print(f"\n>>> VERDICT: Discrepancy observed (max_abs={max_abs:.4f}, agreement={acc:.1f}%) <<<")

if __name__ == "__main__":
    main()
