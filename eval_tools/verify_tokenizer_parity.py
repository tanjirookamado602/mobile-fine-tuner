#!/usr/bin/env python3
"""
Verify token-by-token parity between C++ QwenBPETokenizer and Hugging Face AutoTokenizer
"""

import json
from transformers import AutoTokenizer

test_strings = [
    "The quick brown fox jumps over the lazy dog. Artificial intelligence on mobile devices enables fast",
    "移动端微调技术与深度学习模型部署",
    "機械学習とオンデバイス推論の検証",
    "الذكاء الاصطناعي على الأجهزة المحمولة",
    "Машинное обучение и оптимизация нейронных сетей",
    "Über die Brücke fährt ein déjà vu mit Café",
    "🚀 Mobile AI 🔥 2026 📱🤖✨",
    "int main() { std::vector<float> v = {1.0f, 2.0f}; return 0; }",
    "def forward(self, x: torch.Tensor) -> torch.Tensor:",
    "{\"name\": \"Qwen2.5\", \"params\": 0.5, \"active\": true}",
    "  Multiple   spaces    and\ttabs\nand newlines.\r\n",
    "pi = 3.141592653589793, e^(i*pi) + 1 = 0",
    "Supercalifragilisticexpialidocious and antidisestablishmentarianism",
    "They're couldn't shouldn't \"quoted\" and 'single' `backticks`",
    "https://huggingface.co/Qwen/Qwen2.5-0.5B/resolve/main/model.safetensors",
    "!@#$%^&*()_+-=[]{}|;':,./<>?",
    "helio_g88_armv8_2a_fp16_bf16_int8_q8_0",
    "یہ موبائل فائن ٹیوننگ کا نیا ماڈل ہے",
    "मोबाइल डिवाइस पर मशीन लर्निंग मॉडल का परीक्षण",
    "© 2026 Edge Intelligence Lab ™ • § ¶ † ‡",
    "aaaa bbbb cccc dddd aaaa bbbb cccc dddd"
]

def main():
    tokenizer = AutoTokenizer.from_pretrained("models/Qwen2.5-0.5B")
    
    with open("eval_tools/cpp_tokenized.json") as f:
        cpp_results = json.load(f)

    all_matched = True
    total_tokens = 0
    total_matches = 0

    print("=" * 80)
    print("TOKENIZER PARITY AUDIT: C++ QwenBPETokenizer vs Hugging Face AutoTokenizer")
    print("=" * 80)

    for i, text in enumerate(test_strings):
        hf_ids = tokenizer.encode(text, add_special_tokens=False)
        cpp_ids = cpp_results[i]["ids"]
        
        matches = (hf_ids == cpp_ids)
        if not matches:
            all_matched = False
        
        status = "PASS (100% Match)" if matches else "FAIL (Mismatch)"
        print(f"[{i+1:02d}/21] {status}: '{text[:45]}...'")
        print(f"      HF  len={len(hf_ids)}: {hf_ids[:10]}{'...' if len(hf_ids) > 10 else ''}")
        print(f"      C++ len={len(cpp_ids)}: {cpp_ids[:10]}{'...' if len(cpp_ids) > 10 else ''}")
        
        if not matches:
            print(f"      DIFFERENCE: HF={hf_ids} vs C++={cpp_ids}")
            
        total_tokens += len(hf_ids)
        total_matches += sum(1 for a, b in zip(hf_ids, cpp_ids) if a == b) if len(hf_ids) == len(cpp_ids) else 0

    print("=" * 80)
    if all_matched:
        print(f"ALL 21 TEST STRINGS PASSED WITH 100.0% EXACT TOKEN ID MATCH ({total_tokens}/{total_tokens} tokens)")
    else:
        print(f"TOKENIZER PARITY FAILED: {total_matches}/{total_tokens} tokens matched.")

if __name__ == "__main__":
    main()
