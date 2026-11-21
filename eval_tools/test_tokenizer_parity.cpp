#include "finetune_ops/core/tokenizer.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

using namespace ops;

int main() {
    std::string model_dir = "models/Qwen2.5-0.5B";
    auto tokenizer = TokenizerFactory::from_pretrained(model_dir);
    if (!tokenizer) {
        std::cerr << "Failed to load tokenizer from " << model_dir << std::endl;
        return 1;
    }

    std::vector<std::string> test_strings = {
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
    };

    std::ofstream out("eval_tools/cpp_tokenized.json");
    out << "[\n";
    for (size_t s = 0; s < test_strings.size(); ++s) {
        auto ids = tokenizer->encode(test_strings[s]);
        out << "  {\n";
        out << "    \"index\": " << s << ",\n";
        out << "    \"ids\": [";
        for (size_t i = 0; i < ids.size(); ++i) {
            out << ids[i] << (i + 1 < ids.size() ? ", " : "");
        }
        out << "]\n";
        out << "  }" << (s + 1 < test_strings.size() ? "," : "") << "\n";
    }
    out << "]\n";
    out.close();

    std::cout << "Successfully tokenized " << test_strings.size() << " test strings with C++ tokenizer." << std::endl;
    return 0;
}
