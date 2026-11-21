#include "finetune_ops/core/tensor.h"
#include "finetune_ops/core/ops.h"
#include "finetune_ops/core/tokenizer.h"
#include "finetune_ops/graph/auto_model.h"

#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <cmath>

using namespace ops;

int main(int argc, char** argv) {
    std::string model_dir = "models/Qwen2.5-0.5B";
    if (argc > 1) {
        model_dir = argv[1];
    }

    std::cout << "========================================================================\n";
    std::cout << "  Qwen2.5-0.5B C++ Host Forward Pass & Parity Evaluator                 \n";
    std::cout << "========================================================================\n";
    std::cout << "Loading model from: " << model_dir << "\n";

    // 1. Load Tokenizer
    std::cout << "[1/4] Initializing TokenizerFactory...\n";
    auto tokenizer = TokenizerFactory::from_pretrained(model_dir);
    if (!tokenizer) {
        std::cerr << "Error: Failed to load tokenizer from " << model_dir << "\n";
        return 1;
    }
    std::cout << "      Tokenizer loaded. Vocab size: " << tokenizer->get_vocab_size() << "\n";

    // 2. Tokenize fixed 16-token text
    std::string text = "The quick brown fox jumps over the lazy dog. Artificial intelligence on mobile devices enables fast";
    std::vector<int> full_ids = tokenizer->encode(text);
    if (full_ids.size() < 16) {
        std::cerr << "Warning: text produced fewer than 16 tokens: " << full_ids.size() << "\n";
    }
    std::vector<int32_t> token_ids(full_ids.begin(), full_ids.begin() + std::min<size_t>(16, full_ids.size()));

    std::cout << "[2/4] Tokenized prompt to 16 tokens:\n      IDs: [";
    for (size_t i = 0; i < token_ids.size(); ++i) {
        std::cout << token_ids[i] << (i + 1 < token_ids.size() ? ", " : "]\n");
    }

    // 3. Load Model with real SafeTensors weights
    std::cout << "[3/4] Loading AutoModelForCausalLM (FP32 weights from safetensors)...\n";
    auto load_start = std::chrono::high_resolution_clock::now();
    AutoModelLoadOptions opts;
    opts.load_weights = true;
    opts.verbose = false;

    auto model = AutoModelForCausalLM::from_pretrained(model_dir, opts);
    if (!model) {
        std::cerr << "Error: Failed to create AutoModelForCausalLM\n";
        return 1;
    }
    auto load_end = std::chrono::high_resolution_clock::now();
    double load_sec = std::chrono::duration<double>(load_end - load_start).count();
    std::cout << "      Model weights loaded in " << std::fixed << std::setprecision(2) << load_sec << " seconds.\n";

    // 4. Construct input tensor [1, 16]
    auto input_tensor = std::make_shared<Tensor>(
        std::vector<int64_t>{1, static_cast<int64_t>(token_ids.size())},
        DType::kInt32,
        kCPU
    );
    int32_t* id_ptr = input_tensor->data<int32_t>();
    for (size_t i = 0; i < token_ids.size(); ++i) {
        id_ptr[i] = token_ids[i];
    }

    // 5. Measure forward pass
    std::cout << "[4/4] Running forward pass on fixed 16-token sequence...\n";
    auto fwd_start = std::chrono::high_resolution_clock::now();
    auto logits = model->forward(input_tensor);
    auto fwd_end = std::chrono::high_resolution_clock::now();
    double fwd_ms = std::chrono::duration<double, std::milli>(fwd_end - fwd_start).count();

    std::cout << "      >>> FORWARD PASS COMPLETE in " << std::fixed << std::setprecision(2) << fwd_ms << " ms ("
              << (fwd_ms / 1000.0) << " s) <<<\n";

    // 6. Inspect logits
    const auto& shape = logits->shape();
    std::cout << "      Logits shape: [";
    for (size_t i = 0; i < shape.size(); ++i) {
        std::cout << shape[i] << (i + 1 < shape.size() ? ", " : "]\n");
    }

    int64_t B = shape[0];
    int64_t S = shape[1];
    int64_t V = shape[2];
    const float* logit_data = logits->data<float>();

    float min_val = 1e9f, max_val = -1e9f;
    double sum_val = 0.0;
    int64_t total_numel = B * S * V;
    for (int64_t i = 0; i < total_numel; ++i) {
        float val = logit_data[i];
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
        sum_val += val;
    }
    std::cout << "      Logits Range: [" << min_val << ", " << max_val << "], Mean: "
              << (sum_val / total_numel) << "\n";

    // Inspect last token (pos 15) predictions
    int64_t last_token_offset = (S - 1) * V;
    std::vector<std::pair<float, int>> topk;
    for (int v = 0; v < V; ++v) {
        topk.push_back({logit_data[last_token_offset + v], v});
    }
    std::partial_sort(topk.begin(), topk.begin() + 5, topk.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });

    std::cout << "      Top-5 Predicted Next Tokens after position 15:\n";
    for (int i = 0; i < 5; ++i) {
        int tid = topk[i].second;
        float logit = topk[i].first;
        std::string tok_str = tokenizer->decode({tid});
        std::cout << "        #" << (i + 1) << " ID " << tid << " ('" << tok_str << "'): logit = " << logit << "\n";
    }

    // 7. Save binary logits
    std::string out_bin = model_dir + "/qwen_cpp_logits_fp32.bin";
    std::ofstream out(out_bin, std::ios::binary);
    if (out.is_open()) {
        int32_t b32 = static_cast<int32_t>(B);
        int32_t s32 = static_cast<int32_t>(S);
        int32_t v32 = static_cast<int32_t>(V);
        out.write(reinterpret_cast<const char*>(&b32), sizeof(b32));
        out.write(reinterpret_cast<const char*>(&s32), sizeof(s32));
        out.write(reinterpret_cast<const char*>(&v32), sizeof(v32));
        out.write(reinterpret_cast<const char*>(logit_data), total_numel * sizeof(float));
        out.close();
        std::cout << "      Saved raw logits to: " << out_bin << "\n";
    }

    // 8. Save token IDs binary
    std::string out_ids = model_dir + "/qwen_cpp_token_ids.bin";
    std::ofstream out_id(out_ids, std::ios::binary);
    if (out_id.is_open()) {
        int32_t s32 = static_cast<int32_t>(token_ids.size());
        out_id.write(reinterpret_cast<const char*>(&s32), sizeof(s32));
        out_id.write(reinterpret_cast<const char*>(token_ids.data()), token_ids.size() * sizeof(int32_t));
        out_id.close();
        std::cout << "      Saved token IDs to: " << out_ids << "\n";
    }

    std::cout << "========================================================================\n";
    std::cout << "SUCCESS: Host FP32 forward pass finished and exported.\n";
    return 0;
}
