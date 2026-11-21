#include "finetune_ops/core/tensor.h"
#include "finetune_ops/core/ops.h"
#include "finetune_ops/core/tokenizer.h"
#include "finetune_ops/graph/auto_model.h"

#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <vector>
#include <random>
#include <cmath>
#include <cstring>

using namespace ops;

void write_bin(const std::string& path, const float* data, size_t count) {
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) {
        std::cerr << "Failed to open " << path << std::endl;
        return;
    }
    out.write(reinterpret_cast<const char*>(data), count * sizeof(float));
    out.close();
}

TensorPtr lookup_embeds(const TensorPtr& weight, const TensorPtr& indices) {
    auto shape = indices->shape();
    int64_t B = shape[0], S = shape[1], H = weight->shape()[1];
    auto out = zeros({B, S, H}, kFloat32, kCPU);
    const int32_t* idx = static_cast<const int32_t*>(indices->data_ptr());
    const float* w = static_cast<const float*>(weight->data_ptr());
    float* o = static_cast<float*>(out->data_ptr());
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t s = 0; s < S; ++s) {
            int32_t id = idx[b * S + s];
            if (id >= 0 && id < weight->shape()[0]) {
                std::memcpy(o + (b * S + s) * H, w + id * H, H * sizeof(float));
            }
        }
    }
    return out;
}

int main(int argc, char** argv) {
    std::string model_dir = "models/Qwen2.5-0.5B";
    std::cout << "========================================================================\n";
    std::cout << "  Qwen2.5-0.5B Parity, Gradient & 100-Token Benchmark Evaluator        \n";
    std::cout << "========================================================================\n";

    // 1. Load Model
    AutoModelLoadOptions opts;
    opts.load_weights = true;
    auto model = AutoModelForCausalLM::from_pretrained(model_dir, opts);
    if (!model) {
        std::cerr << "Error loading model\n";
        return 1;
    }
    auto qwen = model->qwen();

    // Enable gradients on target layers
    auto all_params = qwen->parameters();
    // params[1] is final_norm_weight_
    all_params[1]->set_requires_grad(true);
    // params[280] is layers[23].q_proj_weight
    all_params[280]->set_requires_grad(true);
    // params[289] is layers[23].down_proj_weight
    all_params[289]->set_requires_grad(true);

    // 2. Load Tokenizer
    auto tokenizer = TokenizerFactory::from_pretrained(model_dir);
    std::string text = "The quick brown fox jumps over the lazy dog. Artificial intelligence on mobile devices enables fast";
    std::vector<int> full_ids = tokenizer->encode(text);
    std::vector<int32_t> text_ids(full_ids.begin(), full_ids.begin() + 16);

    // 3. Generate deterministic fixed prefix embedding [1, 8, 896]
    int64_t P = 8;
    int64_t H = 896;
    auto prefix_tensor = zeros({1, P, H}, kFloat32, kCPU);
    prefix_tensor->set_requires_grad(true);
    float* p_data = static_cast<float*>(prefix_tensor->data_ptr());

    std::mt19937 rng(12345);
    std::normal_distribution<float> dist(0.0f, 0.05f);
    for (int64_t i = 0; i < P * H; ++i) {
        p_data[i] = dist(rng);
    }
    // Export binary prefix for PyTorch to load identical values
    write_bin("eval_tools/fixed_prefix_8x896.bin", p_data, P * H);

    // 4. Get text embeddings [1, 16, 896]
    auto text_id_tensor = std::make_shared<Tensor>(
        std::vector<int64_t>{1, static_cast<int64_t>(text_ids.size())},
        DType::kInt32, kCPU
    );
    int32_t* id_ptr = static_cast<int32_t*>(text_id_tensor->data_ptr());
    for (size_t i = 0; i < text_ids.size(); ++i) {
        id_ptr[i] = text_ids[i];
    }
    auto text_embeds = lookup_embeds(qwen->embedding_weight(), text_id_tensor);

    // 5. Concatenate prefix [1, 8, 896] and text [1, 16, 896] -> spliced [1, 24, 896]
    auto spliced_embeds = concat(std::vector<TensorPtr>{prefix_tensor, text_embeds}, 1);

    // 6. Forward pass through forward_hidden_from_embeds
    std::cout << "[Step 2] Executing forward_hidden_from_embeds on spliced [1, 24, 896]...\n";
    auto hidden = qwen->forward_hidden_from_embeds(spliced_embeds);
    auto logits = qwen->lm_head(hidden); // [1, 24, 151936]

    // Save spliced logits
    const float* logit_ptr = static_cast<const float*>(logits->data_ptr());
    write_bin("eval_tools/cpp_spliced_logits_fp32.bin", logit_ptr, 1 * 24 * 151936);

    // 7. Compute Cross-Entropy Loss with first 8 tokens masked as -100
    int64_t pred_seq = 23;
    auto sliced_logits = logits->slice(1, 0, pred_seq, 1); // [1, 23, 151936]
    auto flat_logits = reshape(sliced_logits, {pred_seq, 151936});

    auto target_tensor = zeros({pred_seq}, DType::kInt32, kCPU);
    int32_t* tgt_ptr = static_cast<int32_t*>(target_tensor->data_ptr());
    for (int i = 0; i < 7; ++i) {
        tgt_ptr[i] = -100; // masked prefix
    }
    tgt_ptr[7] = text_ids[0]; // predicts first text token
    for (size_t i = 1; i < text_ids.size(); ++i) {
        tgt_ptr[7 + i] = text_ids[i];
    }

    // Save targets for PyTorch
    std::ofstream tgt_out("eval_tools/targets_23.bin", std::ios::binary);
    tgt_out.write(reinterpret_cast<const char*>(tgt_ptr), pred_seq * sizeof(int32_t));
    tgt_out.close();

    auto loss = cross_entropy_loss(flat_logits, target_tensor, "mean");
    float loss_val = static_cast<const float*>(loss->data_ptr())[0];
    std::cout << "      C++ Cross-Entropy Loss (masked prefix): " << std::fixed << std::setprecision(6) << loss_val << "\n";

    // 8. Backward pass (routes gradients through ConcatBackward and all 24 layers!)
    std::cout << "[Step 3] Running loss->backward() ...\n";
    auto bwd_start = std::chrono::high_resolution_clock::now();
    loss->backward();
    auto bwd_end = std::chrono::high_resolution_clock::now();
    double bwd_ms = std::chrono::duration<double, std::milli>(bwd_end - bwd_start).count();
    std::cout << "      Backward pass complete in " << std::fixed << std::setprecision(2) << bwd_ms << " ms.\n";

    // Export Gradients
    if (prefix_tensor->grad()) {
        const float* g_prefix = static_cast<const float*>(prefix_tensor->grad()->data_ptr());
        write_bin("eval_tools/cpp_grad_prefix.bin", g_prefix, P * H);
        std::cout << "      Exported prefix grad [1, 8, 896].\n";
    } else {
        std::cerr << "ERROR: prefix_tensor->grad() is null!\n";
    }

    // params[280]: layers[23].q_proj_weight [896, 896]
    if (all_params[280]->grad()) {
        const float* g_qproj = static_cast<const float*>(all_params[280]->grad()->data_ptr());
        write_bin("eval_tools/cpp_grad_qproj23.bin", g_qproj, 896 * 896);
        std::cout << "      Exported layers[23].q_proj grad [896, 896].\n";
    }

    // params[289]: layers[23].down_proj_weight [4864, 896]
    if (all_params[289]->grad()) {
        const float* g_down = static_cast<const float*>(all_params[289]->grad()->data_ptr());
        write_bin("eval_tools/cpp_grad_downproj23.bin", g_down, 4864 * 896);
        std::cout << "      Exported layers[23].down_proj grad [4864, 896].\n";
    }

    // params[1]: final_norm_weight [896]
    if (all_params[1]->grad()) {
        const float* g_norm = static_cast<const float*>(all_params[1]->grad()->data_ptr());
        write_bin("eval_tools/cpp_grad_norm.bin", g_norm, 896);
        std::cout << "      Exported final_norm grad [896].\n";
    }

    // 9. Benchmark: 100-token forward + backward step (Task 5)
    std::cout << "\n[Step 5] Benchmarking 100-token full training step (64 image + 36 text)...\n";
    int64_t P100 = 64;
    int64_t T100 = 36;
    auto prefix100 = zeros({1, P100, H}, kFloat32, kCPU);
    prefix100->set_requires_grad(true);

    auto text_ids100 = zeros({1, T100}, DType::kInt32, kCPU);
    int32_t* tid100_ptr = static_cast<int32_t*>(text_ids100->data_ptr());
    for (int i = 0; i < T100; ++i) {
        tid100_ptr[i] = (i < 16) ? text_ids[i] : 100 + i;
    }
    auto embeds100 = lookup_embeds(qwen->embedding_weight(), text_ids100);
    auto spliced100 = concat(std::vector<TensorPtr>{prefix100, embeds100}, 1);

    // Forward
    auto t0 = std::chrono::high_resolution_clock::now();
    auto h100 = qwen->forward_hidden_from_embeds(spliced100);
    auto l100 = qwen->lm_head(h100);
    auto t1 = std::chrono::high_resolution_clock::now();
    double fwd_100_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Simple loss
    auto flat_l100 = reshape(l100->slice(1, 0, 99, 1), {99, 151936});
    auto targets100 = zeros({99}, DType::kInt32, kCPU);
    int32_t* t100_ptr = static_cast<int32_t*>(targets100->data_ptr());
    for (int i = 0; i < 99; ++i) t100_ptr[i] = (i < 63) ? -100 : 100 + (i - 63);
    auto loss100 = cross_entropy_loss(flat_l100, targets100, "mean");

    // Backward
    auto t2 = std::chrono::high_resolution_clock::now();
    loss100->backward();
    auto t3 = std::chrono::high_resolution_clock::now();
    double bwd_100_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
    double total_100_ms = fwd_100_ms + bwd_100_ms;

    std::cout << "      >>> 100-TOKEN BENCHMARK RESULTS <<<\n";
    std::cout << "      Forward Pass  (100 tokens): " << std::fixed << std::setprecision(2) << fwd_100_ms << " ms (" << fwd_100_ms / 1000.0 << " s)\n";
    std::cout << "      Backward Pass (100 tokens): " << std::fixed << std::setprecision(2) << bwd_100_ms << " ms (" << bwd_100_ms / 1000.0 << " s)\n";
    std::cout << "      Total Step    (100 tokens): " << std::fixed << std::setprecision(2) << total_100_ms << " ms (" << total_100_ms / 1000.0 << " s)\n";

    std::cout << "========================================================================\n";
    std::cout << "SUCCESS: C++ forward, backward, gradient export, and 100-token benchmark finished.\n";
    return 0;
}
