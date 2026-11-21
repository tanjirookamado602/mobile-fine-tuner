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
#include <sys/stat.h>
#include <malloc.h>

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

void write_bin_i32(const std::string& path, const int32_t* data, size_t count) {
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) {
        std::cerr << "Failed to open " << path << std::endl;
        return;
    }
    out.write(reinterpret_cast<const char*>(data), count * sizeof(int32_t));
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

void reset_all_grads(const std::vector<TensorPtr>& params, const TensorPtr& prefix) {
    if (prefix && prefix->grad()) {
        prefix->set_grad(nullptr);
    }
    for (auto& p : params) {
        if (p && p->grad()) {
            p->set_grad(nullptr);
        }
    }
}

int main(int argc, char** argv) {
    std::string model_dir = "models/Qwen2.5-0.5B";
    std::string out_dir = "eval_tools/sweep_out";
    mkdir(out_dir.c_str(), 0777);

    std::cout << "========================================================================\n";
    std::cout << "  Comprehensive Multi-Prompt Multi-Layer Gradient Parity Sweep          \n";
    std::cout << "  Host: Intel(R) Core(TM) i5-7200U CPU @ 2.50GHz                        \n";
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
    auto all_params = qwen->parameters();

    // Target parameters to track
    // params[0]: embed_tokens_ [151936, 896]
    // params[1]: final_norm_weight_ [896]
    // params[4]: layer 0 q_proj [896, 896]
    // params[13]: layer 0 down_proj [4864, 896]
    // params[148]: layer 12 q_proj [896, 896]
    // params[157]: layer 12 down_proj [4864, 896]
    // params[280]: layer 23 q_proj [896, 896]
    // params[289]: layer 23 down_proj [4864, 896]

    // Disable gradients on all parameters first
    for (auto& p : all_params) {
        if (p) p->set_requires_grad(false);
    }
    // Enable on target tensors
    all_params[0]->set_requires_grad(true);   // embed_tokens_ & lm_head
    all_params[1]->set_requires_grad(true);   // final_norm
    all_params[4]->set_requires_grad(true);   // l0 q_proj
    all_params[13]->set_requires_grad(true);  // l0 down_proj
    all_params[148]->set_requires_grad(true); // l12 q_proj
    all_params[157]->set_requires_grad(true); // l12 down_proj
    all_params[280]->set_requires_grad(true); // l23 q_proj
    all_params[289]->set_requires_grad(true); // l23 down_proj

    // 2. Load Tokenizer
    auto tokenizer = TokenizerFactory::from_pretrained(model_dir);

    // 5 diverse prompts
    std::vector<std::string> prompts = {
        "The quick brown fox jumps over the lazy dog. Artificial intelligence on mobile devices enables fast",
        "Deep learning models require careful optimization of memory bandwidth and floating point computation.",
        "In computer science, gradient descent is a first-order iterative optimization algorithm for finding a local minimum.",
        "Translate and summarize: On-device fine-tuning empowers personalized user experiences with privacy.",
        "def train_step(model, inputs, targets):\n    logits = model(inputs)\n    loss = cross_entropy(logits, targets)"
    };

    int64_t P = 8;
    int64_t H = 896;
    int64_t T = 16;
    int64_t total_tokens = P + T; // 24 tokens

    for (size_t p_idx = 0; p_idx < prompts.size(); ++p_idx) {
        std::cout << "\n------------------------------------------------------------------------\n";
        std::cout << "[Prompt " << (p_idx + 1) << "/5]: \"" << prompts[p_idx].substr(0, 50) << "...\"\n";
        std::cout << "------------------------------------------------------------------------\n";

        // Tokenize 16 tokens
        std::vector<int> full_ids = tokenizer->encode(prompts[p_idx]);
        std::vector<int32_t> text_ids;
        for (size_t i = 0; i < (size_t)T; ++i) {
            text_ids.push_back(i < full_ids.size() ? full_ids[i] : 100 + i);
        }

        // Generate deterministic prefix [1, P, H]
        auto prefix_tensor = zeros({1, P, H}, kFloat32, kCPU);
        prefix_tensor->set_requires_grad(true);
        float* p_data = static_cast<float*>(prefix_tensor->data_ptr());
        std::mt19937 rng(1000 + p_idx);
        std::normal_distribution<float> dist(0.0f, 0.05f);
        for (int64_t i = 0; i < P * H; ++i) {
            p_data[i] = dist(rng);
        }

        // Export prefix and text_ids
        std::string pfx_path = out_dir + "/prefix_p" + std::to_string(p_idx) + ".bin";
        std::string tid_path = out_dir + "/text_ids_p" + std::to_string(p_idx) + ".bin";
        write_bin(pfx_path, p_data, P * H);
        write_bin_i32(tid_path, text_ids.data(), T);

        // Lookup text embeds
        auto text_id_tensor = std::make_shared<Tensor>(
            std::vector<int64_t>{1, T}, DType::kInt32, kCPU
        );
        std::memcpy(text_id_tensor->data_ptr(), text_ids.data(), T * sizeof(int32_t));
        auto text_embeds = lookup_embeds(qwen->embedding_weight(), text_id_tensor);

        // Concatenate spliced [1, 24, 896]
        auto spliced_embeds = concat(std::vector<TensorPtr>{prefix_tensor, text_embeds}, 1);

        // Forward pass
        auto hidden = qwen->forward_hidden_from_embeds(spliced_embeds);
        auto logits = qwen->lm_head(hidden); // [1, 24, 151936]

        int64_t pred_seq = total_tokens - 1; // 23
        auto sliced_logits = logits->slice(1, 0, pred_seq, 1);
        auto flat_logits = reshape(sliced_logits, {pred_seq, 151936});

        // Test both: Mode 0 = Masked, Mode 1 = Unmasked
        for (int mode = 0; mode < 2; ++mode) {
            bool masked = (mode == 0);
            std::string mode_str = masked ? "Masked Loss (prefix -100)" : "Unmasked Loss (all tokens)";
            std::cout << "  Evaluating Mode: " << mode_str << " ...\n";

            auto target_tensor = zeros({pred_seq}, DType::kInt32, kCPU);
            int32_t* tgt_ptr = static_cast<int32_t*>(target_tensor->data_ptr());
            for (int i = 0; i < pred_seq; ++i) {
                if (masked && i < P - 1) {
                    tgt_ptr[i] = -100;
                } else if (i < P - 1) {
                    tgt_ptr[i] = 500 + i; // unmasked prefix tokens
                } else {
                    tgt_ptr[i] = text_ids[i - (P - 1)];
                }
            }

            // Export targets
            std::string tgt_path = out_dir + "/targets_p" + std::to_string(p_idx) + "_m" + std::to_string(mode) + ".bin";
            write_bin_i32(tgt_path, tgt_ptr, pred_seq);

            // Compute loss
            auto loss = cross_entropy_loss(flat_logits, target_tensor, "mean");
            float loss_val = static_cast<const float*>(loss->data_ptr())[0];
            std::cout << "    Loss: " << std::fixed << std::setprecision(6) << loss_val << "\n";

            // Backward
            reset_all_grads(all_params, prefix_tensor);
            loss->backward();

            // Export gradients
            std::string tag = "_p" + std::to_string(p_idx) + "_m" + std::to_string(mode);

            // 1. Prefix [1, 8, 896]
            if (prefix_tensor->grad()) {
                write_bin(out_dir + "/grad_prefix" + tag + ".bin", 
                          static_cast<const float*>(prefix_tensor->grad()->data_ptr()), P * H);
            }
            // 2. Embed tokens / lm_head [151936, 896]
            if (all_params[0]->grad()) {
                write_bin(out_dir + "/grad_embed" + tag + ".bin",
                          static_cast<const float*>(all_params[0]->grad()->data_ptr()), 151936 * 896);
            }
            // 3. Final norm [896]
            if (all_params[1]->grad()) {
                write_bin(out_dir + "/grad_norm" + tag + ".bin",
                          static_cast<const float*>(all_params[1]->grad()->data_ptr()), 896);
            }
            // 4. Layer 0 Q-Proj [896, 896]
            if (all_params[4]->grad()) {
                write_bin(out_dir + "/grad_l0_qproj" + tag + ".bin",
                          static_cast<const float*>(all_params[4]->grad()->data_ptr()), 896 * 896);
            }
            // 5. Layer 0 Down-Proj [4864, 896]
            if (all_params[13]->grad()) {
                write_bin(out_dir + "/grad_l0_downproj" + tag + ".bin",
                          static_cast<const float*>(all_params[13]->grad()->data_ptr()), 4864 * 896);
            }
            // 6. Layer 12 Q-Proj [896, 896]
            if (all_params[148]->grad()) {
                write_bin(out_dir + "/grad_l12_qproj" + tag + ".bin",
                          static_cast<const float*>(all_params[148]->grad()->data_ptr()), 896 * 896);
            }
            // 7. Layer 12 Down-Proj [4864, 896]
            if (all_params[157]->grad()) {
                write_bin(out_dir + "/grad_l12_downproj" + tag + ".bin",
                          static_cast<const float*>(all_params[157]->grad()->data_ptr()), 4864 * 896);
            }
            // 8. Layer 23 Q-Proj [896, 896]
            if (all_params[280]->grad()) {
                write_bin(out_dir + "/grad_l23_qproj" + tag + ".bin",
                          static_cast<const float*>(all_params[280]->grad()->data_ptr()), 896 * 896);
            }
            // 9. Layer 23 Down-Proj [4864, 896]
            if (all_params[289]->grad()) {
                write_bin(out_dir + "/grad_l23_downproj" + tag + ".bin",
                          static_cast<const float*>(all_params[289]->grad()->data_ptr()), 4864 * 896);
            }
            reset_all_grads(all_params, prefix_tensor);
            malloc_trim(0);
            std::cout << "    Gradients exported for mode " << mode << ".\n";
        }
    }

    std::cout << "\n========================================================================\n";
    std::cout << "SUCCESS: All 5 prompts evaluated across both masked & unmasked losses.\n";
    std::cout << "========================================================================\n";
    return 0;
}
