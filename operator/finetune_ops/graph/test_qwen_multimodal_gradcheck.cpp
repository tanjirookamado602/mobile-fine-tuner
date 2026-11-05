#include "qwen_multimodal.h"
#include "../core/ops.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <random>
#include <vector>

using namespace ops;

namespace {

void require(bool ok, const char* message) {
    if (!ok) {
        std::cerr << "FAIL: " << message << std::endl;
        std::exit(1);
    }
}

void test_projector_numerical_gradient_check() {
    std::cout << "==========================================================" << std::endl;
    std::cout << "Running Qwen Projector Numerical vs Analytical Grad Check" << std::endl;
    std::cout << "==========================================================" << std::endl;

    // 1. Build test config
    QwenMultimodalConfig mm_cfg;
    mm_cfg.text_config.vocab_size = 64;
    mm_cfg.text_config.hidden_size = 32;
    mm_cfg.text_config.intermediate_size = 64;
    mm_cfg.text_config.num_hidden_layers = 1;
    mm_cfg.text_config.num_attention_heads = 2;
    mm_cfg.text_config.num_key_value_heads = 1;
    mm_cfg.text_config.max_position_embeddings = 256;
    mm_cfg.text_config.rms_norm_eps = 1e-6f;
    mm_cfg.vision_dim = 16;
    mm_cfg.use_projector_bias = false;
    mm_cfg.seed = 1234;

    QwenMultimodalModel model(mm_cfg);

    // Initialize base weights deterministically
    for (auto& p : model.base_model()->parameters()) {
        if (!p) continue;
        float* d = p->data<float>();
        for (int64_t i = 0; i < p->numel(); ++i) {
            d[i] = ((i % 11) - 5.0f) * 0.02f;
        }
    }
    model.base_model()->assign_weight("final_norm.weight", full({mm_cfg.text_config.hidden_size}, 1.0f));
    for (int l = 0; l < mm_cfg.text_config.num_hidden_layers; ++l) {
        model.base_model()->assign_weight("layers." + std::to_string(l) + ".input_norm.weight", full({mm_cfg.text_config.hidden_size}, 1.0f));
        model.base_model()->assign_weight("layers." + std::to_string(l) + ".post_norm.weight", full({mm_cfg.text_config.hidden_size}, 1.0f));
    }
    model.freeze_base();

    const int64_t B = 1;
    const int64_t V_len = 2;
    const int64_t T_len = 2;
    const int64_t total_len = V_len + T_len;

    auto vision_feats = zeros({B, V_len, mm_cfg.vision_dim}, kFloat32, kCPU);
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 0.05f);
    for (int64_t i = 0; i < vision_feats->numel(); ++i) {
        vision_feats->data<float>()[i] = dist(gen);
    }

    auto input_ids = zeros({B, T_len}, kInt32, kCPU);
    for (int64_t i = 0; i < T_len; ++i) {
        input_ids->data<int32_t>()[i] = static_cast<int32_t>((i * 7 + 3) % mm_cfg.text_config.vocab_size);
    }

    auto attention_mask = full({B, T_len}, 1.0f, kFloat32, kCPU);

    auto targets = zeros({B, total_len}, kInt32, kCPU);
    for (int64_t i = 0; i < total_len; ++i) {
        targets->data<int32_t>()[i] = (i < V_len) ? -100 : static_cast<int32_t>((i * 5 + 7) % mm_cfg.text_config.vocab_size);
    }

    auto compute_loss = [&]() -> float {
        auto logits = model.forward_multimodal(vision_feats, input_ids, attention_mask);
        auto flat_logits = ops::reshape(logits, {total_len, mm_cfg.text_config.vocab_size});
        auto flat_targets = ops::reshape(targets, {total_len});
        auto loss = ops::cross_entropy_loss(flat_logits, flat_targets);
        return loss->item();
    };

    // 1. Analytical backward pass
    auto initial_logits = model.forward_multimodal(vision_feats, input_ids, attention_mask);
    auto flat_initial_logits = ops::reshape(initial_logits, {total_len, mm_cfg.text_config.vocab_size});
    auto flat_initial_targets = ops::reshape(targets, {total_len});
    auto analytical_loss = ops::cross_entropy_loss(flat_initial_logits, flat_initial_targets);
    analytical_loss->backward();

    require(model.projector_weight()->grad() != nullptr, "analytical gradient exists");
    auto analytical_grads = model.projector_weight()->grad()->clone();

    // 2. Numerical finite-difference check on representative elements
    const float eps = 1e-3f;
    float* w_data = model.projector_weight()->data<float>();
    const float* a_grad_data = analytical_grads->data<float>();
    int64_t total_elements = model.projector_weight()->numel(); // 32 * 16 = 512

    std::vector<int64_t> test_indices = {
        0, 1, 15, 16, 31, 64, 127, 200, 255, 300, 384, 500, 511
    };

    std::cout << std::left << std::setw(8) << "Index"
              << std::setw(16) << "Analytical"
              << std::setw(16) << "Numerical"
              << std::setw(16) << "Rel Error" << std::endl;
    std::cout << "--------------------------------------------------------" << std::endl;

    float max_rel_err = 0.0f;
    for (int64_t idx : test_indices) {
        if (idx >= total_elements) continue;

        float original_val = w_data[idx];
        float analytical_g = a_grad_data[idx];

        // L(w + eps)
        w_data[idx] = original_val + eps;
        float loss_plus = compute_loss();

        // L(w - eps)
        w_data[idx] = original_val - eps;
        float loss_minus = compute_loss();

        // Reset
        w_data[idx] = original_val;

        float numerical_g = (loss_plus - loss_minus) / (2.0f * eps);
        float diff = std::abs(analytical_g - numerical_g);
        float denom = std::max({std::abs(analytical_g), std::abs(numerical_g), 1e-4f});
        float rel_err = diff / denom;

        if (rel_err > max_rel_err) {
            max_rel_err = rel_err;
        }

        std::cout << std::left << std::setw(8) << idx
                  << std::setw(16) << analytical_g
                  << std::setw(16) << numerical_g
                  << std::setw(16) << rel_err << std::endl;

        require(rel_err < 0.02f, "Element relative error < 0.02 (2%)");
    }

    std::cout << "--------------------------------------------------------" << std::endl;
    std::cout << "Max Relative Error: " << max_rel_err << std::endl;
    require(max_rel_err < 0.02f, "Max relative error < 0.02 (2%)");
    std::cout << "[SUCCESS] Numerical vs Analytical Gradient check passed!" << std::endl;
}

}  // namespace

int main() {
    test_projector_numerical_gradient_check();
    return 0;
}
