#include "qwen_multimodal.h"
#include "../core/ops.h"
#include <iostream>
#include <cmath>
#include <random>

using namespace ops;

namespace {

void require(bool ok, const char* message) {
    if (!ok) {
        std::cerr << "FAIL: " << message << std::endl;
        std::exit(1);
    }
}

void test_qwen_multimodal_model_end_to_end() {
    std::cout << "==========================================================" << std::endl;
    std::cout << "Running QwenMultimodalModel Class Test" << std::endl;
    std::cout << "==========================================================" << std::endl;

    // 1. Build test config
    QwenMultimodalConfig mm_cfg;
    mm_cfg.text_config.vocab_size = 128;
    mm_cfg.text_config.hidden_size = 64;
    mm_cfg.text_config.intermediate_size = 128;
    mm_cfg.text_config.num_hidden_layers = 2;
    mm_cfg.text_config.num_attention_heads = 4;
    mm_cfg.text_config.num_key_value_heads = 2;
    mm_cfg.text_config.max_position_embeddings = 512;
    mm_cfg.text_config.rms_norm_eps = 1e-6f;
    mm_cfg.vision_dim = 32;
    mm_cfg.use_projector_bias = false;
    mm_cfg.seed = 42;

    std::cout << "--- Step 1: Initialize QwenMultimodalModel ---" << std::endl;
    QwenMultimodalModel model(mm_cfg);

    // Initialize base weights deterministically
    for (auto& p : model.base_model()->parameters()) {
        if (!p) continue;
        float* d = p->data<float>();
        for (int64_t i = 0; i < p->numel(); ++i) {
            d[i] = ((i % 17) - 8.0f) * 0.02f;
        }
    }
    model.base_model()->assign_weight("final_norm.weight", full({mm_cfg.text_config.hidden_size}, 1.0f));
    for (int l = 0; l < mm_cfg.text_config.num_hidden_layers; ++l) {
        model.base_model()->assign_weight("layers." + std::to_string(l) + ".input_norm.weight", full({mm_cfg.text_config.hidden_size}, 1.0f));
        model.base_model()->assign_weight("layers." + std::to_string(l) + ".post_norm.weight", full({mm_cfg.text_config.hidden_size}, 1.0f));
    }

    std::cout << "--- Step 2: Freeze Base Backbone ---" << std::endl;
    model.freeze_base();

    auto trainables = model.trainable_parameters();
    require(!trainables.empty(), "trainable parameters not empty");
    require(trainables.size() == 1, "exactly 1 trainable parameter (projector weight)");
    require(trainables[0] == model.projector_weight(), "trainable parameter is projector weight");
    require(model.projector_weight()->shape() == std::vector<int64_t>({64, 32}), "projector shape [64, 32]");

    std::cout << "--- Step 3: Forward Multimodal Pass ---" << std::endl;
    const int64_t B = 1;
    const int64_t V_len = 4;
    const int64_t T_len = 4;
    const int64_t total_len = V_len + T_len;

    auto vision_feats = zeros({B, V_len, mm_cfg.vision_dim}, kFloat32, kCPU);
    std::mt19937 gen(100);
    std::normal_distribution<float> dist(0.0f, 0.02f);
    for (int64_t i = 0; i < vision_feats->numel(); ++i) {
        vision_feats->data<float>()[i] = dist(gen);
    }

    auto input_ids = zeros({B, T_len}, kInt32, kCPU);
    for (int64_t i = 0; i < T_len; ++i) {
        input_ids->data<int32_t>()[i] = static_cast<int32_t>((i * 13 + 5) % mm_cfg.text_config.vocab_size);
    }

    auto attention_mask = full({B, T_len}, 1.0f, kFloat32, kCPU);

    auto logits = model.forward_multimodal(vision_feats, input_ids, attention_mask);
    require(logits != nullptr, "logits not null");
    require(logits->shape() == std::vector<int64_t>({B, total_len, mm_cfg.text_config.vocab_size}), "logits shape [1, 8, 128]");

    std::cout << "--- Step 4: Backward Pass ---" << std::endl;
    auto targets = zeros({B, total_len}, kInt32, kCPU);
    for (int64_t i = 0; i < total_len; ++i) {
        targets->data<int32_t>()[i] = (i < V_len) ? -100 : static_cast<int32_t>((i * 11 + 3) % mm_cfg.text_config.vocab_size);
    }

    auto flat_logits = ops::reshape(logits, {total_len, mm_cfg.text_config.vocab_size});
    auto flat_targets = ops::reshape(targets, {total_len});
    auto loss = ops::cross_entropy_loss(flat_logits, flat_targets);
    float loss_val = loss->item();
    std::cout << "Initial Loss: " << loss_val << std::endl;
    require(std::isfinite(loss_val) && loss_val > 0.0f, "initial loss valid");

    loss->backward();

    require(model.projector_weight()->grad() != nullptr, "projector gradient exists");
    require(model.projector_weight()->grad()->shape() == std::vector<int64_t>({64, 32}), "projector gradient shape [64, 32]");

    float grad_norm_sq = 0.0f;
    const float* g_data = model.projector_weight()->grad()->data<float>();
    for (int64_t i = 0; i < model.projector_weight()->numel(); ++i) {
        require(std::isfinite(g_data[i]), "gradient value is finite");
        grad_norm_sq += g_data[i] * g_data[i];
    }
    float grad_norm = std::sqrt(grad_norm_sq);
    std::cout << "Projector Gradient Norm: " << grad_norm << std::endl;
    require(grad_norm > 1e-4f, "gradient norm is non-zero");

    std::cout << "[SUCCESS] QwenMultimodalModel verified successfully!" << std::endl;
}

}  // namespace

int main() {
    test_qwen_multimodal_model_end_to_end();
    return 0;
}

