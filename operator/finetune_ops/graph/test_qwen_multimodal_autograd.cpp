#include "qwen_model.h"
#include "../core/ops.h"
#include "../core/tensor.h"
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <random>

using namespace ops;

namespace {

void require(bool ok, const char* message) {
    if (!ok) {
        std::cerr << "FAIL: " << message << std::endl;
        std::exit(1);
    }
}

QwenConfig make_test_config() {
    QwenConfig cfg;
    cfg.vocab_size = 128;
    cfg.hidden_size = 64;
    cfg.intermediate_size = 128;
    cfg.num_hidden_layers = 2;
    cfg.num_attention_heads = 4;
    cfg.num_key_value_heads = 2;
    cfg.max_position_embeddings = 512;
    cfg.rms_norm_eps = 1e-6f;
    return cfg;
}

void test_multimodal_forward_backward_convergence() {
    std::cout << "--- Step 1: Initialize Qwen Model ---" << std::endl;
    auto cfg = make_test_config();
    QwenModel model(cfg);
    
    // Initialize model parameters with non-zero weights
    for (auto& p : model.parameters()) {
        if (!p) continue;
        float* d = p->data<float>();
        for (int64_t i = 0; i < p->numel(); ++i) {
            d[i] = ((i % 17) - 8.0f) * 0.02f;
        }
    }
    // Assign 1.0 to final_norm and input/post norms
    model.assign_weight("final_norm.weight", full({cfg.hidden_size}, 1.0f));
    for (int l = 0; l < cfg.num_hidden_layers; ++l) {
        model.assign_weight("layers." + std::to_string(l) + ".input_norm.weight", full({cfg.hidden_size}, 1.0f));
        model.assign_weight("layers." + std::to_string(l) + ".post_norm.weight", full({cfg.hidden_size}, 1.0f));
    }
    model.freeze_base();

    // 1. Vision specifications
    const int64_t vision_dim = 32;       // Simulated ViT feature dimension
    const int64_t hidden_size = cfg.hidden_size; // 64
    const int64_t num_patches = 4;      // 4 visual tokens
    const int64_t num_text_tokens = 4;  // 4 prompt tokens

    // 2. Trainable Vision Projection Layer: W_proj [out_features=64, in_features=32]
    auto W_proj = zeros({hidden_size, vision_dim}, kFloat32, kCPU);
    std::mt19937 gen(42);
    std::normal_distribution<float> dist(0.0f, 0.02f);
    for (int64_t i = 0; i < W_proj->numel(); ++i) {
        W_proj->data<float>()[i] = dist(gen);
    }
    W_proj->set_requires_grad(true);

    // 3. Simulated Frozen Vision Output (requires_grad = false)
    auto raw_image_patches = zeros({1, num_patches, vision_dim}, kFloat32, kCPU);
    for (int64_t i = 0; i < raw_image_patches->numel(); ++i) {
        raw_image_patches->data<float>()[i] = dist(gen);
    }
    raw_image_patches->set_requires_grad(false);

    // 4. Text Input IDs
    auto input_ids = zeros({1, num_text_tokens}, kInt32, kCPU);
    input_ids->data<int32_t>()[0] = 5;
    input_ids->data<int32_t>()[1] = 12;
    input_ids->data<int32_t>()[2] = 25;
    input_ids->data<int32_t>()[3] = 40;

    // Helper forward function
    auto forward_pass = [&](const TensorPtr& w_proj_tensor) -> TensorPtr {
        // A. Project visual features to model dimension: [1, 4, 32] @ [64, 32]^T -> [1, 4, 64]
        auto projected_vision = ops::linear(raw_image_patches, w_proj_tensor);

        // B. Lookup text embeddings: [1, 4, 64]
        auto text_embeds = zeros({1, num_text_tokens, hidden_size}, kFloat32, kCPU);
        const auto& embed_weight = model.embedding_weight();
        for (int64_t s = 0; s < num_text_tokens; ++s) {
            int32_t tid = input_ids->data<int32_t>()[s];
            std::memcpy(
                text_embeds->data<float>() + s * hidden_size,
                embed_weight->data<float>() + tid * hidden_size,
                hidden_size * sizeof(float)
            );
        }

        // C. THE SPLICE POINT: Concat vision + text along sequence axis (dim=1) -> [1, 8, 64]
        auto fused_multimodal_embeds = ops::concat({projected_vision, text_embeds}, 1);

        // D. Forward through transformer stack
        auto hidden = model.forward_hidden_from_embeds(fused_multimodal_embeds);

        // E. Compute vocabulary logits: [1, 8, 128]
        auto logits = model.lm_head(hidden);
        return logits;
    };

    std::cout << "--- Step 2: Execute Initial Multimodal Forward Pass ---" << std::endl;
    auto initial_logits = forward_pass(W_proj);
    require(initial_logits != nullptr, "initial logits not null");
    require(initial_logits->shape() == std::vector<int64_t>({1, 8, cfg.vocab_size}), "logits shape [1,8,128]");

    // Target sequence for 8 tokens
    auto targets = zeros({1, 8}, kInt32, kCPU);
    for (int i = 0; i < 8; ++i) targets->data<int32_t>()[i] = (i * 11 + 7) % cfg.vocab_size;

    auto flat_logits = ops::reshape(initial_logits, {8, cfg.vocab_size});
    auto flat_targets = ops::reshape(targets, {8});
    auto loss0 = ops::cross_entropy_loss(flat_logits, flat_targets);
    float initial_loss_val = loss0->item();
    std::cout << "Initial Multimodal Loss: " << initial_loss_val << std::endl;
    require(std::isfinite(initial_loss_val), "initial loss is finite");

    std::cout << "--- Step 3: Execute Analytical Autograd Backward Pass ---" << std::endl;
    loss0->backward();

    // Verify Projector Gradients
    require(W_proj->grad() != nullptr, "W_proj gradient exists");
    std::cout << "W_proj shape: [" << W_proj->shape()[0] << ", " << W_proj->shape()[1] << "]" << std::endl;
    std::cout << "W_proj grad shape: [" << W_proj->grad()->shape()[0] << ", " << W_proj->grad()->shape()[1] << "]" << std::endl;
    require(W_proj->grad()->shape() == std::vector<int64_t>({hidden_size, vision_dim}), "W_proj grad shape");

    float grad_norm_sq = 0.0f;
    const float* g_data = W_proj->grad()->data<float>();
    for (int64_t i = 0; i < W_proj->numel(); ++i) {
        require(std::isfinite(g_data[i]), "gradient value is finite");
        grad_norm_sq += g_data[i] * g_data[i];
    }
    float grad_norm = std::sqrt(grad_norm_sq);
    std::cout << "Projector Gradient Norm: " << grad_norm << std::endl;
    require(grad_norm > 1e-6f, "gradient norm is non-zero and positive");

    std::cout << "--- Step 4: Perform Parameter Update Step & Verify Loss Decrease ---" << std::endl;
    const float lr = 0.001f;
    auto W_proj_updated = zeros(W_proj->shape(), W_proj->dtype(), W_proj->device());
    for (int64_t i = 0; i < W_proj->numel(); ++i) {
        W_proj_updated->data<float>()[i] = W_proj->data<float>()[i] - lr * g_data[i];
    }

    auto updated_logits = forward_pass(W_proj_updated);
    auto loss1 = ops::cross_entropy_loss(ops::reshape(updated_logits, {8, cfg.vocab_size}), ops::reshape(targets, {8}));
    float updated_loss_val = loss1->item();
    std::cout << "Updated Multimodal Loss: " << updated_loss_val << std::endl;
    std::cout << "Loss Delta: " << (initial_loss_val - updated_loss_val) << std::endl;

    require(updated_loss_val < initial_loss_val, "Loss genuinely decreased after gradient update!");
    std::cout << "[SUCCESS] Multimodal forward -> backward -> gradient update verified!" << std::endl;
}

} // namespace

int main() {
    std::cout << "==========================================================" << std::endl;
    std::cout << "Running Qwen Multimodal Autograd & Parameter Update Test" << std::endl;
    std::cout << "==========================================================" << std::endl;
    test_multimodal_forward_backward_convergence();
    std::cout << "\n[ALL TESTS PASSED SUCCESSFULLY]" << std::endl;
    return 0;
}
