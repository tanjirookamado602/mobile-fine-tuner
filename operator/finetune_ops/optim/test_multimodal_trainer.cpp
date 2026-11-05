#include "multimodal_trainer.h"
#include "../core/ops.h"
#include <iostream>
#include <iomanip>
#include <cmath>
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

void test_multimodal_trainer_20step_convergence() {
    std::cout << "==========================================================" << std::endl;
    std::cout << "Running MultimodalTrainer 20-Step Fixed-Batch Test" << std::endl;
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
    mm_cfg.seed = 42;

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

    // 2. Build Trainer
    MultimodalTrainerConfig trainer_cfg;
    trainer_cfg.learning_rate = 1e-3f;
    trainer_cfg.max_grad_norm = 1.0f;
    trainer_cfg.weight_decay = 0.0f;
    MultimodalTrainer trainer(model, trainer_cfg);

    // 3. Build fixed batch
    const int64_t B = 1;
    const int64_t V_len = 2;
    const int64_t T_len = 3;

    MultimodalBatch batch;
    batch.vision_features = zeros({B, V_len, mm_cfg.vision_dim}, kFloat32, kCPU);
    std::mt19937 gen(1234);
    std::normal_distribution<float> dist(0.0f, 0.05f);
    for (int64_t i = 0; i < batch.vision_features->numel(); ++i) {
        batch.vision_features->data<float>()[i] = dist(gen);
    }

    batch.input_ids = zeros({B, T_len}, kInt32, kCPU);
    for (int64_t i = 0; i < T_len; ++i) {
        batch.input_ids->data<int32_t>()[i] = static_cast<int32_t>((i * 7 + 11) % mm_cfg.text_config.vocab_size);
    }

    batch.attention_mask = full({B, T_len}, 1.0f, kFloat32, kCPU);
    batch.labels = nullptr;

    // 4. Run 20 training steps
    std::vector<float> losses;
    losses.reserve(20);

    for (int step = 0; step < 20; ++step) {
        float loss = trainer.train_step(batch);
        require(std::isfinite(loss) && loss > 0.0f, "Step loss is finite and positive");
        losses.push_back(loss);
        std::cout << "Step " << std::setw(2) << step + 1 << "/20 - Loss: " << std::fixed << std::setprecision(5) << loss << std::endl;
    }

    float initial_loss = losses.front();
    float final_loss = losses.back();
    float total_decrease = initial_loss - final_loss;

    std::cout << "--------------------------------------------------------" << std::endl;
    std::cout << "Initial Loss: " << initial_loss << " -> Final Loss: " << final_loss << std::endl;
    std::cout << "Total Loss Decrease: " << total_decrease << " (" << (total_decrease / initial_loss * 100.0f) << "%)" << std::endl;

    require(final_loss < initial_loss, "Final loss must be strictly lower than initial loss");
    require(total_decrease > 0.1f, "Loss must decrease by at least 0.1 over 20 steps");

    std::cout << "[SUCCESS] MultimodalTrainer 20-step convergence test passed!" << std::endl;
}

}  // namespace

int main() {
    test_multimodal_trainer_20step_convergence();
    return 0;
}
