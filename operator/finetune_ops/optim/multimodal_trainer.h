/**
 * @file multimodal_trainer.h
 * @brief Trainer for Qwen multimodal model aligning visual projector.
 */

#pragma once

#include "../core/tensor.h"
#include "../graph/qwen_multimodal.h"
#include "adam.h"
#include <memory>
#include <string>
#include <vector>

namespace ops {

struct MultimodalBatch {
    TensorPtr vision_features; // [B, V_len, vision_dim]
    TensorPtr input_ids;       // [B, T_len] (kInt32)
    TensorPtr attention_mask;  // [B, T_len] (optional, 1=valid, 0=pad)
    TensorPtr labels;          // [B, T_len] (optional, default: input_ids)
};

struct MultimodalTrainerConfig {
    float learning_rate = 1e-3f;
    float max_grad_norm = 1.0f;
    float weight_decay = 0.0f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float eps = 1e-8f;
};

class MultimodalTrainer {
public:
    MultimodalTrainer(QwenMultimodalModel& model, const MultimodalTrainerConfig& config);
    ~MultimodalTrainer() = default;

    float train_step(const MultimodalBatch& batch);
    void clip_gradients();

    const MultimodalTrainerConfig& config() const { return config_; }
    QwenMultimodalModel& model() { return model_; }
    const std::shared_ptr<Adam>& optimizer() const { return optimizer_; }
    size_t step_count() const { return step_counter_; }

private:
    QwenMultimodalModel& model_;
    MultimodalTrainerConfig config_;
    std::shared_ptr<Adam> optimizer_;
    size_t step_counter_ = 0;
};

}  // namespace ops

