#include "multimodal_trainer.h"
#include "../core/ops.h"
#include <cmath>
#include <iostream>

namespace ops {

MultimodalTrainer::MultimodalTrainer(QwenMultimodalModel& model, const MultimodalTrainerConfig& config)
    : model_(model), config_(config) {
    AdamConfig adam_cfg;
    adam_cfg.learning_rate = config_.learning_rate;
    adam_cfg.weight_decay = config_.weight_decay;
    adam_cfg.beta1 = config_.beta1;
    adam_cfg.beta2 = config_.beta2;
    adam_cfg.epsilon = config_.eps;
    adam_cfg.clip_grad_norm = config_.max_grad_norm;

    optimizer_ = std::make_shared<Adam>(adam_cfg);
}

void MultimodalTrainer::clip_gradients() {
    auto params = model_.trainable_parameters();
    float total_norm = 0.0f;
    for (const auto& param : params) {
        if (!param || !param->grad()) continue;
        const float* g = param->grad()->data<float>();
        for (int64_t i = 0; i < param->grad()->numel(); ++i) {
            total_norm += g[i] * g[i];
        }
    }
    total_norm = std::sqrt(total_norm);
    if (total_norm <= config_.max_grad_norm || config_.max_grad_norm <= 0.0f) return;

    float scale = config_.max_grad_norm / (total_norm + 1e-6f);
    for (const auto& param : params) {
        if (!param || !param->grad()) continue;
        float* g = param->grad()->data<float>();
        for (int64_t i = 0; i < param->grad()->numel(); ++i) {
            g[i] *= scale;
        }
    }
}

float MultimodalTrainer::train_step(const MultimodalBatch& batch) {
    if (!batch.vision_features || !batch.input_ids) {
        throw TensorError("MultimodalTrainer::train_step: vision_features and input_ids required");
    }

    int64_t B = batch.input_ids->shape()[0];
    int64_t V_len = batch.vision_features->shape()[1];
    int64_t T_len = batch.input_ids->shape()[1];
    int64_t total_len = V_len + T_len;
    int64_t pred_len = total_len - 1;

    if (pred_len <= 0) {
        throw TensorError("MultimodalTrainer::train_step: sequence too short for next-token prediction");
    }

    // 1. Forward pass
    auto logits = model_.forward_multimodal(batch.vision_features, batch.input_ids, batch.attention_mask);

    // 2. Shift logits: slice positions [0, pred_len) along dim 1
    auto shift_logits = logits->slice(1, 0, pred_len, 1);

    // 3. Shift targets: length pred_len, with -100 on vision positions
    auto shift_targets = zeros({B, pred_len}, kInt32, kCPU);
    int32_t* target_ptr = shift_targets->data<int32_t>();

    const int32_t* label_src = batch.labels ? batch.labels->data<int32_t>() : batch.input_ids->data<int32_t>();

    for (int64_t b = 0; b < B; ++b) {
        for (int64_t t = 0; t < pred_len; ++t) {
            int64_t next_pos = t + 1;
            if (next_pos < V_len) {
                // Next token is still a visual token -> mask out with -100
                target_ptr[b * pred_len + t] = -100;
            } else {
                // Next token is a text token
                int64_t text_idx = next_pos - V_len;
                target_ptr[b * pred_len + t] = label_src[b * T_len + text_idx];
            }
        }
    }

    int64_t vocab_size = model_.config().text_config.vocab_size;
    auto flat_logits = ops::reshape(shift_logits, {B * pred_len, vocab_size});
    auto flat_targets = ops::reshape(shift_targets, {B * pred_len});

    // 4. Compute cross entropy loss with ignore_index=-100
    auto loss = ops::cross_entropy_loss(flat_logits, flat_targets);
    float loss_val = loss->item();

    // 5. Zero grads
    auto params = model_.trainable_parameters();
    for (const auto& param : params) {
        if (param && param->grad()) {
            param->zero_grad();
        }
    }

    // 6. Backward pass
    loss->backward();

    // 7. Gradient clipping
    clip_gradients();

    // 8. Optimizer step
    std::vector<TensorPtr> grads;
    grads.reserve(params.size());
    for (const auto& param : params) {
        grads.push_back(param ? param->grad() : nullptr);
    }
    optimizer_->step(params, grads);

    step_counter_++;
    return loss_val;
}

}  // namespace ops

