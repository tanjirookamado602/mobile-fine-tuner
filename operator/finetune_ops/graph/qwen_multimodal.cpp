#include "qwen_multimodal.h"
#include "../core/ops.h"
#include <cstring>
#include <random>

namespace ops {

QwenMultimodalConfig QwenMultimodalConfig::from_pretrained(const std::string& config_path, int vision_dim) {
    QwenMultimodalConfig cfg;
    cfg.text_config = QwenConfig::from_pretrained(config_path);
    cfg.vision_dim = vision_dim;
    return cfg;
}

QwenMultimodalModel::QwenMultimodalModel(const QwenMultimodalConfig& config)
    : config_(config) {
    base_model_ = std::make_shared<QwenModel>(config_.text_config);
    init_projector(config_.seed);
}

void QwenMultimodalModel::init_projector(uint64_t seed) {
    int64_t out_dim = config_.text_config.hidden_size;
    int64_t in_dim = config_.vision_dim;

    projector_weight_ = zeros({out_dim, in_dim}, kFloat32, kCPU);
    std::mt19937 gen(seed);
    std::normal_distribution<float> dist(0.0f, 0.02f);
    for (int64_t i = 0; i < projector_weight_->numel(); ++i) {
        projector_weight_->data<float>()[i] = dist(gen);
    }
    projector_weight_->set_requires_grad(true);

    if (config_.use_projector_bias) {
        projector_bias_ = zeros({out_dim}, kFloat32, kCPU);
        projector_bias_->set_requires_grad(true);
    } else {
        projector_bias_ = nullptr;
    }
}

TensorPtr QwenMultimodalModel::forward_multimodal(const TensorPtr& vision_features,
                                                 const TensorPtr& input_ids,
                                                 const TensorPtr& attention_mask) {
    if (!vision_features || !input_ids) {
        throw TensorError("forward_multimodal: vision_features and input_ids cannot be null");
    }

    // 1. Project visual features [B, V_len, vision_dim] -> [B, V_len, hidden_size]
    auto projected_vision = ops::linear(vision_features, projector_weight_, projector_bias_);

    // 2. Lookup text token embeddings [B, T_len] -> [B, T_len, hidden_size]
    const auto& embed_weight = base_model_->embedding_weight();
    int64_t B = input_ids->shape()[0];
    int64_t T_len = input_ids->shape()[1];
    int64_t H = config_.text_config.hidden_size;

    auto text_embeds = zeros({B, T_len, H}, kFloat32, kCPU);
    const int32_t* idx = input_ids->data<int32_t>();
    float* o = text_embeds->data<float>();
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t s = 0; s < T_len; ++s) {
            int32_t id = idx[b * T_len + s];
            float* dst = o + (b * T_len + s) * H;
            const float* src = embed_weight->data<float>() + id * H;
            std::memcpy(dst, src, sizeof(float) * H);
        }
    }

    // 3. Fused multimodal sequence [B, V_len + T_len, hidden_size]
    auto fused_embeds = ops::concat({projected_vision, text_embeds}, 1);

    // 4. Extend attention mask over vision positions
    TensorPtr fused_mask = nullptr;
    if (attention_mask) {
        int64_t V_len = vision_features->shape()[1];
        auto vision_mask = full({B, V_len}, 1.0f, kFloat32, kCPU);
        fused_mask = ops::concat({vision_mask, attention_mask}, 1);
    }

    // 5. Forward through transformer backbone
    auto hidden = base_model_->forward_hidden_from_embeds(fused_embeds, fused_mask);

    // 6. Compute output vocabulary logits [B, V_len + T_len, vocab_size]
    return base_model_->lm_head(hidden);
}

std::vector<TensorPtr> QwenMultimodalModel::parameters() const {
    auto params = base_model_->parameters();
    if (projector_weight_) params.push_back(projector_weight_);
    if (projector_bias_) params.push_back(projector_bias_);
    return params;
}

std::vector<TensorPtr> QwenMultimodalModel::trainable_parameters() const {
    std::vector<TensorPtr> trainables;
    for (const auto& p : parameters()) {
        if (p && p->requires_grad()) {
            trainables.push_back(p);
        }
    }
    return trainables;
}

void QwenMultimodalModel::freeze_base() {
    base_model_->freeze_base();
    if (projector_weight_) projector_weight_->set_requires_grad(true);
    if (projector_bias_) projector_bias_->set_requires_grad(true);
}

}  // namespace ops

