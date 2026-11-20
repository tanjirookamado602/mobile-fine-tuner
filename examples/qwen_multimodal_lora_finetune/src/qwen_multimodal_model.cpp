/**
 * @file qwen_multimodal_model.cpp
 * @brief Implementation of QwenMultimodalModel.
 */

#include "qwen_multimodal_model.h"
#include <cstring>
#include <random>
#include <cmath>

namespace ops {

QwenMultimodalConfig QwenMultimodalConfig::from_pretrained(const std::string& config_path,
                                                          int vision_dim,
                                                          const std::string& precision) {
    QwenMultimodalConfig cfg;
    cfg.text_config = QwenConfig::from_pretrained(config_path);
    cfg.vision_dim = vision_dim;
    cfg.precision = precision;
    return cfg;
}

QwenMultimodalModel::QwenMultimodalModel(const QwenMultimodalConfig& config)
    : config_(config) {
    base_model_ = std::make_shared<QwenModel>(config_.text_config);
    init_base_weights(config_.seed);
    init_projector(config_.seed);
}

void QwenMultimodalModel::init_base_weights(uint64_t seed) {
    std::mt19937 rng(static_cast<uint32_t>(seed));
    auto params = base_model_->parameters();
    if (params.empty()) return;

    size_t idx = 0;
    std::uniform_real_distribution<float> w_dist(-0.02f, 0.02f);

    // 1. embed_tokens_
    if (idx < params.size()) {
        float* d = params[idx]->data<float>();
        for (int64_t i = 0; i < params[idx]->numel(); ++i) d[i] = w_dist(rng);
        idx++;
    }
    // 2. final_norm_weight_
    if (idx < params.size()) {
        float* d = params[idx]->data<float>();
        for (int64_t i = 0; i < params[idx]->numel(); ++i) d[i] = 1.0f;
        idx++;
    }

    // 3. Layer weights
    const int layers = config_.text_config.num_hidden_layers;
    for (int layer = 0; layer < layers && idx < params.size(); ++layer) {
        // input_norm_weight
        if (idx < params.size()) {
            float* d = params[idx]->data<float>();
            for (int64_t i = 0; i < params[idx]->numel(); ++i) d[i] = 1.0f;
            idx++;
        }
        // post_norm_weight
        if (idx < params.size()) {
            float* d = params[idx]->data<float>();
            for (int64_t i = 0; i < params[idx]->numel(); ++i) d[i] = 1.0f;
            idx++;
        }
        // Remaining 10 projections & biases per block
        size_t block_end = std::min(params.size(), 2 + static_cast<size_t>(layer + 1) * 12);
        while (idx < block_end) {
            auto p = params[idx];
            float* d = p->data<float>();
            if (p->ndim() == 1) {
                for (int64_t i = 0; i < p->numel(); ++i) d[i] = 0.0f;
            } else {
                for (int64_t i = 0; i < p->numel(); ++i) d[i] = w_dist(rng);
            }
            idx++;
        }
    }
}

void QwenMultimodalModel::init_projector(uint64_t seed) {
    const int64_t H = config_.text_config.hidden_size;
    const int64_t V = config_.vision_dim;
    const int64_t H_proj = (config_.projector_hidden_dim > 0) ? config_.projector_hidden_dim : H;

    std::mt19937 gen(static_cast<unsigned int>(seed));
    // Xavier / Glorot initialization
    float std_w1 = std::sqrt(2.0f / (V + H_proj));
    std::normal_distribution<float> dist_w1(0.0f, std_w1);

    if (config_.use_mlp_projector) {
        // 2-layer MLP projector: V -> H_proj -> H
        proj_w1_ = zeros({H_proj, V}, kFloat32, kCPU);
        for (int64_t i = 0; i < proj_w1_->numel(); ++i) {
            proj_w1_->data<float>()[i] = dist_w1(gen);
        }
        proj_w1_->set_requires_grad(true);

        float std_w2 = std::sqrt(2.0f / (H_proj + H));
        std::normal_distribution<float> dist_w2(0.0f, std_w2);
        proj_w2_ = zeros({H, H_proj}, kFloat32, kCPU);
        for (int64_t i = 0; i < proj_w2_->numel(); ++i) {
            proj_w2_->data<float>()[i] = dist_w2(gen);
        }
        proj_w2_->set_requires_grad(true);

        if (config_.use_projector_bias) {
            proj_b1_ = zeros({H_proj}, kFloat32, kCPU);
            proj_b1_->set_requires_grad(true);
            proj_b2_ = zeros({H}, kFloat32, kCPU);
            proj_b2_->set_requires_grad(true);
        }
    } else {
        // Single linear projector: V -> H
        proj_w1_ = zeros({H, V}, kFloat32, kCPU);
        for (int64_t i = 0; i < proj_w1_->numel(); ++i) {
            proj_w1_->data<float>()[i] = dist_w1(gen);
        }
        proj_w1_->set_requires_grad(true);
        proj_w2_ = nullptr;

        if (config_.use_projector_bias) {
            proj_b1_ = zeros({H}, kFloat32, kCPU);
            proj_b1_->set_requires_grad(true);
        }
    }
}

TensorPtr QwenMultimodalModel::project_vision(const TensorPtr& vision_features) {
    if (!vision_features) {
        throw TensorError("project_vision: vision_features cannot be null");
    }

    if (config_.use_mlp_projector) {
        // Layer 1: linear -> GELU activation
        auto h1 = ops::linear(vision_features, proj_w1_, proj_b1_);
        auto act = ops::gelu(h1);
        // Layer 2: linear -> LLM hidden dimension
        return ops::linear(act, proj_w2_, proj_b2_);
    } else {
        return ops::linear(vision_features, proj_w1_, proj_b1_);
    }
}

TensorPtr QwenMultimodalModel::forward_multimodal_hidden(const TensorPtr& vision_features,
                                                        const TensorPtr& input_ids,
                                                        const TensorPtr& attention_mask) {
    if (!vision_features || !input_ids) {
        throw TensorError("forward_multimodal_hidden: vision_features and input_ids cannot be null");
    }

    // 1. Project visual features [B, V_len, vision_dim] -> [B, V_len, hidden_size]
    auto projected_vision = project_vision(vision_features);

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

    // 3. Concatenate visual tokens and text tokens along sequence length (axis=1)
    auto fused_embeds = ops::concat({projected_vision, text_embeds}, 1);

    // 4. Construct attention mask covering visual tokens + text tokens
    TensorPtr fused_mask = nullptr;
    if (attention_mask) {
        int64_t V_len = vision_features->shape()[1];
        auto vision_mask = full({B, V_len}, 1.0f, kFloat32, kCPU);
        fused_mask = ops::concat({vision_mask, attention_mask}, 1);
    }

    // 5. Forward through transformer backbone (returns hidden states)
    return base_model_->forward_hidden_from_embeds(fused_embeds, fused_mask);
}

TensorPtr QwenMultimodalModel::forward_multimodal(const TensorPtr& vision_features,
                                                 const TensorPtr& input_ids,
                                                 const TensorPtr& attention_mask) {
    auto hidden = forward_multimodal_hidden(vision_features, input_ids, attention_mask);
    return base_model_->lm_head(hidden);
}

TensorPtr QwenMultimodalModel::forward_from_embeddings(const TensorPtr& fused_embeddings,
                                                      const TensorPtr& fused_mask) {
    auto hidden = base_model_->forward_hidden_from_embeds(fused_embeddings, fused_mask);
    return base_model_->lm_head(hidden);
}

void QwenMultimodalModel::init_lora(int r, float alpha, float dropout, bool qv_only, uint64_t seed) {
    base_model_->init_lora(r, alpha, dropout, qv_only, seed);
}

void QwenMultimodalModel::freeze_base() {
    base_model_->freeze_base();
    if (proj_w1_) proj_w1_->set_requires_grad(true);
    if (proj_b1_) proj_b1_->set_requires_grad(true);
    if (proj_w2_) proj_w2_->set_requires_grad(true);
    if (proj_b2_) proj_b2_->set_requires_grad(true);
}

std::vector<TensorPtr> QwenMultimodalModel::parameters() const {
    auto params = base_model_->parameters();
    auto lora_params = base_model_->get_lora_parameters();
    params.insert(params.end(), lora_params.begin(), lora_params.end());
    if (proj_w1_) params.push_back(proj_w1_);
    if (proj_b1_) params.push_back(proj_b1_);
    if (proj_w2_) params.push_back(proj_w2_);
    if (proj_b2_) params.push_back(proj_b2_);
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

} // namespace ops
