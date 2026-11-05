/**
 * @file qwen_multimodal.h
 * @brief Multimodal wrapper for Qwen model with visual projection layer.
 */

#pragma once

#include "qwen_model.h"
#include "../core/tensor.h"
#include <memory>
#include <string>
#include <vector>

namespace ops {

struct QwenMultimodalConfig {
    QwenConfig text_config;
    int vision_dim = 32;
    bool use_projector_bias = false;
    uint64_t seed = 42;

    static QwenMultimodalConfig from_pretrained(const std::string& config_path, int vision_dim = 32);
};

class QwenMultimodalModel {
public:
    explicit QwenMultimodalModel(const QwenMultimodalConfig& config);
    ~QwenMultimodalModel() = default;

    TensorPtr forward_multimodal(const TensorPtr& vision_features,
                                const TensorPtr& input_ids,
                                const TensorPtr& attention_mask = nullptr);

    std::vector<TensorPtr> parameters() const;
    std::vector<TensorPtr> trainable_parameters() const;

    void freeze_base();
    void init_projector(uint64_t seed = 42);

    const QwenMultimodalConfig& config() const { return config_; }
    const std::shared_ptr<QwenModel>& base_model() const { return base_model_; }
    const TensorPtr& projector_weight() const { return projector_weight_; }
    const TensorPtr& projector_bias() const { return projector_bias_; }
    void set_projector_weight(const TensorPtr& w) { projector_weight_ = w; }
    void set_projector_bias(const TensorPtr& b) { projector_bias_ = b; }

private:
    QwenMultimodalConfig config_;
    std::shared_ptr<QwenModel> base_model_;
    TensorPtr projector_weight_; // [hidden_size, vision_dim]
    TensorPtr projector_bias_;   // [hidden_size] (optional)
};

}  // namespace ops

