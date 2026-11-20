/**
 * @file qwen_multimodal_model.h
 * @brief Multimodal Qwen model with visual projection layer and LoRA support.
 * Located in examples/qwen_multimodal_lora_finetune/ to keep core operator/ clean.
 */

#pragma once

#include "finetune_ops/core/tensor.h"
#include "finetune_ops/core/ops.h"
#include "finetune_ops/core/dtype.h"
#include "finetune_ops/graph/qwen_model.h"
#include "finetune_ops/graph/safetensors_loader.h"
#include <memory>
#include <string>
#include <vector>

namespace ops {

struct QwenMultimodalConfig {
    QwenConfig text_config;
    int vision_dim = 768; // Default to standard ViT/SigLIP patch embedding dimension
    int num_vision_tokens = 64; // Number of visual patch tokens
    bool use_mlp_projector = true; // 2-layer MLP projector (standard in modern VLMs)
    int projector_hidden_dim = -1; // -1 means text_config.hidden_size
    bool use_projector_bias = false;
    uint64_t seed = 42;
    std::string precision = "fp32"; // "fp32", "bf16", "fp16"

    static QwenMultimodalConfig from_pretrained(const std::string& config_path,
                                                int vision_dim = 768,
                                                const std::string& precision = "fp32");
};

class QwenMultimodalModel {
public:
    explicit QwenMultimodalModel(const QwenMultimodalConfig& config);
    ~QwenMultimodalModel() = default;

    /**
     * @brief Forward pass returning transformer hidden states before LM head.
     * Avoids allocating large [B, S, V] logits matrix when used with selected_token_lm_cross_entropy.
     * @return [B, V_len + T_len, hidden_size]
     */
    TensorPtr forward_multimodal_hidden(const TensorPtr& vision_features,
                                        const TensorPtr& input_ids,
                                        const TensorPtr& attention_mask = nullptr);

    /**
     * @brief Forward pass combining visual patch features and text token IDs.
     * @param vision_features [B, V_len, vision_dim] float32 tensor of image patch tokens.
     * @param input_ids [B, T_len] int32 tensor of text token IDs.
     * @param attention_mask [B, T_len] optional text attention mask (1=active, 0=pad).
     * @return Logits tensor [B, V_len + T_len, vocab_size].
     */
    TensorPtr forward_multimodal(const TensorPtr& vision_features,
                                 const TensorPtr& input_ids,
                                 const TensorPtr& attention_mask = nullptr);

    /**
     * @brief Direct forward pass using already embedded and concatenated representations.
     */
    TensorPtr forward_from_embeddings(const TensorPtr& fused_embeddings,
                                      const TensorPtr& fused_mask = nullptr);

    /**
     * @brief Project visual features into the model's text embedding space.
     * @param vision_features [B, V_len, vision_dim]
     * @return [B, V_len, hidden_size]
     */
    TensorPtr project_vision(const TensorPtr& vision_features);

    /**
     * @brief Initialize LoRA adapters on the base Qwen model attention projections.
     */
    void init_lora(int r = 16, float alpha = 32.0f, float dropout = 0.05f, bool qv_only = false, uint64_t seed = 42);

    /**
     * @brief Freeze the base language model; only projector and LoRA weights receive gradients.
     */
    void freeze_base();

    std::vector<TensorPtr> parameters() const;
    std::vector<TensorPtr> trainable_parameters() const;

    const QwenMultimodalConfig& config() const { return config_; }
    const std::shared_ptr<QwenModel>& base_model() const { return base_model_; }
    const TensorPtr& embedding_weight() const { return base_model_->embedding_weight(); }
    const TensorPtr& proj_w1() const { return proj_w1_; }
    const TensorPtr& proj_b1() const { return proj_b1_; }
    const TensorPtr& proj_w2() const { return proj_w2_; }
    const TensorPtr& proj_b2() const { return proj_b2_; }

    void set_proj_w1(const TensorPtr& w) { proj_w1_ = w; }
    void set_proj_w2(const TensorPtr& w) { proj_w2_ = w; }

private:
    void init_projector(uint64_t seed);
    void init_base_weights(uint64_t seed);

    QwenMultimodalConfig config_;
    std::shared_ptr<QwenModel> base_model_;
    TensorPtr proj_w1_; // [H_proj, vision_dim]
    TensorPtr proj_b1_; // [H_proj] (optional)
    TensorPtr proj_w2_; // [hidden_size, H_proj] (for MLP projector)
    TensorPtr proj_b2_; // [hidden_size] (optional)
};

} // namespace ops
