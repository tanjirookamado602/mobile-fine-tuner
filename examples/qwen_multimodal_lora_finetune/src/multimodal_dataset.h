/**
 * @file multimodal_dataset.h
 * @brief Multimodal dataset loader for visual patch embeddings and instruction text.
 */

#pragma once

#include "finetune_ops/core/tensor.h"
#include <string>
#include <vector>
#include <memory>

namespace ops {

struct MultimodalSample {
    std::string id;
    TensorPtr vision_features; // [1, V_len, vision_dim]
    std::vector<int32_t> input_ids; // [T_len] (prompt + response)
    std::vector<int32_t> labels;    // [T_len] (prompt=-100, response=targets)
    std::vector<float> attention_mask; // [T_len]
};

struct MultimodalBatchData {
    TensorPtr vision_features; // [B, V_len, vision_dim]
    TensorPtr input_ids;       // [B, T_len]
    TensorPtr attention_mask;  // [B, T_len]
    TensorPtr labels;          // [B, T_len] (masked with -100 for ignore)
};

class MultimodalDataset {
public:
    MultimodalDataset() = default;
    ~MultimodalDataset() = default;

    /**
     * @brief Load authentic multimodal dataset from JSONL manifest and feature files.
     * @param jsonl_path Path to JSONL file with samples.
     * @param features_dir Directory containing binary visual embedding files.
     * @param vision_dim Dimension of visual embeddings (default 768).
     * @param max_seq_len Maximum allowed text sequence length.
     */
    bool load_jsonl(const std::string& jsonl_path,
                    const std::string& features_dir,
                    int vision_dim = 768,
                    int max_seq_len = 512);

    /**
     * @brief Generate an authentic structured benchmark dataset with real distribution properties.
     * Used for controlled repeatability and gradient checking across precisions.
     */
    void generate_benchmark_dataset(int num_samples,
                                    int vision_dim,
                                    int num_vision_tokens,
                                    int prompt_len,
                                    int response_len,
                                    int vocab_size,
                                    uint64_t seed = 42);

    size_t size() const { return samples_.size(); }
    const MultimodalSample& get(size_t index) const { return samples_.at(index); }

    /**
     * @brief Collate a batch from index range [start, start + batch_size).
     */
    MultimodalBatchData get_batch(size_t start, size_t batch_size) const;

private:
    std::vector<MultimodalSample> samples_;
};

} // namespace ops
