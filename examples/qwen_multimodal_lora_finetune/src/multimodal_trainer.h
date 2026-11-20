/**
 * @file multimodal_trainer.h
 * @brief High-performance multimodal trainer with system metrics and RSS logging.
 */

#pragma once

#include "qwen_multimodal_model.h"
#include "multimodal_dataset.h"
#include "finetune_ops/optim/adam.h"
#include "finetune_ops/energy/power_monitor.h"
#include <fstream>
#include <string>
#include <memory>
#include <chrono>

namespace ops {

struct MultimodalTrainerConfig {
    float learning_rate = 1e-5f; // Standard fine-tuning learning rate (1e-5)
    float max_grad_norm = 1.0f;
    float weight_decay = 0.01f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float eps = 1e-8f;
    std::string precision = "fp32"; // "fp32", "bf16", "fp16"

    std::string metrics_ndjson_path = "";
    std::string rss_csv_path = "";
    bool enable_rss_logging = true;
};

struct StepMetrics {
    int64_t step = 0;
    float loss = 0.0f;
    float perplexity = 0.0f;
    float grad_norm = 0.0f;
    float learning_rate = 1e-5f;
    int64_t tokens = 0;
    double step_time_ms = 0.0;
    double rss_mb = 0.0;
    double peak_hwm_mb = 0.0;
};

class MultimodalTrainer {
public:
    MultimodalTrainer(QwenMultimodalModel& model, const MultimodalTrainerConfig& config);
    ~MultimodalTrainer();

    /**
     * @brief Execute a single training step on a multimodal batch.
     * Performs forward pass, next-token prediction loss, backward pass,
     * gradient clipping, optimizer step, and telemetry logging.
     */
    StepMetrics train_step(const MultimodalBatchData& batch);

    void clip_gradients();
    void reset_step_counter() { step_counter_ = 0; }

    const MultimodalTrainerConfig& config() const { return config_; }
    QwenMultimodalModel& model() { return model_; }
    size_t step_count() const { return step_counter_; }

    /**
     * @brief Sample current process memory from /proc/self/status.
     * @param[out] rss_mb Current resident set size in MB.
     * @param[out] hwm_mb Peak memory (High Water Mark) in MB.
     */
    static void get_process_memory_mb(double& rss_mb, double& hwm_mb);

private:
    void log_telemetry(const StepMetrics& m);

    QwenMultimodalModel& model_;
    MultimodalTrainerConfig config_;
    std::shared_ptr<Adam> optimizer_;
    size_t step_counter_ = 0;

    std::ofstream ndjson_out_;
    std::ofstream rss_out_;
    std::chrono::steady_clock::time_point start_time_;
};

} // namespace ops
