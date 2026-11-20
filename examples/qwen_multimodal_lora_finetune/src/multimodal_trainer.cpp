/**
 * @file multimodal_trainer.cpp
 * @brief Implementation of MultimodalTrainer with memory & telemetry logging.
 */

#include "multimodal_trainer.h"
#include "finetune_ops/core/ops.h"
#include "finetune_ops/core/lm_loss.h"
#include <cmath>
#include <iostream>
#include <iomanip>
#include <sstream>

namespace ops {

void MultimodalTrainer::get_process_memory_mb(double& rss_mb, double& hwm_mb) {
    rss_mb = 0.0;
    hwm_mb = 0.0;
    std::ifstream status_file("/proc/self/status");
    if (!status_file.is_open()) return;

    std::string line;
    while (std::getline(status_file, line)) {
        if (line.rfind("VmRSS:", 0) == 0) {
            std::istringstream iss(line.substr(6));
            double kb = 0.0;
            iss >> kb;
            rss_mb = kb / 1024.0;
        } else if (line.rfind("VmHWM:", 0) == 0) {
            std::istringstream iss(line.substr(6));
            double kb = 0.0;
            iss >> kb;
            hwm_mb = kb / 1024.0;
        }
    }
}

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
    start_time_ = std::chrono::steady_clock::now();

    if (!config_.metrics_ndjson_path.empty()) {
        ndjson_out_.open(config_.metrics_ndjson_path, std::ios::app);
        if (!ndjson_out_.is_open()) {
            std::cerr << "[MultimodalTrainer] Warning: failed to open " << config_.metrics_ndjson_path << std::endl;
        }
    }

    if (!config_.rss_csv_path.empty() && config_.enable_rss_logging) {
        bool exists = std::ifstream(config_.rss_csv_path).good();
        rss_out_.open(config_.rss_csv_path, std::ios::app);
        if (rss_out_.is_open() && !exists) {
            rss_out_ << "tick,timestamp,rss_mb\n";
            rss_out_.flush();
        }
    }
}

MultimodalTrainer::~MultimodalTrainer() {
    if (ndjson_out_.is_open()) ndjson_out_.close();
    if (rss_out_.is_open()) rss_out_.close();
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

StepMetrics MultimodalTrainer::train_step(const MultimodalBatchData& batch) {
    if (!batch.vision_features || !batch.input_ids) {
        throw TensorError("MultimodalTrainer::train_step: vision_features and input_ids required");
    }

    const auto step_start = std::chrono::steady_clock::now();

    int64_t B = batch.input_ids->shape()[0];
    int64_t V_len = batch.vision_features->shape()[1];
    int64_t T_len = batch.input_ids->shape()[1];
    int64_t total_len = V_len + T_len;
    int64_t pred_len = total_len - 1;

    if (pred_len <= 0) {
        throw TensorError("MultimodalTrainer::train_step: sequence too short for next-token prediction");
    }

    // 1. Forward pass through visual projector and Qwen transformer
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
                target_ptr[b * pred_len + t] = -100;
            } else {
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

    // 5. Zero gradients on trainable parameters
    auto params = model_.trainable_parameters();
    for (const auto& param : params) {
        if (param && param->grad()) {
            param->zero_grad();
        }
    }

    // 6. Autograd backward pass (routes gradients to projector and LoRA weights)
    loss->backward();

    // 7. Calculate raw gradient norm
    float raw_norm = 0.0f;
    for (const auto& param : params) {
        if (!param || !param->grad()) continue;
        const float* g = param->grad()->data<float>();
        for (int64_t i = 0; i < param->grad()->numel(); ++i) {
            raw_norm += g[i] * g[i];
        }
    }
    raw_norm = std::sqrt(raw_norm);

    // 8. Clip gradients
    clip_gradients();

    // 9. Optimizer update step
    std::vector<TensorPtr> grads;
    grads.reserve(params.size());
    for (const auto& param : params) {
        grads.push_back(param ? param->grad() : nullptr);
    }
    optimizer_->step(params, grads);
    ++step_counter_;

    const auto step_end = std::chrono::steady_clock::now();
    double step_ms = std::chrono::duration<double, std::milli>(step_end - step_start).count();

    // 10. Sample system memory telemetry (Resident Set Size & High Water Mark)
    double rss_mb = 0.0;
    double hwm_mb = 0.0;
    get_process_memory_mb(rss_mb, hwm_mb);

    StepMetrics m;
    m.step = static_cast<int64_t>(step_counter_);
    m.loss = loss_val;
    m.perplexity = (loss_val < 30.0f) ? std::exp(loss_val) : 9999.0f;
    m.grad_norm = raw_norm;
    m.learning_rate = config_.learning_rate;
    m.tokens = B * total_len;
    m.step_time_ms = step_ms;
    m.rss_mb = rss_mb;
    m.peak_hwm_mb = hwm_mb;

    log_telemetry(m);
    return m;
}

void MultimodalTrainer::log_telemetry(const StepMetrics& m) {
    if (ndjson_out_.is_open()) {
        ndjson_out_ << "{\"step\":" << m.step
                    << ",\"loss\":" << std::fixed << std::setprecision(4) << m.loss
                    << ",\"ppl\":" << std::fixed << std::setprecision(2) << m.perplexity
                    << ",\"grad_norm\":" << std::fixed << std::setprecision(4) << m.grad_norm
                    << ",\"lr\":" << std::scientific << std::setprecision(2) << m.learning_rate
                    << ",\"tokens\":" << m.tokens
                    << ",\"rss_mb\":" << std::fixed << std::setprecision(2) << m.rss_mb
                    << ",\"peak_hwm_mb\":" << std::fixed << std::setprecision(2) << m.peak_hwm_mb
                    << ",\"step_time_ms\":" << std::fixed << std::setprecision(2) << m.step_time_ms
                    << "}\n";
        ndjson_out_.flush();
    }

    if (rss_out_.is_open()) {
        auto now = std::chrono::steady_clock::now();
        double elapsed_s = std::chrono::duration<double>(now - start_time_).count();
        rss_out_ << m.step << "," << std::fixed << std::setprecision(3) << elapsed_s
                 << "," << std::fixed << std::setprecision(2) << m.rss_mb << "\n";
        rss_out_.flush();
    }
}

} // namespace ops
