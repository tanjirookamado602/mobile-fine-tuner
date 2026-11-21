#include "finetune_ops/core/tensor.h"
#include "finetune_ops/core/ops.h"
#include "finetune_ops/graph/auto_model.h"

#include <iostream>
#include <chrono>
#include <iomanip>
#include <vector>
#include <cstring>
#include <ctime>
#include <cmath>
#include <numeric>
#include <malloc.h>

using namespace ops;

std::string get_wall_time_iso() {
    auto now = std::chrono::system_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
    auto timer = std::chrono::system_clock::to_time_t(now);
    std::tm bt{};
    localtime_r(&timer, &bt);
    std::ostringstream oss;
    oss << std::put_time(&bt, "%Y-%m-%d %H:%M:%S") << '.' << std::setfill('0') << std::setw(3) << ms.count();
    return oss.str();
}

TensorPtr lookup_embeds(const TensorPtr& weight, const TensorPtr& indices) {
    auto shape = indices->shape();
    int64_t B = shape[0], S = shape[1], H = weight->shape()[1];
    auto out = zeros({B, S, H}, kFloat32, kCPU);
    const int32_t* idx = static_cast<const int32_t*>(indices->data_ptr());
    const float* w = static_cast<const float*>(weight->data_ptr());
    float* o = static_cast<float*>(out->data_ptr());
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t s = 0; s < S; ++s) {
            int32_t id = idx[b * S + s];
            if (id >= 0 && id < weight->shape()[0]) {
                std::memcpy(o + (b * S + s) * H, w + id * H, H * sizeof(float));
            }
        }
    }
    return out;
}

struct StepTiming {
    double fwd_s;
    double bwd_s;
    double total_s;
    float loss;
};

StepTiming run_step(QwenModel* qwen, bool freeze_backbone, int64_t P100, int64_t T100, int64_t H) {
    int64_t total_tokens = P100 + T100;

    auto all_params = qwen->parameters();
    if (freeze_backbone) {
        qwen->freeze_base();
    } else {
        for (auto& p : all_params) {
            if (p) {
                p->set_requires_grad(true);
                p->set_grad(nullptr);
            }
        }
    }

    auto prefix100 = zeros({1, P100, H}, kFloat32, kCPU);
    prefix100->set_requires_grad(true);

    auto text_ids100 = zeros({1, T100}, DType::kInt32, kCPU);
    int32_t* tid_ptr = static_cast<int32_t*>(text_ids100->data_ptr());
    for (int i = 0; i < T100; ++i) tid_ptr[i] = 100 + i;

    auto embeds100 = lookup_embeds(qwen->embedding_weight(), text_ids100);
    auto spliced100 = concat(std::vector<TensorPtr>{prefix100, embeds100}, 1);

    auto t0 = std::chrono::high_resolution_clock::now();
    auto h100 = qwen->forward_hidden_from_embeds(spliced100);
    auto l100 = qwen->lm_head(h100);
    auto t1 = std::chrono::high_resolution_clock::now();
    double fwd_s = std::chrono::duration<double>(t1 - t0).count();

    auto flat_l100 = reshape(l100->slice(1, 0, total_tokens - 1, 1), {total_tokens - 1, 151936});
    auto targets100 = zeros({total_tokens - 1}, DType::kInt32, kCPU);
    int32_t* t100_ptr = static_cast<int32_t*>(targets100->data_ptr());
    for (int i = 0; i < total_tokens - 1; ++i) {
        t100_ptr[i] = (i < P100 - 1) ? -100 : 100 + (i - (P100 - 1));
    }
    auto loss100 = cross_entropy_loss(flat_l100, targets100, "mean");
    float loss_val = static_cast<const float*>(loss100->data_ptr())[0];

    auto t2 = std::chrono::high_resolution_clock::now();
    loss100->backward();
    auto t3 = std::chrono::high_resolution_clock::now();
    double bwd_s = std::chrono::duration<double>(t3 - t2).count();

    // Release all gradient tensors and autograd references immediately to prevent RSS accumulation
    prefix100->set_grad(nullptr);
    for (auto& p : all_params) {
        if (p) p->set_grad(nullptr);
    }
    malloc_trim(0);

    return {fwd_s, bwd_s, fwd_s + bwd_s, loss_val};
}

void print_summary(const std::string& name, const std::vector<StepTiming>& runs) {
    std::cout << "\n--- " << name << " (3 Runs) ---\n";
    std::vector<double> fwd_vec, bwd_vec, tot_vec;
    for (size_t i = 0; i < runs.size(); ++i) {
        std::cout << "  Run " << (i + 1) << ": Forward = " << std::fixed << std::setprecision(2) << runs[i].fwd_s << " s"
                  << " | Backward = " << runs[i].bwd_s << " s"
                  << " | Total = " << runs[i].total_s << " s"
                  << " | Loss = " << std::setprecision(4) << runs[i].loss << "\n";
        fwd_vec.push_back(runs[i].fwd_s);
        bwd_vec.push_back(runs[i].bwd_s);
        tot_vec.push_back(runs[i].total_s);
    }

    auto calc_stats = [](const std::vector<double>& v) {
        double sum = std::accumulate(v.begin(), v.end(), 0.0);
        double mean = sum / v.size();
        double min_val = *std::min_element(v.begin(), v.end());
        double max_val = *std::max_element(v.begin(), v.end());
        double spread = max_val - min_val;
        return std::make_tuple(mean, spread, min_val, max_val);
    };

    auto [fwd_mean, fwd_spread, fwd_min, fwd_max] = calc_stats(fwd_vec);
    auto [bwd_mean, bwd_spread, bwd_min, bwd_max] = calc_stats(bwd_vec);
    auto [tot_mean, tot_spread, tot_min, tot_max] = calc_stats(tot_vec);

    std::cout << "  Summary Statistics:\n";
    std::cout << "    Forward:  mean = " << std::fixed << std::setprecision(2) << fwd_mean 
              << " s, spread = " << fwd_spread << " s [" << fwd_min << " s - " << fwd_max << " s]\n";
    std::cout << "    Backward: mean = " << bwd_mean 
              << " s, spread = " << bwd_spread << " s [" << bwd_min << " s - " << bwd_max << " s]\n";
    std::cout << "    Total:    mean = " << tot_mean 
              << " s, spread = " << tot_spread << " s [" << tot_min << " s - " << tot_max << " s]\n";
}

int main(int argc, char** argv) {
    std::string model_dir = "models/Qwen2.5-0.5B";
    std::cout << "========================================================================\n";
    std::cout << "  100-Token Benchmark: Frozen Backbone vs Full Model Training Step      \n";
    std::cout << "  Host Hardware: Intel(R) Core(TM) i5-7200U CPU @ 2.50GHz (2C/4T)       \n";
    std::cout << "  Model: Qwen2.5-0.5B (24 Layers, Hidden=896, Vocab=151936)             \n";
    std::cout << "  Sequence: 100 Tokens (64 Vision Prefix Tokens + 36 Text Prompt Tokens) \n";
    std::cout << "  Repetitions: 3 iterations per configuration                           \n";
    std::cout << "========================================================================\n";

    std::cout << "[" << get_wall_time_iso() << "] Loading Qwen2.5-0.5B from " << model_dir << "...\n";
    AutoModelLoadOptions opts;
    opts.load_weights = true;
    auto model = AutoModelForCausalLM::from_pretrained(model_dir, opts);
    if (!model) {
        std::cerr << "Failed to load model\n";
        return 1;
    }
    auto qwen = model->qwen();

    int64_t H = 896;
    int64_t P100 = 64; // 64 vision patch tokens
    int64_t T100 = 36; // 36 text prompt tokens

    // 1. Frozen Backbone (Stage 1 Multimodal Training)
    std::cout << "\n========================================================================\n";
    std::cout << "  CONFIGURATION 1: Frozen Backbone (Backbone Frozen, Projector Trainable)\n";
    std::cout << "========================================================================\n";
    std::vector<StepTiming> frozen_runs;
    for (int i = 0; i < 3; ++i) {
        std::cout << "[" << get_wall_time_iso() << "] Executing Run " << (i + 1) << " / 3...\n";
        auto timing = run_step(qwen, true, P100, T100, H);
        frozen_runs.push_back(timing);
        std::cout << "[" << get_wall_time_iso() << "] Run " << (i + 1) << " done: Fwd=" 
                  << std::fixed << std::setprecision(2) << timing.fwd_s << "s, Bwd=" 
                  << timing.bwd_s << "s, Tot=" << timing.total_s << "s\n";
    }
    print_summary("Frozen Backbone Configuration", frozen_runs);

    // 2. Full Model Fine-Tuning (All 290 parameter tensors computed)
    std::cout << "\n========================================================================\n";
    std::cout << "  CONFIGURATION 2: Full Model Fine-Tuning (All 290 Parameter Tensors)    \n";
    std::cout << "========================================================================\n";
    std::vector<StepTiming> full_runs;
    for (int i = 0; i < 3; ++i) {
        std::cout << "[" << get_wall_time_iso() << "] Executing Run " << (i + 1) << " / 3...\n";
        auto timing = run_step(qwen, false, P100, T100, H);
        full_runs.push_back(timing);
        std::cout << "[" << get_wall_time_iso() << "] Run " << (i + 1) << " done: Fwd=" 
                  << std::fixed << std::setprecision(2) << timing.fwd_s << "s, Bwd=" 
                  << timing.bwd_s << "s, Tot=" << timing.total_s << "s\n";
    }
    print_summary("Full Model Configuration", full_runs);

    return 0;
}
