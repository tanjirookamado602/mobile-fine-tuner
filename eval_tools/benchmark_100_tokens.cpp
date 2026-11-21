#include "finetune_ops/core/tensor.h"
#include "finetune_ops/core/ops.h"
#include "finetune_ops/graph/auto_model.h"

#include <iostream>
#include <chrono>
#include <iomanip>
#include <vector>
#include <cstring>

using namespace ops;

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

int main(int argc, char** argv) {
    std::string model_dir = "models/Qwen2.5-0.5B";
    std::cout << "Loading Qwen2.5-0.5B from " << model_dir << "...\n";
    AutoModelLoadOptions opts;
    opts.load_weights = true;
    auto model = AutoModelForCausalLM::from_pretrained(model_dir, opts);
    auto qwen = model->qwen();

    int64_t H = 896;
    int64_t P100 = 64; // 64 visual patch tokens
    int64_t T100 = 36; // 36 text prompt tokens
    int64_t total_tokens = P100 + T100; // 100 tokens

    auto prefix100 = zeros({1, P100, H}, kFloat32, kCPU);
    prefix100->set_requires_grad(true);

    auto text_ids100 = zeros({1, T100}, DType::kInt32, kCPU);
    int32_t* tid_ptr = static_cast<int32_t*>(text_ids100->data_ptr());
    for (int i = 0; i < T100; ++i) tid_ptr[i] = 100 + i;

    auto embeds100 = lookup_embeds(qwen->embedding_weight(), text_ids100);
    auto spliced100 = concat(std::vector<TensorPtr>{prefix100, embeds100}, 1);

    std::cout << "Running 100-token Forward Pass (24 layers, 100 sequence length)...\n";
    auto t0 = std::chrono::high_resolution_clock::now();
    auto h100 = qwen->forward_hidden_from_embeds(spliced100);
    auto l100 = qwen->lm_head(h100);
    auto t1 = std::chrono::high_resolution_clock::now();
    double fwd_100_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Mask prefix positions (0..62) as -100, supervise text positions (63..98)
    auto flat_l100 = reshape(l100->slice(1, 0, total_tokens - 1, 1), {total_tokens - 1, 151936});
    auto targets100 = zeros({total_tokens - 1}, DType::kInt32, kCPU);
    int32_t* t100_ptr = static_cast<int32_t*>(targets100->data_ptr());
    for (int i = 0; i < total_tokens - 1; ++i) {
        t100_ptr[i] = (i < P100 - 1) ? -100 : 100 + (i - (P100 - 1));
    }
    auto loss100 = cross_entropy_loss(flat_l100, targets100, "mean");
    std::cout << "Loss computed: " << static_cast<const float*>(loss100->data_ptr())[0] << "\n";

    std::cout << "Running 100-token Backward Pass (24 layers autograd + ConcatBackward)...\n";
    auto t2 = std::chrono::high_resolution_clock::now();
    loss100->backward();
    auto t3 = std::chrono::high_resolution_clock::now();
    double bwd_100_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

    double total_100_ms = fwd_100_ms + bwd_100_ms;

    std::cout << "\n========================================================================\n";
    std::cout << "  100-TOKEN MEASURED HARDWARE LATENCY RESULTS (Host CPU i5-7200U)       \n";
    std::cout << "========================================================================\n";
    std::cout << "  Sequence Length:        100 tokens (64 vision + 36 text)\n";
    std::cout << "  Forward Pass Latency:   " << std::fixed << std::setprecision(2) << fwd_100_ms << " ms (" << fwd_100_ms / 1000.0 << " s)\n";
    std::cout << "  Backward Pass Latency:  " << std::fixed << std::setprecision(2) << bwd_100_ms << " ms (" << bwd_100_ms / 1000.0 << " s)\n";
    std::cout << "  Full Training Step:     " << std::fixed << std::setprecision(2) << total_100_ms << " ms (" << total_100_ms / 1000.0 << " s)\n";
    std::cout << "========================================================================\n";
    return 0;
}
