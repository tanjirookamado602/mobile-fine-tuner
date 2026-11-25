#include "finetune_ops/core/tensor.h"
#include "finetune_ops/core/ops.h"
#include "finetune_ops/graph/auto_model.h"
#include "finetune_ops/core/quant_ops.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <malloc.h>
#include <unistd.h>

using namespace ops;

double get_rss_mb() {
    std::ifstream statm("/proc/self/statm");
    if (!statm.is_open()) return 0.0;
    long total_pages = 0, resident_pages = 0;
    statm >> total_pages >> resident_pages;
    long page_size = sysconf(_SC_PAGESIZE);
    return (resident_pages * page_size) / (1024.0 * 1024.0);
}

int main(int argc, char** argv) {
    std::cout << "========================================================================\n";
    std::cout << "  MobileFineTuner: Peak RSS & Memory Breakdown Diagnostic Tool         \n";
    std::cout << "========================================================================\n";

    double rss0 = get_rss_mb();
    std::cout << "[Stage 0: Process Baseline] RSS = " << std::fixed << std::setprecision(2) << rss0 << " MB\n";

    // 1. Load Model
    std::string model_dir = "models/Qwen2.5-0.5B";
    AutoModelLoadOptions opts;
    opts.load_weights = true;
    auto model = AutoModelForCausalLM::from_pretrained(model_dir, opts);
    if (!model) {
        std::cerr << "Failed to load model\n";
        return 1;
    }
    auto qwen = model->qwen();
    qwen->freeze_base();

    double rss_model = get_rss_mb();
    std::cout << "[Stage 1: Qwen2.5-0.5B Weights in Memory] RSS = " << rss_model 
              << " MB (Delta = " << (rss_model - rss0) << " MB)\n";

    // 2. Initialize Projector (2-layer GELU MLP)
    int in_dim = 768, out_dim = 896, num_tokens = 64;
    auto ln_w = ones({in_dim}, kFloat32, kCPU);
    auto ln_b = zeros({in_dim}, kFloat32, kCPU);
    auto W1 = zeros({out_dim, in_dim}, kFloat32, kCPU);
    auto b1 = zeros({out_dim}, kFloat32, kCPU);
    auto W2 = zeros({out_dim, out_dim}, kFloat32, kCPU);
    auto b2 = zeros({out_dim}, kFloat32, kCPU);
    auto pos_embed = zeros({1, num_tokens, out_dim}, kFloat32, kCPU);

    ln_w->set_requires_grad(true);
    ln_b->set_requires_grad(true);
    W1->set_requires_grad(true);
    b1->set_requires_grad(true);
    W2->set_requires_grad(true);
    b2->set_requires_grad(true);
    pos_embed->set_requires_grad(true);

    double rss_proj = get_rss_mb();
    std::cout << "[Stage 2: Trainable Projector Initialized] RSS = " << rss_proj 
              << " MB (Delta = " << (rss_proj - rss_model) << " MB)\n";

    // 3. Test Full Sequence (64 vision tokens + 35 text tokens = 99 tokens)
    std::cout << "\n--- Testing Baseline Sequence (99 Tokens: 64 prefix + 35 text) ---\n";
    {
        int seq_text = 35;
        auto v_in = zeros({1, num_tokens, in_dim}, kFloat32, kCPU);
        auto ln_out = layer_norm(v_in, ln_w, ln_b);
        auto h1 = add(matmul_rhs_T(ln_out, W1), b1);
        auto act = gelu(h1);
        auto h2 = add(matmul_rhs_T(act, W2), b2);
        auto scaled = mul(h2, 0.01548f);
        auto p_tokens = add(scaled, pos_embed);

        auto t_in = zeros({1, seq_text, out_dim}, kFloat32, kCPU);
        auto spliced = concat(std::vector<TensorPtr>{p_tokens, t_in}, 1);

        double rss_before_fwd = get_rss_mb();
        auto h = qwen->forward_hidden_from_embeds(spliced);
        double rss_after_fwd = get_rss_mb();
        std::cout << "  [Fwd Complete] RSS = " << rss_after_fwd 
                  << " MB (Activations Delta = " << (rss_after_fwd - rss_before_fwd) << " MB)\n";

        // Full LM Head Logits Materialization
        int64_t total_seq = spliced->shape()[1];
        int64_t pred_seq = total_seq - 1;
        auto logits = qwen->lm_head(h);
        auto flat_l = reshape(logits->slice(1, 0, pred_seq, 1), {pred_seq, 151936});
        auto targets = zeros({pred_seq}, DType::kInt32, kCPU);
        auto loss = cross_entropy_loss(flat_l, targets, "mean");
        double rss_after_loss = get_rss_mb();
        std::cout << "  [Loss & Logits Materialized] RSS = " << rss_after_loss 
                  << " MB (Logits Delta = " << (rss_after_loss - rss_after_fwd) << " MB)\n";

        // Backward Pass
        loss->backward();
        double rss_after_bwd = get_rss_mb();
        std::cout << "  [Bwd Complete] RSS = " << rss_after_bwd 
                  << " MB (Gradients Delta = " << (rss_after_bwd - rss_after_loss) << " MB)\n";

        // Zero grad & malloc_trim
        W1->zero_grad(); b1->zero_grad();
        W2->zero_grad(); b2->zero_grad();
        pos_embed->zero_grad(); ln_w->zero_grad(); ln_b->zero_grad();
        malloc_trim(0);
        double rss_after_trim = get_rss_mb();
        std::cout << "  [Post-Step Cleanup & Trim] RSS = " << rss_after_trim << " MB\n";
    }

    // 4. Test Shorter Sequence (16 vision tokens + 16 text tokens = 32 tokens)
    std::cout << "\n--- Testing Shorter Sequence (32 Tokens: 16 prefix + 16 text) ---\n";
    {
        int short_tokens = 16;
        int seq_text = 16;
        auto pos_embed_short = zeros({1, short_tokens, out_dim}, kFloat32, kCPU);
        pos_embed_short->set_requires_grad(true);

        auto v_in = zeros({1, short_tokens, in_dim}, kFloat32, kCPU);
        auto ln_out = layer_norm(v_in, ln_w, ln_b);
        auto h1 = add(matmul_rhs_T(ln_out, W1), b1);
        auto act = gelu(h1);
        auto h2 = add(matmul_rhs_T(act, W2), b2);
        auto scaled = mul(h2, 0.01548f);
        auto p_tokens = add(scaled, pos_embed_short);

        auto t_in = zeros({1, seq_text, out_dim}, kFloat32, kCPU);
        auto spliced = concat(std::vector<TensorPtr>{p_tokens, t_in}, 1);

        double rss_before_fwd = get_rss_mb();
        auto h = qwen->forward_hidden_from_embeds(spliced);
        double rss_after_fwd = get_rss_mb();
        std::cout << "  [Fwd Complete (32 toks)] RSS = " << rss_after_fwd 
                  << " MB (Activations Delta = " << (rss_after_fwd - rss_before_fwd) << " MB)\n";

        int64_t total_seq = spliced->shape()[1];
        int64_t pred_seq = total_seq - 1;
        auto logits = qwen->lm_head(h);
        auto flat_l = reshape(logits->slice(1, 0, pred_seq, 1), {pred_seq, 151936});
        auto targets = zeros({pred_seq}, DType::kInt32, kCPU);
        auto loss = cross_entropy_loss(flat_l, targets, "mean");
        double rss_after_loss = get_rss_mb();
        std::cout << "  [Loss Complete (32 toks)] RSS = " << rss_after_loss << " MB\n";

        loss->backward();
        double rss_after_bwd = get_rss_mb();
        std::cout << "  [Bwd Complete (32 toks)] RSS = " << rss_after_bwd << " MB\n";

        W1->zero_grad(); b1->zero_grad();
        W2->zero_grad(); b2->zero_grad();
        pos_embed_short->zero_grad(); ln_w->zero_grad(); ln_b->zero_grad();
        malloc_trim(0);
        double rss_after_trim = get_rss_mb();
        std::cout << "  [Post-Step Cleanup & Trim (32 toks)] RSS = " << rss_after_trim << " MB\n";
    }

    // 5. Test with Activation Checkpointing Enabled
    std::cout << "\n--- Testing with Activation Checkpointing Enabled (99 Tokens) ---\n";
    {
        qwen->set_activation_checkpointing(true);
        int seq_text = 35;
        auto v_in = zeros({1, num_tokens, in_dim}, kFloat32, kCPU);
        auto ln_out = layer_norm(v_in, ln_w, ln_b);
        auto h1 = add(matmul_rhs_T(ln_out, W1), b1);
        auto act = gelu(h1);
        auto h2 = add(matmul_rhs_T(act, W2), b2);
        auto scaled = mul(h2, 0.01548f);
        auto p_tokens = add(scaled, pos_embed);

        auto t_in = zeros({1, seq_text, out_dim}, kFloat32, kCPU);
        auto spliced = concat(std::vector<TensorPtr>{p_tokens, t_in}, 1);

        double rss_before_fwd = get_rss_mb();
        auto h = qwen->forward_hidden_from_embeds(spliced);
        double rss_after_fwd = get_rss_mb();
        std::cout << "  [Fwd Complete (ActCkpt)] RSS = " << rss_after_fwd 
                  << " MB (Activations Delta = " << (rss_after_fwd - rss_before_fwd) << " MB)\n";

        int64_t total_seq = spliced->shape()[1];
        int64_t pred_seq = total_seq - 1;
        auto logits = qwen->lm_head(h);
        auto flat_l = reshape(logits->slice(1, 0, pred_seq, 1), {pred_seq, 151936});
        auto targets = zeros({pred_seq}, DType::kInt32, kCPU);
        auto loss = cross_entropy_loss(flat_l, targets, "mean");
        loss->backward();
        double rss_after_bwd = get_rss_mb();
        std::cout << "  [Bwd Complete (ActCkpt)] RSS = " << rss_after_bwd << " MB\n";

        W1->zero_grad(); b1->zero_grad();
        W2->zero_grad(); b2->zero_grad();
        pos_embed->zero_grad(); ln_w->zero_grad(); ln_b->zero_grad();
        malloc_trim(0);
        double rss_after_trim = get_rss_mb();
        std::cout << "  [Post-Step Cleanup (ActCkpt)] RSS = " << rss_after_trim << " MB\n";
    }

    std::cout << "\n========================================================================\n";
    std::cout << "Memory Breakdown Diagnostic Completed.\n";
    std::cout << "========================================================================\n";
    return 0;
}
