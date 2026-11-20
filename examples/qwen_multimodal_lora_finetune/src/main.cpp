/**
 * @file main.cpp
 * @brief High-efficiency Qwen Multimodal LoRA fine-tuning CLI.
 * Features: lr=1e-5 default, selected_token_lm_loss, RSS & NDJSON metrics, FP32/BF16/FP16 precision benchmarking.
 */

#include "qwen_multimodal_model.h"
#include "multimodal_dataset.h"
#include "multimodal_trainer.h"
#include "finetune_ops/core/ops.h"
#include "finetune_ops/core/lm_loss.h"
#include "finetune_ops/core/dtype.h"
#include "finetune_ops/core/memory_manager.h"
#include "finetune_ops/energy/power_monitor.h"

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <cmath>

using namespace ops;

struct Args {
    std::string model_dir = "";
    std::string data_dir = "";
    std::string jsonl_train = "";
    std::string features_dir = "";
    std::string output_dir = "outputs/qwen_multimodal";
    float lr = 1e-5f; // Standard fine-tuning learning rate (1e-5)
    float max_grad_norm = 1.0f;
    int max_steps = 50;
    int batch_size = 1;
    int num_layers = 4;
    int hidden_size = 256;
    int intermediate_size = 512;
    int vocab_size = 1000;
    int vision_dim = 768; // ViT patch embedding dimension
    int num_vision_tokens = 64;
    int prompt_len = 16;
    int response_len = 16;
    int lora_r = 16;
    float lora_alpha = 32.0f;
    float lora_dropout = 0.05f;
    std::string precision = "fp32"; // fp32, bf16, fp16
    bool benchmark_precisions = false;
    bool synthetic_smoke = false;
    uint64_t seed = 42;
};

static void parse_args(int argc, char** argv, Args& args) {
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto get = [&](const char* name) {
            if (i + 1 >= argc) throw std::runtime_error(std::string("Missing value for ") + name);
            if (k != name) throw std::runtime_error(std::string("Unexpected argument ") + k);
            return std::string(argv[++i]);
        };

        if (k == "--model_dir") args.model_dir = get("--model_dir");
        else if (k == "--data_dir") args.data_dir = get("--data_dir");
        else if (k == "--jsonl_train") args.jsonl_train = get("--jsonl_train");
        else if (k == "--features_dir") args.features_dir = get("--features_dir");
        else if (k == "--output_dir") args.output_dir = get("--output_dir");
        else if (k == "--lr") args.lr = std::stof(get("--lr"));
        else if (k == "--max_grad_norm") args.max_grad_norm = std::stof(get("--max_grad_norm"));
        else if (k == "--max_steps") args.max_steps = std::stoi(get("--max_steps"));
        else if (k == "--batch_size") args.batch_size = std::stoi(get("--batch_size"));
        else if (k == "--num_layers") args.num_layers = std::stoi(get("--num_layers"));
        else if (k == "--hidden_size") args.hidden_size = std::stoi(get("--hidden_size"));
        else if (k == "--intermediate_size") args.intermediate_size = std::stoi(get("--intermediate_size"));
        else if (k == "--vocab_size") args.vocab_size = std::stoi(get("--vocab_size"));
        else if (k == "--vision_dim") args.vision_dim = std::stoi(get("--vision_dim"));
        else if (k == "--num_vision_tokens") args.num_vision_tokens = std::stoi(get("--num_vision_tokens"));
        else if (k == "--lora_r") args.lora_r = std::stoi(get("--lora_r"));
        else if (k == "--lora_alpha") args.lora_alpha = std::stof(get("--lora_alpha"));
        else if (k == "--precision") args.precision = get("--precision");
        else if (k == "--benchmark_precisions") args.benchmark_precisions = true;
        else if (k == "--synthetic_smoke") args.synthetic_smoke = true;
        else if (k == "--seed") args.seed = std::stoull(get("--seed"));
    }
}

static int run_synthetic_smoke(const Args& args) {
    std::cout << "\n=========================================================================\n";
    std::cout << "        QWEN MULTIMODAL LoRA SYNTHETIC SMOKE VERIFICATION               \n";
    std::cout << "=========================================================================\n";

    QwenMultimodalConfig mcfg;
    mcfg.text_config.vocab_size = 64;
    mcfg.text_config.hidden_size = 32;
    mcfg.text_config.intermediate_size = 64;
    mcfg.text_config.num_hidden_layers = 2;
    mcfg.text_config.num_attention_heads = 4;
    mcfg.text_config.num_key_value_heads = 2;
    mcfg.text_config.max_position_embeddings = 128;
    mcfg.vision_dim = 32;
    mcfg.num_vision_tokens = 8;
    mcfg.precision = "fp32";
    mcfg.seed = args.seed;

    QwenMultimodalModel model(mcfg);
    model.init_lora(args.lora_r, args.lora_alpha, args.lora_dropout, false, args.seed);
    model.freeze_base();

    auto trainables = model.trainable_parameters();
    if (trainables.empty()) {
        std::cerr << "[FAIL] No trainable parameters found!\n";
        return 1;
    }

    std::vector<std::vector<float>> initial_weights;
    for (const auto& p : trainables) {
        const float* d = p->data<float>();
        initial_weights.emplace_back(d, d + p->numel());
    }

    MultimodalDataset dataset;
    dataset.generate_benchmark_dataset(5, mcfg.vision_dim, mcfg.num_vision_tokens, 8, 8, mcfg.text_config.vocab_size, args.seed);

    MultimodalTrainerConfig tcfg;
    tcfg.learning_rate = 1e-3f;
    tcfg.max_grad_norm = 1.0f;
    tcfg.precision = "fp32";
    MultimodalTrainer trainer(model, tcfg);

    int smoke_steps = 5;
    for (int step = 0; step < smoke_steps; ++step) {
        auto batch = dataset.get_batch(step, 1);
        auto m = trainer.train_step(batch);
        std::cout << "[Smoke Step " << (step + 1) << "/" << smoke_steps << "] "
                  << "Loss: " << std::fixed << std::setprecision(4) << m.loss
                  << " | GradNorm: " << std::fixed << std::setprecision(4) << m.grad_norm
                  << " | RSS: " << std::fixed << std::setprecision(1) << m.rss_mb << " MB"
                  << " | Latency: " << std::fixed << std::setprecision(1) << m.step_time_ms << " ms"
                  << std::endl;
        if (!std::isfinite(m.loss) || m.grad_norm <= 0.0f) {
            std::cerr << "[FAIL] Non-finite loss or zero gradient norm at step " << step << "\n";
            return 1;
        }
    }

    // Check parameter delta
    float max_delta = 0.0f;
    for (size_t i = 0; i < trainables.size(); ++i) {
        const float* d = trainables[i]->data<float>();
        for (int64_t j = 0; j < trainables[i]->numel(); ++j) {
            max_delta = std::max(max_delta, std::fabs(d[j] - initial_weights[i][j]));
        }
    }
    std::cout << "Max parameter delta after smoke updates: " << std::defaultfloat << std::setprecision(6) << max_delta << std::endl;
    if (max_delta > 0.0f) {
        std::cout << ">>> [PASS] Qwen Multimodal LoRA synthetic smoke passed!\n";
        return 0;
    } else {
        std::cerr << ">>> [FAIL] Weights did not update!\n";
        return 1;
    }
}

static void run_precision_benchmark(const Args& args) {
    std::cout << "\n=========================================================================\n";
    std::cout << "  EMPIRICAL PRECISION LOSS & SYSTEM EFFICIENCY STUDY (FP32 vs BF16 vs FP16)\n";
    std::cout << "=========================================================================\n";

    const std::vector<std::string> precisions = {"fp32", "bf16", "fp16"};
    struct Result {
        std::string precision;
        float start_loss = 0.0f;
        float final_loss = 0.0f;
        float delta_loss = 0.0f;
        float rel_grad_err = 0.0f;
        double peak_rss_mb = 0.0f;
        double avg_step_time_ms = 0.0f;
        double tokens_per_sec = 0.0f;
    };
    std::vector<Result> results;

    std::vector<float> fp32_reference_gradients;

    for (const auto& prec : precisions) {
        std::cout << "\n>>> Benchmarking Precision Mode: " << prec << " (lr=" << args.lr << ")\n";

        QwenMultimodalConfig mcfg;
        mcfg.text_config.vocab_size = args.vocab_size;
        mcfg.text_config.hidden_size = args.hidden_size;
        mcfg.text_config.intermediate_size = args.intermediate_size;
        mcfg.text_config.num_hidden_layers = args.num_layers;
        mcfg.text_config.num_attention_heads = 4;
        mcfg.text_config.num_key_value_heads = 2;
        mcfg.text_config.max_position_embeddings = 512;
        mcfg.vision_dim = args.vision_dim;
        mcfg.num_vision_tokens = args.num_vision_tokens;
        mcfg.precision = prec;
        mcfg.seed = args.seed;

        QwenMultimodalModel model(mcfg);
        model.init_lora(args.lora_r, args.lora_alpha, args.lora_dropout, false, args.seed);
        model.freeze_base();

        // Apply precision quantization to trainable weights if bf16/fp16
        if (prec == "bf16") {
            model.set_proj_w1(ops::cast(ops::cast(model.proj_w1(), kBFloat16), kFloat32));
            if (model.proj_w2()) {
                model.set_proj_w2(ops::cast(ops::cast(model.proj_w2(), kBFloat16), kFloat32));
            }
        } else if (prec == "fp16") {
            model.set_proj_w1(ops::cast(ops::cast(model.proj_w1(), kFloat16), kFloat32));
            if (model.proj_w2()) {
                model.set_proj_w2(ops::cast(ops::cast(model.proj_w2(), kFloat16), kFloat32));
            }
        }

        MultimodalDataset dataset;
        if (!args.jsonl_train.empty() && std::filesystem::exists(args.jsonl_train)) {
            dataset.load_jsonl(args.jsonl_train, args.features_dir, args.vision_dim);
        } else {
            dataset.generate_benchmark_dataset(args.max_steps,
                                              args.vision_dim,
                                              args.num_vision_tokens,
                                              args.prompt_len,
                                              args.response_len,
                                              mcfg.text_config.vocab_size,
                                              args.seed);
        }

        std::filesystem::create_directories(args.output_dir + "/" + prec);
        MultimodalTrainerConfig tcfg;
        tcfg.learning_rate = args.lr;
        tcfg.max_grad_norm = args.max_grad_norm;
        tcfg.precision = prec;
        tcfg.metrics_ndjson_path = args.output_dir + "/" + prec + "/metrics.ndjson";
        tcfg.rss_csv_path = args.output_dir + "/" + prec + "/rss.csv";
        tcfg.enable_rss_logging = true;

        MultimodalTrainer trainer(model, tcfg);

        Result res;
        res.precision = prec;
        double total_ms = 0.0;
        int64_t total_tokens = 0;

        for (int step = 0; step < args.max_steps; ++step) {
            auto batch = dataset.get_batch(step % dataset.size(), 1);
            auto m = trainer.train_step(batch);

            if (step == 0) {
                res.start_loss = m.loss;
                // Capture step-0 gradients for reference comparison
                if (prec == "fp32") {
                    fp32_reference_gradients.clear();
                    for (const auto& p : model.trainable_parameters()) {
                        if (p && p->grad()) {
                            const float* g = p->grad()->data<float>();
                            fp32_reference_gradients.insert(fp32_reference_gradients.end(), g, g + p->grad()->numel());
                        }
                    }
                } else if (!fp32_reference_gradients.empty()) {
                    // Compute relative error against FP32 ground truth
                    std::vector<float> cur_grads;
                    for (const auto& p : model.trainable_parameters()) {
                        if (p && p->grad()) {
                            const float* g = p->grad()->data<float>();
                            cur_grads.insert(cur_grads.end(), g, g + p->grad()->numel());
                        }
                    }
                    if (cur_grads.size() == fp32_reference_gradients.size()) {
                        float diff_sq = 0.0f;
                        float ref_sq = 0.0f;
                        for (size_t i = 0; i < cur_grads.size(); ++i) {
                            float d = cur_grads[i] - fp32_reference_gradients[i];
                            diff_sq += d * d;
                            ref_sq += fp32_reference_gradients[i] * fp32_reference_gradients[i];
                        }
                        res.rel_grad_err = std::sqrt(diff_sq) / (std::sqrt(ref_sq) + 1e-8f);
                    }
                }
            }
            res.final_loss = m.loss;
            res.peak_rss_mb = std::max(res.peak_rss_mb, m.rss_mb);
            total_ms += m.step_time_ms;
            total_tokens += m.tokens;

            if ((step + 1) % 10 == 0 || step == 0 || step == args.max_steps - 1) {
                std::cout << "  [Step " << std::setw(3) << (step + 1) << "/" << args.max_steps << "]"
                          << " Loss: " << std::fixed << std::setprecision(4) << m.loss
                          << " | GradNorm: " << std::fixed << std::setprecision(4) << m.grad_norm
                          << " | RSS: " << std::fixed << std::setprecision(1) << m.rss_mb << " MB"
                          << " | Time: " << std::fixed << std::setprecision(1) << m.step_time_ms << " ms"
                          << std::endl;
            }
        }

        res.avg_step_time_ms = total_ms / args.max_steps;
        res.tokens_per_sec = (total_tokens * 1000.0) / (total_ms + 1e-5);
        results.push_back(res);
    }

    // Compute relative loss differences against FP32 baseline
    float base_fp32_loss = results.empty() ? 0.0f : results.front().final_loss;
    for (auto& r : results) {
        r.delta_loss = std::fabs(r.final_loss - base_fp32_loss);
    }

    std::cout << "\n=========================================================================\n";
    std::cout << "                         EMPIRICAL COMPARISON TABLE                      \n";
    std::cout << "=========================================================================\n";
    std::cout << std::left << std::setw(12) << "Precision"
              << std::setw(14) << "Final Loss"
              << std::setw(14) << "Loss Error"
              << std::setw(16) << "Rel Grad Err"
              << std::setw(14) << "Peak RSS"
              << std::setw(16) << "Step Latency"
              << std::setw(14) << "Throughput" << "\n";
    std::cout << "-------------------------------------------------------------------------\n";

    for (const auto& r : results) {
        std::cout << std::left << std::setw(12) << r.precision
                  << std::fixed << std::setprecision(4) << std::setw(14) << r.final_loss
                  << std::fixed << std::setprecision(5) << std::setw(14) << r.delta_loss
                  << std::fixed << std::setprecision(4) << std::setw(16) << (r.precision == "fp32" ? 0.0f : r.rel_grad_err)
                  << std::fixed << std::setprecision(1) << std::setw(14) << (std::to_string((int)r.peak_rss_mb) + " MB")
                  << std::fixed << std::setprecision(2) << std::setw(16) << (std::to_string(r.avg_step_time_ms).substr(0,5) + " ms")
                  << std::fixed << std::setprecision(1) << std::setw(14) << (std::to_string(r.tokens_per_sec).substr(0,6) + " t/s")
                  << "\n";
    }
    std::cout << "=========================================================================\n";

    // Write markdown report table for research paper documentation
    std::ofstream rpt(args.output_dir + "/PRECISION_STUDY_REPORT.md");
    if (rpt.is_open()) {
        rpt << "# Empirical Precision Loss & System Efficiency Study\n\n";
        rpt << "| Precision | Final Loss | Loss Error (|&Delta;L|) | Rel Grad Error | Peak RSS (MB) | Step Latency (ms) | Throughput (tok/s) |\n";
        rpt << "|:---|:---:|:---:|:---:|:---:|:---:|:---:|\n";
        for (const auto& r : results) {
            rpt << "| **" << r.precision << "** | "
                << std::fixed << std::setprecision(4) << r.final_loss << " | "
                << std::fixed << std::setprecision(5) << r.delta_loss << " | "
                << (r.precision == "fp32" ? "0.0000 (Base)" : std::to_string(r.rel_grad_err).substr(0,6)) << " | "
                << std::fixed << std::setprecision(1) << r.peak_rss_mb << " MB | "
                << std::fixed << std::setprecision(2) << r.avg_step_time_ms << " ms | "
                << std::fixed << std::setprecision(1) << r.tokens_per_sec << " t/s |\n";
        }
        rpt << "\n*Generated automatically by MobileFineTuner Qwen Multimodal LoRA Benchmark.* \n";
        std::cout << "Wrote research benchmark report to: " << args.output_dir << "/PRECISION_STUDY_REPORT.md" << std::endl;
    }
}

int main(int argc, char** argv) {
    Args args;
    try {
        parse_args(argc, argv, args);
    } catch (const std::exception& e) {
        std::cerr << "Error parsing arguments: " << e.what() << "\n";
        std::cerr << "Usage: train_multimodal [--model_dir DIR] [--jsonl_train FILE] [--lr 1e-5] [--precision fp32|bf16|fp16] [--benchmark_precisions]\n";
        return 1;
    }

    if (args.synthetic_smoke) {
        return run_synthetic_smoke(args);
    }

    if (args.benchmark_precisions) {
        run_precision_benchmark(args);
        return 0;
    }

    std::cout << "\n=========================================================================\n";
    std::cout << "  QWEN MULTIMODAL LoRA FINE-TUNING PIPELINE (MobileFineTuner Examples)   \n";
    std::cout << "=========================================================================\n";
    std::cout << "  Learning Rate : " << args.lr << "\n";
    std::cout << "  Precision     : " << args.precision << "\n";
    std::cout << "  Max Steps     : " << args.max_steps << "\n";
    std::cout << "  LoRA Rank     : " << args.lora_r << " (alpha=" << args.lora_alpha << ")\n";
    std::cout << "  Vision Dim    : " << args.vision_dim << " (" << args.num_vision_tokens << " patch tokens)\n";
    std::cout << "  Output Dir    : " << args.output_dir << "\n";
    std::cout << "-------------------------------------------------------------------------\n";

    QwenMultimodalConfig mcfg;
    if (!args.model_dir.empty() && std::filesystem::exists(args.model_dir + "/config.json")) {
        std::cout << "Loading model configuration from: " << args.model_dir << "/config.json\n";
        mcfg = QwenMultimodalConfig::from_pretrained(args.model_dir + "/config.json", args.vision_dim, args.precision);
    } else {
        std::cout << "Model architecture: hidden=" << args.hidden_size
                  << ", layers=" << args.num_layers
                  << ", intermediate=" << args.intermediate_size
                  << ", vocab=" << args.vocab_size << "\n";
        mcfg.text_config.vocab_size = args.vocab_size;
        mcfg.text_config.hidden_size = args.hidden_size;
        mcfg.text_config.intermediate_size = args.intermediate_size;
        mcfg.text_config.num_hidden_layers = args.num_layers;
        mcfg.text_config.num_attention_heads = 4;
        mcfg.text_config.num_key_value_heads = 2;
        mcfg.text_config.max_position_embeddings = 2048;
        mcfg.vision_dim = args.vision_dim;
        mcfg.num_vision_tokens = args.num_vision_tokens;
        mcfg.precision = args.precision;
    }
    mcfg.seed = args.seed;

    std::cout << "Initializing QwenMultimodalModel..." << std::endl;
    QwenMultimodalModel model(mcfg);
    model.init_lora(args.lora_r, args.lora_alpha, args.lora_dropout, false, args.seed);
    model.freeze_base();

    std::cout << "Trainable parameters count: " << model.trainable_parameters().size() << " tensors\n";

    MultimodalDataset dataset;
    if (!args.jsonl_train.empty() && std::filesystem::exists(args.jsonl_train)) {
        std::cout << "Loading dataset from JSONL: " << args.jsonl_train << std::endl;
        dataset.load_jsonl(args.jsonl_train, args.features_dir, args.vision_dim);
    } else {
        std::cout << "Generating authentic benchmark dataset (" << args.max_steps << " multimodal samples)...\n";
        dataset.generate_benchmark_dataset(args.max_steps,
                                          args.vision_dim,
                                          args.num_vision_tokens,
                                          args.prompt_len,
                                          args.response_len,
                                          mcfg.text_config.vocab_size,
                                          args.seed);
    }

    std::filesystem::create_directories(args.output_dir);
    MultimodalTrainerConfig tcfg;
    tcfg.learning_rate = args.lr;
    tcfg.max_grad_norm = args.max_grad_norm;
    tcfg.precision = args.precision;
    tcfg.metrics_ndjson_path = args.output_dir + "/metrics.ndjson";
    tcfg.rss_csv_path = args.output_dir + "/rss.csv";
    tcfg.enable_rss_logging = true;

    MultimodalTrainer trainer(model, tcfg);

    std::cout << "\nStarting training loop with lr=" << args.lr << "...\n";
    auto t_start = std::chrono::steady_clock::now();

    for (int step = 0; step < args.max_steps; ++step) {
        auto batch = dataset.get_batch(step % dataset.size(), args.batch_size);
        auto m = trainer.train_step(batch);

        if ((step + 1) % 10 == 0 || step == 0 || step == args.max_steps - 1) {
            std::cout << "[Step " << std::setw(3) << m.step << "/" << args.max_steps << "] "
                      << "Loss: " << std::fixed << std::setprecision(4) << m.loss
                      << " | PPL: " << std::fixed << std::setprecision(2) << m.perplexity
                      << " | GradNorm: " << std::fixed << std::setprecision(4) << m.grad_norm
                      << " | RSS: " << std::fixed << std::setprecision(1) << m.rss_mb << " MB"
                      << " | Latency: " << std::fixed << std::setprecision(1) << m.step_time_ms << " ms"
                      << std::endl;
        }
    }

    auto t_end = std::chrono::steady_clock::now();
    double total_sec = std::chrono::duration<double>(t_end - t_start).count();

    std::cout << "\n-------------------------------------------------------------------------\n";
    std::cout << "Training session completed in " << std::fixed << std::setprecision(2) << total_sec << " seconds.\n";
    std::cout << "Emitted structured telemetry: " << tcfg.metrics_ndjson_path << "\n";
    std::cout << "Emitted memory stream:        " << tcfg.rss_csv_path << "\n";
    std::cout << "=========================================================================\n";

    return 0;
}
