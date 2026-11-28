#include "finetune_ops/core/tensor.h"
#include "finetune_ops/core/ops.h"
#include "finetune_ops/core/autograd_engine.h"
#include "finetune_ops/core/memory_manager.h"
#include "finetune_ops/core/quant_io.h"
#include "finetune_ops/graph/auto_model.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iomanip>
#include <cmath>
#include <cstring>
#include <chrono>
#include <regex>
#include <filesystem>
#if defined(__GLIBC__)
#include <malloc.h>
#endif

using namespace ops;

TensorPtr load_bin(const std::string& path, const std::vector<int64_t>& shape) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        throw std::runtime_error("Cannot open file: " + path);
    }
    int64_t total = 1;
    for (auto d : shape) total *= d;
    auto t = zeros(shape, kFloat32, kCPU);
    f.read(reinterpret_cast<char*>(t->data_ptr()), total * sizeof(float));
    return t;
}

void save_bin(const std::string& path, const TensorPtr& t) {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(t->data_ptr()), t->numel() * sizeof(float));
}

double l2_norm(const TensorPtr& t) {
    const float* p = static_cast<const float*>(t->data_ptr());
    double s = 0.0;
    for (int64_t i = 0; i < t->numel(); ++i) s += static_cast<double>(p[i]) * p[i];
    return std::sqrt(s);
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

class SimpleAdam {
public:
    float lr_;
    float beta1_ = 0.9f;
    float beta2_ = 0.999f;
    float eps_ = 1e-8f;
    size_t step_ = 0;
    std::unordered_map<TensorPtr, std::pair<std::vector<float>, std::vector<float>>> state_;

    explicit SimpleAdam(float lr = 1e-3f) : lr_(lr) {}

    void step(const std::vector<TensorPtr>& params) {
        step_++;
        float bc1 = 1.0f - std::pow(beta1_, step_);
        float bc2 = 1.0f - std::pow(beta2_, step_);

        for (auto& p : params) {
            if (!p || !p->grad()) continue;
            size_t n = p->numel();
            if (state_.find(p) == state_.end()) {
                state_[p] = {std::vector<float>(n, 0.0f), std::vector<float>(n, 0.0f)};
            }
            auto& [m, v] = state_[p];
            float* p_ptr = static_cast<float*>(p->data_ptr());
            const float* g_ptr = static_cast<const float*>(p->grad()->data_ptr());

            for (size_t i = 0; i < n; ++i) {
                float g = g_ptr[i];
                m[i] = beta1_ * m[i] + (1.0f - beta1_) * g;
                v[i] = beta2_ * v[i] + (1.0f - beta2_) * g * g;
                float m_hat = m[i] / bc1;
                float v_hat = v[i] / bc2;
                p_ptr[i] -= lr_ * m_hat / (std::sqrt(v_hat) + eps_);
            }
        }
    }
};

static std::pair<double, double> read_proc_mem_mb() {
    std::ifstream f("/proc/self/status");
    std::string line;
    double rss = 0.0, hwm = 0.0;
    while (std::getline(f, line)) {
        if (line.rfind("VmRSS:", 0) == 0) {
            std::istringstream iss(line.substr(6));
            double kb = 0; iss >> kb; rss = kb / 1024.0;
        } else if (line.rfind("VmHWM:", 0) == 0) {
            std::istringstream iss(line.substr(6));
            double kb = 0; iss >> kb; hwm = kb / 1024.0;
        }
    }
    return {rss, hwm};
}

int main(int argc, char** argv) {
    bool use_q8 = false;
    int max_steps = 5;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--q8") use_q8 = true;
        else if (a == "--steps" && i + 1 < argc) max_steps = std::stoi(argv[++i]);
    }
    std::string data_dir = "data/parity_test";
    std::string dump_dir = data_dir + (use_q8 ? "/cpp_dump_q8" : "/cpp_dump");
    std::filesystem::create_directories(dump_dir);
    std::string model_dir = "models/Qwen2.5-0.5B";
    std::string q8_path = "models/qwen2.5-0.5b-q8_0.q8";

    std::cout << "========================================================================\n";
    std::cout << "  MobileFineTuner: 5-Step Exact Parity Verification (C++ Runner)       \n";
    std::cout << "========================================================================\n";

    auto ln_w = load_bin(data_dir + "/ln_w.bin", {768});
    auto ln_b = load_bin(data_dir + "/ln_b.bin", {768});
    auto W1 = load_bin(data_dir + "/W1.bin", {896, 768});
    auto b1 = load_bin(data_dir + "/b1.bin", {896});
    auto W2 = load_bin(data_dir + "/W2.bin", {896, 896});
    auto b2 = load_bin(data_dir + "/b2.bin", {896});
    auto scale = full({1}, 0.4633f / std::sqrt(896.0f), kFloat32, kCPU);
    auto pos_embed = load_bin(data_dir + "/pos_embed.bin", {1, 64, 896});

    ln_w->set_requires_grad(true);
    ln_b->set_requires_grad(true);
    W1->set_requires_grad(true);
    b1->set_requires_grad(true);
    W2->set_requires_grad(true);
    b2->set_requires_grad(true);
    scale->set_requires_grad(true);
    pos_embed->set_requires_grad(true);

    std::vector<std::pair<std::string, TensorPtr>> named_params = {
        {"ln_w", ln_w}, {"ln_b", ln_b},
        {"W1", W1}, {"b1", b1},
        {"W2", W2}, {"b2", b2},
        {"scale", scale}, {"pos_embed", pos_embed}
    };
    std::vector<TensorPtr> params;
    for (auto& kv : named_params) params.push_back(kv.second);

    auto v_feat = load_bin(data_dir + "/visual_features.bin", {1, 64, 768});

    std::ifstream f_in(data_dir + "/input_ids.bin", std::ios::binary);
    std::vector<int32_t> input_ids(34);
    f_in.read(reinterpret_cast<char*>(input_ids.data()), 34 * sizeof(int32_t));

    std::ifstream f_lb(data_dir + "/labels.bin", std::ios::binary);
    std::vector<int32_t> labels(34);
    f_lb.read(reinterpret_cast<char*>(labels.data()), 34 * sizeof(int32_t));

    std::shared_ptr<AutoModelForCausalLM> model;
    QwenModel* qwen = nullptr;
    if (use_q8) {
        AutoModelLoadOptions opts;
        opts.load_weights = false;
        model = AutoModelForCausalLM::from_pretrained(model_dir, opts);
        qwen = model->qwen();
        qwen->init_linear_wrappers();
        std::vector<ops::QuantTensor> tensors;
        std::string qerr;
        if (!ops::read_quant_model(q8_path, tensors, &qerr)) {
            std::cerr << "Failed to read Q8 model: " << qerr << "\n";
            return 1;
        }
        std::regex q8_name(R"(layers\.(\d+)\.(.+))");
        for (auto& t : tensors) {
            if (t.is_q8) {
                std::smatch m;
                if (!std::regex_match(t.name, m, q8_name)) continue;
                int li = std::stoi(m[1].str());
                std::string sub = m[2].str();
                auto& blk = qwen->get_block(li);
                LoRALinear* lin = nullptr;
                if (sub == "self_attn.q_proj.weight") lin = blk.q_lin.get();
                else if (sub == "self_attn.k_proj.weight") lin = blk.k_lin.get();
                else if (sub == "self_attn.v_proj.weight") lin = blk.v_lin.get();
                else if (sub == "self_attn.o_proj.weight") lin = blk.o_lin.get();
                else if (sub == "mlp.gate_proj.weight") lin = blk.gate_lin.get();
                else if (sub == "mlp.up_proj.weight") lin = blk.up_lin.get();
                else if (sub == "mlp.down_proj.weight") lin = blk.down_lin.get();
                if (lin) lin->adopt_quantized_weight(t.k, t.m, std::move(t.q8));
                t.q8.clear(); t.q8.shrink_to_fit();
            } else {
                auto ten = std::make_shared<Tensor>(t.shape, static_cast<const void*>(t.f32.data()), DType::kFloat32, kCPU);
                qwen->assign_weight(t.name, ten);
                t.f32.clear(); t.f32.shrink_to_fit();
            }
        }
        for (int li = 0; li < qwen->config().num_hidden_layers; ++li) {
            auto& blk = qwen->get_block(li);
            if (blk.q_lin) blk.q_lin->set_base_weights(nullptr, blk.q_proj_bias);
            if (blk.k_lin) blk.k_lin->set_base_weights(nullptr, blk.k_proj_bias);
            if (blk.v_lin) blk.v_lin->set_base_weights(nullptr, blk.v_proj_bias);
            if (blk.o_lin) blk.o_lin->set_base_weights(nullptr, nullptr);
            if (blk.gate_lin) blk.gate_lin->set_base_weights(nullptr, nullptr);
            if (blk.up_lin) blk.up_lin->set_base_weights(nullptr, nullptr);
            if (blk.down_lin) blk.down_lin->set_base_weights(nullptr, nullptr);
        }
        MemoryManager::instance().clear_unused_memory();
    } else {
        AutoModelLoadOptions opts;
        opts.load_weights = true;
        model = AutoModelForCausalLM::from_pretrained(model_dir, opts);
        qwen = model->qwen();
    }
    qwen->freeze_base();

    SimpleAdam optimizer(1e-3f);

    auto text_id_tensor = zeros({1, 34}, DType::kInt32, kCPU);
    std::memcpy(text_id_tensor->data_ptr(), input_ids.data(), 34 * sizeof(int32_t));
    auto text_embeds = lookup_embeds(qwen->embedding_weight(), text_id_tensor);

    std::vector<float> cpp_losses;

    for (int step = 1; step <= max_steps; ++step) {
        auto t0 = std::chrono::steady_clock::now();
        for (auto& p : params) p->zero_grad();

        auto ln_out = layer_norm(v_feat, ln_w, ln_b);
        auto h1 = add(matmul_rhs_T(ln_out, W1), b1);
        auto act = gelu(h1);
        auto h2 = add(matmul_rhs_T(act, W2), b2);
        auto scaled = mul(h2, scale);
        auto p_tokens = add(scaled, pos_embed);

        auto spliced = concat(std::vector<TensorPtr>{p_tokens, text_embeds}, 1);

        auto h = qwen->forward_hidden_from_embeds(spliced);
        auto logits = qwen->lm_head(h);

        if (step == 1) {
            if (const char* ldir = std::getenv("QWEN_DUMP_LAYERED")) {
                std::filesystem::create_directories(ldir);
                std::string ld(ldir);
                save_bin(ld + "/text_embeds.bin", text_embeds);
                save_bin(ld + "/ln_out.bin", ln_out);
                save_bin(ld + "/h1.bin", h1);
                save_bin(ld + "/gelu.bin", act);
                save_bin(ld + "/h2.bin", h2);
                save_bin(ld + "/p_tokens.bin", p_tokens);
                save_bin(ld + "/spliced.bin", spliced);
                save_bin(ld + "/final_norm.bin", h);
                save_bin(ld + "/logits.bin", logits);
            }
        }

        int64_t total_seq = spliced->shape()[1];
        int64_t pred_seq = total_seq - 1;
        auto flat_l = reshape(logits->slice(1, 0, pred_seq, 1), {pred_seq, 151936});

        auto targets = zeros({pred_seq}, DType::kInt32, kCPU);
        int32_t* tgt_ptr = static_cast<int32_t*>(targets->data_ptr());
        int64_t prefix_len = 64;
        for (int64_t i = 0; i < pred_seq; ++i) {
            if (i < prefix_len - 1) {
                tgt_ptr[i] = -100;
            } else {
                int64_t tidx = i - prefix_len + 1;
                tgt_ptr[i] = (tidx >= 0 && tidx < static_cast<int64_t>(labels.size())) ? labels[tidx] : -100;
            }
        }

        auto loss = cross_entropy_loss(flat_l, targets, "mean");
        float loss_val = static_cast<const float*>(loss->data_ptr())[0];
        cpp_losses.push_back(loss_val);

        loss->backward();

        if (step == 1) {
            std::cout << "  Step 1 Gradients (C++):\n";
            for (auto& kv : named_params) {
                auto g = kv.second->grad();
                std::cout << "    grad " << std::setw(10) << std::left << kv.first
                          << ": norm=" << std::fixed << std::setprecision(6) << l2_norm(g) << "\n";
                save_bin(dump_dir + "/" + kv.first + "_grad.bin", g);
            }
        }

        optimizer.step(params);

        if (step == 1) {
            std::cout << "  Step 1 Updated Weights (C++):\n";
            for (auto& kv : named_params) {
                std::cout << "    after step1 " << std::setw(10) << std::left << kv.first
                          << ": norm=" << std::fixed << std::setprecision(6) << l2_norm(kv.second) << "\n";
                save_bin(dump_dir + "/" + kv.first + "_step1.bin", kv.second);
            }
            std::cout << "    scale value after step1 = " << static_cast<const float*>(scale->data_ptr())[0] << "\n";
        }

        ops::autograd::Engine::instance().clear_graph();
        MemoryManager::instance().clear_unused_memory();
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        auto [rss_mb, hwm_mb] = read_proc_mem_mb();
        std::cout << "  C++ (" << (use_q8 ? "Q8_0" : "FP32") << ") Step " << step
                  << ": Loss = " << std::fixed << std::setprecision(6) << loss_val
                  << " | Time = " << std::setprecision(1) << ms << " ms"
                  << " | VmRSS = " << rss_mb << " MB | VmHWM = " << hwm_mb << " MB\n";
#if defined(__GLIBC__)
        malloc_trim(0);
#endif
    }

    std::ofstream out_cpp(data_dir + "/cpp_losses.json");
    out_cpp << "[\n";
    for (size_t i = 0; i < cpp_losses.size(); ++i) {
        out_cpp << "  " << std::setprecision(6) << cpp_losses[i] << (i + 1 < cpp_losses.size() ? ",\n" : "\n");
    }
    out_cpp << "]\n";

    return 0;
}
