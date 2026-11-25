#include "finetune_ops/core/tensor.h"
#include "finetune_ops/core/ops.h"
#include "finetune_ops/core/autograd_engine.h"
#include "finetune_ops/core/lm_loss.h"
#include "finetune_ops/core/memory_manager.h"
#include "finetune_ops/core/quant_io.h"
#include "finetune_ops/graph/auto_model.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <random>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <regex>
#include <cstring>
#include <unistd.h>
#if defined(__ANDROID__)
#include <malloc.h>
#define APP_TRIM_MEMORY() mallopt(M_PURGE, 0)
#elif defined(__GLIBC__)
#include <malloc.h>
#define APP_TRIM_MEMORY() malloc_trim(0)
#else
#define APP_TRIM_MEMORY() ((void)0)
#endif

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

double get_rss_mb() {
    std::ifstream statm("/proc/self/statm");
    if (!statm.is_open()) return 0.0;
    long total_pages = 0, resident_pages = 0;
    statm >> total_pages >> resident_pages;
    long page_size = sysconf(_SC_PAGESIZE);
    return (resident_pages * page_size) / (1024.0 * 1024.0);
}

double get_vmhwm_mb() {
    std::ifstream st("/proc/self/status");
    std::string line;
    while (std::getline(st, line)) {
        if (line.rfind("VmHWM:", 0) == 0) {
            std::istringstream iss(line.substr(6));
            double kb = 0.0;
            iss >> kb;
            return kb / 1024.0;
        }
    }
    return 0.0;
}

void print_rss_breakdown(const std::string& stage, QwenModel* qwen) {
    double rss = get_rss_mb();
    double hwm = get_vmhwm_mb();
    double pool_use_mb = MemoryManager::instance().get_memory_usage() / (1024.0 * 1024.0);
    double pool_peak_mb = MemoryManager::instance().get_peak_memory() / (1024.0 * 1024.0);

    double smaps_rss = 0.0, smaps_pss = 0.0, smaps_swap = 0.0;
    {
        std::ifstream sr("/proc/self/smaps_rollup");
        std::string line;
        while (std::getline(sr, line)) {
            auto parse_kb = [&](const char* key, double& out) {
                if (line.rfind(key, 0) == 0) {
                    std::istringstream iss(line.substr(std::strlen(key)));
                    double kb = 0.0;
                    iss >> kb;
                    out = kb / 1024.0;
                }
            };
            parse_kb("Rss:", smaps_rss);
            parse_kb("Pss:", smaps_pss);
            parse_kb("Swap:", smaps_swap);
        }
    }

    double q8_mb = 0.0;
    double fp32_base_mb = 0.0;
    if (qwen) {
        if (qwen->embedding_weight()) {
            fp32_base_mb += qwen->embedding_weight()->numel() * 4.0 / (1024.0 * 1024.0);
        }
        if (qwen->final_norm_weight()) {
            fp32_base_mb += qwen->final_norm_weight()->numel() * 4.0 / (1024.0 * 1024.0);
        }
        for (int li = 0; li < qwen->config().num_hidden_layers; ++li) {
            const auto& b = qwen->get_block(li);
            auto add_f32 = [&](const TensorPtr& t) { if (t) fp32_base_mb += t->numel() * 4.0 / (1024.0 * 1024.0); };
            add_f32(b.input_norm_weight); add_f32(b.post_norm_weight);
            add_f32(b.q_proj_weight); add_f32(b.q_proj_bias);
            add_f32(b.k_proj_weight); add_f32(b.k_proj_bias);
            add_f32(b.v_proj_weight); add_f32(b.v_proj_bias);
            add_f32(b.o_proj_weight);
            add_f32(b.gate_proj_weight); add_f32(b.up_proj_weight); add_f32(b.down_proj_weight);
            auto add_q8 = [&](const std::unique_ptr<LoRALinear>& lin) {
                if (lin && lin->is_quantized()) {
                    q8_mb += lin->quantized_weight().size() * sizeof(block_q8_0) / (1024.0 * 1024.0);
                }
            };
            add_q8(b.q_lin); add_q8(b.k_lin); add_q8(b.v_lin); add_q8(b.o_lin);
            add_q8(b.gate_lin); add_q8(b.up_lin); add_q8(b.down_lin);
        }
    }

    std::cout << "[RSS Breakdown | " << stage << "] "
              << std::fixed << std::setprecision(1)
              << "VmRSS=" << rss << " MB | VmHWM=" << hwm << " MB | "
              << "Swap=" << smaps_swap << " MB | "
              << "Q8_Weights=" << q8_mb << " MB | FP32_Model=" << fp32_base_mb << " MB | "
              << "MemPool_Use=" << pool_use_mb << " MB (Peak=" << pool_peak_mb << " MB)\n";
}

// Simple JSON parser for jsonl lines
struct SampleData {
    std::string image_id;
    std::vector<int32_t> input_ids;
    std::vector<int32_t> labels;
    int prompt_len = 0;
};

std::vector<SampleData> load_jsonl(const std::string& path, size_t max_samples = 0) {
    std::vector<SampleData> samples;
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Warning: Could not open " << path << std::endl;
        return samples;
    }
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        SampleData sd;
        // Parse image_id
        auto pos_id = line.find("\"image_id\": \"");
        if (pos_id != std::string::npos) {
            auto end_id = line.find("\"", pos_id + 13);
            sd.image_id = line.substr(pos_id + 13, end_id - (pos_id + 13));
        }
        // Parse input_ids
        auto pos_in = line.find("\"input_ids\": [");
        if (pos_in != std::string::npos) {
            auto end_in = line.find("]", pos_in + 14);
            std::stringstream ss(line.substr(pos_in + 14, end_in - (pos_in + 14)));
            int val;
            char comma;
            while (ss >> val) {
                sd.input_ids.push_back(val);
                ss >> comma;
            }
        }
        // Parse labels
        auto pos_lb = line.find("\"labels\": [");
        if (pos_lb != std::string::npos) {
            auto end_lb = line.find("]", pos_lb + 11);
            std::stringstream ss(line.substr(pos_lb + 11, end_lb - (pos_lb + 11)));
            int val;
            char comma;
            while (ss >> val) {
                sd.labels.push_back(val);
                ss >> comma;
            }
        }
        if (!sd.input_ids.empty()) {
            samples.push_back(sd);
            if (max_samples > 0 && samples.size() >= max_samples) break;
        }
    }
    return samples;
}

// Loads visual features from .npy (supports float16 or float32)
bool load_npy_features(const std::string& path, std::vector<float>& out_features, size_t& num_images, size_t& num_tokens, size_t& feat_dim) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return false;
    char magic[6];
    f.read(magic, 6);
    if (std::memcmp(magic, "\x93NUMPY", 6) != 0) return false;
    uint8_t major, minor;
    f.read(reinterpret_cast<char*>(&major), 1);
    f.read(reinterpret_cast<char*>(&minor), 1);
    uint16_t header_len;
    f.read(reinterpret_cast<char*>(&header_len), 2);
    std::string header(header_len, ' ');
    f.read(&header[0], header_len);

    bool is_fp16 = (header.find("f2") != std::string::npos || header.find("e") != std::string::npos);

    num_images = 50;
    num_tokens = 64;
    feat_dim = 768;

    size_t total_elements = num_images * num_tokens * feat_dim;
    out_features.resize(total_elements);

    if (is_fp16) {
        std::vector<uint16_t> fp16_data(total_elements);
        f.read(reinterpret_cast<char*>(fp16_data.data()), total_elements * sizeof(uint16_t));
        // Convert fp16 to fp32
        for (size_t i = 0; i < total_elements; ++i) {
            uint16_t h = fp16_data[i];
            uint32_t sign = (h & 0x8000) << 16;
            uint32_t exp = (h & 0x7c00) >> 10;
            uint32_t mant = (h & 0x03ff);
            uint32_t f32_val = 0;
            if (exp == 0) {
                f32_val = sign;
            } else if (exp == 0x1f) {
                f32_val = sign | 0x7f800000 | (mant << 13);
            } else {
                f32_val = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
            }
            float val;
            std::memcpy(&val, &f32_val, sizeof(float));
            out_features[i] = val;
        }
    } else {
        f.read(reinterpret_cast<char*>(out_features.data()), total_elements * sizeof(float));
    }
    return true;
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

// Multimodal Visual Projector: 768 -> 896 (Supports 2-Layer GELU MLP or Linear Projection)
class MultimodalProjector {
public:
    std::string arch_;
    int in_dim_;
    int out_dim_;
    int num_tokens_;
    TensorPtr scale_;

    // LayerNorm on input
    TensorPtr ln_w_;
    TensorPtr ln_b_;

    // Layer 1
    TensorPtr W1_;
    TensorPtr b1_;

    // Layer 2 (for 2-layer GELU MLP)
    TensorPtr W2_;
    TensorPtr b2_;

    // Positional embedding (capacity matched to soft prompt)
    TensorPtr pos_embed_;

    MultimodalProjector(int in_dim = 768, int out_dim = 896, int num_tokens = 64,
                        const std::string& arch = "mlp", unsigned seed = 42)
        : arch_(arch), in_dim_(in_dim), out_dim_(out_dim), num_tokens_(num_tokens) {

        std::mt19937 gen(seed);
        float target_norm = 0.4633f;
        float init_scale = target_norm / std::sqrt(static_cast<float>(out_dim_));
        scale_ = full({1}, init_scale, kFloat32, kCPU);
        scale_->set_requires_grad(true);

        // LayerNorm: init weights=1, bias=0
        ln_w_ = ones({in_dim}, kFloat32, kCPU);
        ln_b_ = zeros({in_dim}, kFloat32, kCPU);
        ln_w_->set_requires_grad(true);
        ln_b_->set_requires_grad(true);

        // Layer 1: [out_dim, in_dim]
        // Xavier uniform: [-limit, limit], limit = sqrt(6 / (fan_in + fan_out))
        float limit1 = std::sqrt(6.0f / (in_dim + out_dim));
        std::uniform_real_distribution<float> dist1(-limit1, limit1);
        W1_ = zeros({out_dim, in_dim}, kFloat32, kCPU);
        float* w1_ptr = static_cast<float*>(W1_->data_ptr());
        for (size_t i = 0; i < W1_->numel(); ++i) {
            w1_ptr[i] = dist1(gen);
        }
        b1_ = zeros({out_dim}, kFloat32, kCPU);
        W1_->set_requires_grad(true);
        b1_->set_requires_grad(true);

        if (arch_ == "mlp") {
            // Layer 2: [out_dim, out_dim]
            float limit2 = std::sqrt(6.0f / (out_dim + out_dim));
            std::uniform_real_distribution<float> dist2(-limit2, limit2);
            W2_ = zeros({out_dim, out_dim}, kFloat32, kCPU);
            float* w2_ptr = static_cast<float*>(W2_->data_ptr());
            for (size_t i = 0; i < W2_->numel(); ++i) {
                w2_ptr[i] = dist2(gen);
            }
            b2_ = zeros({out_dim}, kFloat32, kCPU);
            W2_->set_requires_grad(true);
            b2_->set_requires_grad(true);
        }

        // Positional Embedding: [1, num_tokens, out_dim]
        pos_embed_ = zeros({1, static_cast<int64_t>(num_tokens_), static_cast<int64_t>(out_dim_)}, kFloat32, kCPU);
        std::normal_distribution<float> pdist(0.0f, init_scale);
        float* pos_ptr = static_cast<float*>(pos_embed_->data_ptr());
        for (size_t i = 0; i < pos_embed_->numel(); ++i) {
            pos_ptr[i] = pdist(gen);
        }
        pos_embed_->set_requires_grad(true);
    }

    TensorPtr forward(const TensorPtr& visual_tokens) {
        // visual_tokens: [B, num_tokens, in_dim]
        // 1. LayerNorm
        auto ln_out = layer_norm(visual_tokens, ln_w_, ln_b_);

        // 2. Layer 1
        auto h1 = add(matmul_rhs_T(ln_out, W1_), b1_);

        TensorPtr out_tokens;
        if (arch_ == "mlp") {
            // 3. GELU
            auto act = gelu(h1);
            // 4. Layer 2
            auto h2 = add(matmul_rhs_T(act, W2_), b2_);
            // 5. Scaled to Qwen embedding norm
            out_tokens = mul(h2, scale_);
        } else {
            out_tokens = mul(h1, scale_);
        }

        // 6. Add Positional Embedding
        return add(out_tokens, pos_embed_);
    }

    std::vector<TensorPtr> parameters() {
        if (arch_ == "mlp") {
            return {ln_w_, ln_b_, W1_, b1_, W2_, b2_, scale_, pos_embed_};
        } else {
            return {ln_w_, ln_b_, W1_, b1_, scale_, pos_embed_};
        }
    }

    size_t num_parameters() const {
        size_t total = ln_w_->numel() + ln_b_->numel() + W1_->numel() + b1_->numel() + scale_->numel() + pos_embed_->numel();
        if (arch_ == "mlp") {
            total += W2_->numel() + b2_->numel();
        }
        return total;
    }

    void zero_grad() {
        for (auto& p : parameters()) {
            if (p) p->zero_grad();
        }
    }
};

// Freely learnable soft prompt baseline (no image input)
class LearnedPrefixModule {
public:
    TensorPtr prefix_;
    int num_tokens_;
    int dim_;

    LearnedPrefixModule(int num_tokens = 64, int dim = 896, unsigned seed = 42)
        : num_tokens_(num_tokens), dim_(dim) {
        float init_std = 0.4633f / std::sqrt(static_cast<float>(dim_));
        std::mt19937 gen(seed);
        std::normal_distribution<float> dist(0.0f, init_std);

        prefix_ = zeros({1, static_cast<int64_t>(num_tokens_), static_cast<int64_t>(dim_)}, kFloat32, kCPU);
        float* ptr = static_cast<float*>(prefix_->data_ptr());
        for (size_t i = 0; i < prefix_->numel(); ++i) {
            ptr[i] = dist(gen);
        }
        prefix_->set_requires_grad(true);
    }

    TensorPtr forward() {
        return prefix_;
    }

    std::vector<TensorPtr> parameters() {
        return {prefix_};
    }

    size_t num_parameters() const {
        return prefix_->numel();
    }

    void zero_grad() {
        if (prefix_) prefix_->zero_grad();
    }
};

// Adam optimizer for projector parameters
class SimpleAdam {
public:
    float lr_;
    float beta1_ = 0.9f;
    float beta2_ = 0.999f;
    float eps_ = 1e-8f;
    size_t step_ = 0;
    std::unordered_map<TensorPtr, std::pair<std::vector<float>, std::vector<float>>> state_;

    explicit SimpleAdam(float lr = 1e-3f) : lr_(lr) {}

    void step(const std::vector<TensorPtr>& params, float current_lr) {
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
                p_ptr[i] -= current_lr * m_hat / (std::sqrt(v_hat) + eps_);
            }
        }
    }
};

int main(int argc, char** argv) {
    std::string mode = "real"; // real, zeroed, shuffled, text_only, learned_prefix
    std::string arch = "mlp";  // mlp, linear
    std::string model_dir = "models/Qwen2.5-0.5B";
    std::string features_path = "data/vision_features/visual_features.npy";
    std::string train_path = "data/multimodal_splits/train.jsonl";
    std::string val_path = "data/multimodal_splits/val.jsonl";
    int num_steps = 20;
    int val_every = 5;
    int accum_steps = 1;
    float lr = 1e-3f;
    unsigned seed = 42;
    bool quantize_base = false;
    bool full_logits = false;
    std::string save_quantized_path;
    std::string load_quantized_path;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--mode" && i + 1 < argc) mode = argv[++i];
        else if (arg == "--arch" && i + 1 < argc) arch = argv[++i];
        else if (arg == "--steps" && i + 1 < argc) num_steps = std::stoi(argv[++i]);
        else if (arg == "--val_every" && i + 1 < argc) val_every = std::stoi(argv[++i]);
        else if (arg == "--accum_steps" && i + 1 < argc) accum_steps = std::stoi(argv[++i]);
        else if (arg == "--lr" && i + 1 < argc) lr = std::stof(argv[++i]);
        else if (arg == "--seed" && i + 1 < argc) seed = std::stoul(argv[++i]);
        else if (arg == "--quantize_base") quantize_base = true;
        else if (arg == "--full_logits") full_logits = true;
        else if (arg == "--save_quantized" && i + 1 < argc) save_quantized_path = argv[++i];
        else if (arg == "--load_quantized" && i + 1 < argc) load_quantized_path = argv[++i];
        else if (arg == "--model_dir" && i + 1 < argc) model_dir = argv[++i];
        else if (arg == "--train_path" && i + 1 < argc) train_path = argv[++i];
        else if (arg == "--val_path" && i + 1 < argc) val_path = argv[++i];
        else if (arg == "--features_path" && i + 1 < argc) features_path = argv[++i];
    }

    std::cout << "========================================================================\n";
    std::cout << "  MobileFineTuner: Multimodal Visual Projector Training Program        \n";
    std::cout << "  Architecture: " << (arch == "mlp" ? "2-Layer GELU MLP" : "Linear") 
              << " Projector (768 -> 896) + Frozen Qwen2.5-0.5B\n";
    std::cout << "  Execution Mode: [" << mode << "] | Steps: " << num_steps 
              << " | Accum: " << accum_steps << " | LR: " << lr << " | Seed: " << seed 
              << " | Quantize Base: " << (quantize_base ? "Q8_0" : "None (FP32)") << "\n";
    std::cout << "========================================================================\n";

    // 1. Load Visual Features
    std::vector<float> visual_features;
    size_t num_imgs = 0, num_tokens = 64, feat_dim = 768;
    bool has_features = load_npy_features(features_path, visual_features, num_imgs, num_tokens, feat_dim);
    if (!has_features && (mode == "real" || mode == "shuffled")) {
        std::cerr << "Visual features not found at " << features_path << ". Creating synthetic test features.\n";
        num_imgs = 50;
        visual_features.resize(num_imgs * num_tokens * feat_dim, 0.05f);
    }
    std::cout << "[" << get_wall_time_iso() << "] Loaded " << num_imgs << " image features (" 
              << num_tokens << " tokens x " << feat_dim << " dim).\n";

    // 2. Load Dataset Splits
    auto train_data = load_jsonl(train_path, 50);
    auto val_data = load_jsonl(val_path, 10);
    if (train_data.empty()) {
        std::cerr << "Error: Train dataset is empty at " << train_path << "\n";
        return 1;
    }
    std::cout << "[" << get_wall_time_iso() << "] Loaded " << train_data.size() << " train pairs, " 
              << val_data.size() << " validation pairs.\n";

    // 3. Load Qwen Model and Freeze Backbone
    std::shared_ptr<AutoModelForCausalLM> model;
    QwenModel* qwen = nullptr;

    if (!load_quantized_path.empty()) {
        std::cout << "[" << get_wall_time_iso() << "] Direct load from Q8_0 checkpoint: " 
                  << load_quantized_path << "...\n";
        print_rss_breakdown("Before Q8_0 Config Init", nullptr);
        AutoModelLoadOptions opts;
        opts.load_weights = false; // Never touch FP32 safetensors or pre-allocate FP32 base weights!
        model = AutoModelForCausalLM::from_pretrained(model_dir, opts);
        if (!model) {
            std::cerr << "Failed to initialize model config from " << model_dir << "\n";
            return 1;
        }
        qwen = model->qwen();
        qwen->init_linear_wrappers();
        print_rss_breakdown("After Empty Config Init", qwen);

        std::vector<ops::QuantTensor> tensors;
        std::string qerr;
        if (!ops::read_quant_model(load_quantized_path, tensors, &qerr)) {
            std::cerr << "Error reading quantized model " << load_quantized_path << ": " << qerr << "\n";
            return 1;
        }
        print_rss_breakdown("After Raw Q8 File Read", qwen);
        std::regex q8_name(R"(layers\.(\d+)\.(.+))");
        int adopted_q8 = 0, loaded_f32 = 0;
        for (auto& t : tensors) {
            if (t.is_q8) {
                std::smatch m;
                if (!std::regex_match(t.name, m, q8_name)) {
                    std::cerr << "Warning: unexpected q8 tensor name " << t.name << "\n";
                    continue;
                }
                int li = std::stoi(m[1].str());
                std::string sub = m[2].str();
                if (li < 0 || li >= qwen->config().num_hidden_layers) continue;
                auto& blk = qwen->get_block(li);
                LoRALinear* lin = nullptr;
                if (sub == "self_attn.q_proj.weight") lin = blk.q_lin.get();
                else if (sub == "self_attn.k_proj.weight") lin = blk.k_lin.get();
                else if (sub == "self_attn.v_proj.weight") lin = blk.v_lin.get();
                else if (sub == "self_attn.o_proj.weight") lin = blk.o_lin.get();
                else if (sub == "mlp.gate_proj.weight") lin = blk.gate_lin.get();
                else if (sub == "mlp.up_proj.weight") lin = blk.up_lin.get();
                else if (sub == "mlp.down_proj.weight") lin = blk.down_lin.get();
                if (lin) {
                    lin->adopt_quantized_weight(t.k, t.m, std::move(t.q8));
                    adopted_q8++;
                }
                t.q8.clear();
                t.q8.shrink_to_fit();
            } else {
                auto ten = std::make_shared<Tensor>(
                    t.shape, static_cast<const void*>(t.f32.data()), DType::kFloat32, kCPU);
                qwen->assign_weight(t.name, ten);
                t.f32.clear();
                t.f32.shrink_to_fit();
                loaded_f32++;
            }
        }
        tensors.clear();
        tensors.shrink_to_fit();
        for (int li = 0; li < qwen->config().num_hidden_layers; ++li) {
            auto& blk = qwen->get_block(li);
            if (blk.q_lin) blk.q_lin->set_base_weights(nullptr, blk.q_proj_bias);
            if (blk.k_lin) blk.k_lin->set_base_weights(nullptr, blk.k_proj_bias);
            if (blk.v_lin) blk.v_lin->set_base_weights(nullptr, blk.v_proj_bias);
            if (blk.o_lin) blk.o_lin->set_base_weights(nullptr, nullptr);
            if (blk.gate_lin) blk.gate_lin->set_base_weights(nullptr, nullptr);
            if (blk.up_lin) blk.up_lin->set_base_weights(nullptr, nullptr);
            if (blk.down_lin) blk.down_lin->set_base_weights(nullptr, nullptr);
            blk.q_proj_weight.reset();
            blk.k_proj_weight.reset();
            blk.v_proj_weight.reset();
            blk.o_proj_weight.reset();
            blk.gate_proj_weight.reset();
            blk.up_proj_weight.reset();
            blk.down_proj_weight.reset();
        }
        MemoryManager::instance().clear_unused_memory();
        APP_TRIM_MEMORY();
        qwen->freeze_base();
        std::cout << "[" << get_wall_time_iso() << "] Successfully loaded: " << adopted_q8 
                  << " Q8_0 weights, " << loaded_f32 << " FP32 tensors.\n";
        std::cout << "[" << get_wall_time_iso() << "] Base Model RAM upon Q8_0 load: " 
                  << std::fixed << std::setprecision(2) << get_rss_mb() << " MB\n";
        print_rss_breakdown("After Q8_0 Load & Trim", qwen);
    } else {
        std::cout << "[" << get_wall_time_iso() << "] Loading and freezing Qwen2.5-0.5B backbone...\n";
        AutoModelLoadOptions opts;
        opts.load_weights = true;
        model = AutoModelForCausalLM::from_pretrained(model_dir, opts);
        if (!model) {
            std::cerr << "Failed to load model\n";
            return 1;
        }
        qwen = model->qwen();
        qwen->freeze_base(); // CRITICAL: Freezes all 290 backbone parameters

        if (quantize_base || !save_quantized_path.empty()) {
            std::cout << "[" << get_wall_time_iso() << "] RSS before base quantization: " 
                      << std::fixed << std::setprecision(2) << get_rss_mb() << " MB\n";
            qwen->quantize_base_weights_q8_0();
            std::cout << "[" << get_wall_time_iso() << "] RSS after base quantization: " 
                      << std::fixed << std::setprecision(2) << get_rss_mb() << " MB\n";
        }

        if (!save_quantized_path.empty()) {
            std::cout << "[" << get_wall_time_iso() << "] Exporting self-contained Q8_0 checkpoint to " 
                      << save_quantized_path << "...\n";
            std::vector<ops::QuantTensor> out;
            auto push_f32 = [&](const std::string& name, const TensorPtr& tt) {
                if (!tt) return;
                ops::QuantTensor e;
                e.name = name;
                e.shape = tt->shape();
                const float* p = tt->data<float>();
                e.f32.assign(p, p + tt->numel());
                out.push_back(std::move(e));
            };
            auto push_q8 = [&](const std::string& name, const LoRALinear* lin) {
                if (!lin || !lin->is_quantized()) return;
                ops::QuantTensor e;
                e.name = name;
                e.is_q8 = true;
                e.k = lin->quantized_k();
                e.m = lin->quantized_m();
                e.q8 = lin->quantized_weight();
                out.push_back(std::move(e));
            };
            for (int li = 0; li < qwen->config().num_hidden_layers; ++li) {
                auto& b = qwen->get_block(li);
                const std::string p = "layers." + std::to_string(li) + ".";
                push_q8(p + "self_attn.q_proj.weight", b.q_lin.get());
                push_q8(p + "self_attn.k_proj.weight", b.k_lin.get());
                push_q8(p + "self_attn.v_proj.weight", b.v_lin.get());
                push_q8(p + "self_attn.o_proj.weight", b.o_lin.get());
                push_q8(p + "mlp.gate_proj.weight", b.gate_lin.get());
                push_q8(p + "mlp.up_proj.weight", b.up_lin.get());
                push_q8(p + "mlp.down_proj.weight", b.down_lin.get());

                push_f32(p + "self_attn.q_proj.bias", b.q_proj_bias);
                push_f32(p + "self_attn.k_proj.bias", b.k_proj_bias);
                push_f32(p + "self_attn.v_proj.bias", b.v_proj_bias);
                push_f32(p + "input_norm.weight", b.input_norm_weight);
                push_f32(p + "post_norm.weight", b.post_norm_weight);
            }
            push_f32("embed_tokens.weight", qwen->embedding_weight());
            push_f32("final_norm.weight", qwen->final_norm_weight());

            std::string werr;
            if (!ops::write_quant_model(save_quantized_path, out, &werr)) {
                std::cerr << "Error writing quantized model: " << werr << "\n";
                return 1;
            }
            std::cout << "[" << get_wall_time_iso() << "] Successfully exported Q8_0 checkpoint: " 
                      << save_quantized_path << " (" << out.size() << " tensors)\n";
            return 0; // Export complete
        }
    }

    // 4. Instantiate Trainable Projector / Module and Optimizer
    std::unique_ptr<MultimodalProjector> projector;
    std::unique_ptr<LearnedPrefixModule> learned_prefix;
    std::vector<TensorPtr> trainable_params;

    if (mode == "learned_prefix") {
        learned_prefix = std::make_unique<LearnedPrefixModule>(num_tokens, 896, seed);
        trainable_params = learned_prefix->parameters();
        std::cout << "[" << get_wall_time_iso() << "] Initialized Learned Prefix Soft-Prompt (" 
                  << learned_prefix->num_parameters() << " parameters).\n";
    } else {
        projector = std::make_unique<MultimodalProjector>(768, 896, num_tokens, arch, seed);
        trainable_params = projector->parameters();
        std::cout << "[" << get_wall_time_iso() << "] Initialized " 
                  << (arch == "mlp" ? "2-Layer GELU MLP" : "Linear") << " Projector (" 
                  << projector->num_parameters() << " parameters).\n";
    }

    SimpleAdam optimizer(lr);

    auto zero_all_grad = [&]() {
        if (projector) projector->zero_grad();
        if (learned_prefix) learned_prefix->zero_grad();
    };

    // Helper to evaluate validation loss
    auto evaluate_val = [&](int step) {
        if (val_data.empty()) return;
        ops::autograd::Engine::instance().set_enabled(false);
        float total_val_loss = 0.0f;
        int val_count = 0;
        for (size_t vi = 0; vi < std::min<size_t>(5, val_data.size()); ++vi) {
            const auto& sample = val_data[vi];
            int64_t T = sample.input_ids.size();
            auto text_id_tensor = zeros({1, T}, DType::kInt32, kCPU);
            std::memcpy(text_id_tensor->data_ptr(), sample.input_ids.data(), T * sizeof(int32_t));
            auto text_embeds = lookup_embeds(qwen->embedding_weight(), text_id_tensor);

            TensorPtr spliced;
            if (mode == "text_only") {
                spliced = text_embeds;
            } else if (mode == "learned_prefix") {
                auto p_tokens = learned_prefix->forward();
                spliced = concat(std::vector<TensorPtr>{p_tokens, text_embeds}, 1);
            } else {
                auto v_tensor = zeros({1, static_cast<int64_t>(num_tokens), static_cast<int64_t>(feat_dim)}, kFloat32, kCPU);
                if (mode == "real" || mode == "shuffled") {
                    size_t img_idx = (mode == "shuffled") ? ((vi + 17) % num_imgs) : (vi % num_imgs);
                    std::memcpy(v_tensor->data_ptr(), visual_features.data() + img_idx * num_tokens * feat_dim, num_tokens * feat_dim * sizeof(float));
                }
                auto p_tokens = projector->forward(v_tensor);
                spliced = concat(std::vector<TensorPtr>{p_tokens, text_embeds}, 1);
            }

            auto h = qwen->forward_hidden_from_embeds(spliced);
            int64_t total_seq = spliced->shape()[1];
            int64_t prefix_len = (mode == "text_only") ? 0 : num_tokens;
            TensorPtr loss;
            if (full_logits) {
                auto logits = qwen->lm_head(h);
                int64_t pred_seq = total_seq - 1;
                auto flat_l = reshape(logits->slice(1, 0, pred_seq, 1), {pred_seq, 151936});
                auto targets = zeros({pred_seq}, DType::kInt32, kCPU);
                int32_t* tgt_ptr = static_cast<int32_t*>(targets->data_ptr());
                for (int64_t i = 0; i < pred_seq; ++i) {
                    if (i < prefix_len - 1) tgt_ptr[i] = -100;
                    else {
                        int64_t tidx = i - prefix_len + 1;
                        tgt_ptr[i] = (tidx >= 0 && tidx < static_cast<int64_t>(sample.labels.size())) ? sample.labels[tidx] : -100;
                    }
                }
                loss = cross_entropy_loss(flat_l, targets, "mean");
            } else {
                auto full_labels = zeros({1, total_seq}, DType::kInt32, kCPU);
                int32_t* lbl_ptr = static_cast<int32_t*>(full_labels->data_ptr());
                std::fill_n(lbl_ptr, total_seq, -100);
                for (size_t tidx = 0; tidx < sample.labels.size(); ++tidx) {
                    int64_t pos = prefix_len + static_cast<int64_t>(tidx);
                    if (pos < total_seq) lbl_ptr[pos] = sample.labels[tidx];
                }
                loss = selected_token_lm_cross_entropy(h, qwen->embedding_weight(), full_labels, -100, "mean");
            }
            total_val_loss += static_cast<const float*>(loss->data_ptr())[0];
            val_count++;
        }
        ops::autograd::Engine::instance().set_enabled(true);
        ops::autograd::Engine::instance().clear_graph();
        MemoryManager::instance().clear_unused_memory();
        APP_TRIM_MEMORY();
        float avg_val_loss = total_val_loss / (val_count > 0 ? val_count : 1);
        std::cout << "  >>> [Step " << step << " Validation] Mean Loss = " << std::fixed << std::setprecision(4) << avg_val_loss << std::endl;
    };

    // 5. Training Loop
    std::cout << "\n------------------------------------------------------------------------\n";
    std::cout << "Starting Training Steps (" << arch << " Projector, Mode=" << mode << ")..." << std::endl;
    std::cout << "------------------------------------------------------------------------\n";

    if (val_every > 0) {
        evaluate_val(0); // Initial zero-shot validation loss
    }

    std::mt19937 rng(seed);
    zero_all_grad();

    for (int step = 1; step <= num_steps; ++step) {
        auto t_step0 = std::chrono::high_resolution_clock::now();
        const auto& sample = train_data[(step - 1) % train_data.size()];
        int64_t T = sample.input_ids.size();

        // Prepare text embeddings
        auto text_id_tensor = zeros({1, T}, DType::kInt32, kCPU);
        std::memcpy(text_id_tensor->data_ptr(), sample.input_ids.data(), T * sizeof(int32_t));
        auto text_embeds = lookup_embeds(qwen->embedding_weight(), text_id_tensor);

        // Prepare visual tokens based on mode
        TensorPtr spliced;
        int64_t prefix_len = 0;

        if (mode == "text_only") {
            spliced = text_embeds;
            prefix_len = 0;
        } else if (mode == "learned_prefix") {
            prefix_len = num_tokens;
            auto proj_tokens = learned_prefix->forward();
            spliced = concat(std::vector<TensorPtr>{proj_tokens, text_embeds}, 1);
        } else {
            prefix_len = num_tokens;
            auto v_tensor = zeros({1, static_cast<int64_t>(num_tokens), static_cast<int64_t>(feat_dim)}, kFloat32, kCPU);
            if (mode == "real") {
                size_t img_idx = (step - 1) % num_imgs;
                std::memcpy(v_tensor->data_ptr(), visual_features.data() + img_idx * num_tokens * feat_dim, num_tokens * feat_dim * sizeof(float));
            } else if (mode == "shuffled") {
                size_t img_idx = rng() % num_imgs;
                std::memcpy(v_tensor->data_ptr(), visual_features.data() + img_idx * num_tokens * feat_dim, num_tokens * feat_dim * sizeof(float));
            } // if "zeroed", v_tensor remains zeros

            auto proj_tokens = projector->forward(v_tensor);
            spliced = concat(std::vector<TensorPtr>{proj_tokens, text_embeds}, 1);
        }

        // Forward pass through frozen Qwen + selected-token or full LM loss
        auto t_fwd0 = std::chrono::high_resolution_clock::now();
        auto h = qwen->forward_hidden_from_embeds(spliced);
        int64_t total_seq = spliced->shape()[1];
        TensorPtr logits, flat_l, targets, full_labels, loss;
        if (full_logits) {
            logits = qwen->lm_head(h);
            int64_t pred_seq = total_seq - 1;
            flat_l = reshape(logits->slice(1, 0, pred_seq, 1), {pred_seq, 151936});
            targets = zeros({pred_seq}, DType::kInt32, kCPU);
            int32_t* tgt_ptr = static_cast<int32_t*>(targets->data_ptr());
            for (int64_t i = 0; i < pred_seq; ++i) {
                if (i < prefix_len - 1) {
                    tgt_ptr[i] = -100;
                } else {
                    int64_t tidx = i - prefix_len + 1;
                    tgt_ptr[i] = (tidx >= 0 && tidx < static_cast<int64_t>(sample.labels.size())) ? sample.labels[tidx] : -100;
                }
            }
            loss = cross_entropy_loss(flat_l, targets, "mean");
        } else {
            full_labels = zeros({1, total_seq}, DType::kInt32, kCPU);
            int32_t* lbl_ptr = static_cast<int32_t*>(full_labels->data_ptr());
            std::fill_n(lbl_ptr, total_seq, -100);
            for (size_t tidx = 0; tidx < sample.labels.size(); ++tidx) {
                int64_t pos = prefix_len + static_cast<int64_t>(tidx);
                if (pos < total_seq) lbl_ptr[pos] = sample.labels[tidx];
            }
            loss = selected_token_lm_cross_entropy(h, qwen->embedding_weight(), full_labels, -100, "mean");
        }
        auto t_fwd1 = std::chrono::high_resolution_clock::now();
        double fwd_ms = std::chrono::duration<double, std::milli>(t_fwd1 - t_fwd0).count();

        float loss_val = static_cast<const float*>(loss->data_ptr())[0];
        if (step == 1) print_rss_breakdown("Step 1 After Fwd+LMHead+Loss", qwen);

        // Backward pass (with gradient accumulation scaling)
        auto t_bwd0 = std::chrono::high_resolution_clock::now();
        if (accum_steps > 1) {
            auto grad_scale = full(loss->shape(), 1.0f / accum_steps, kFloat32, kCPU);
            loss->backward(grad_scale);
        } else {
            loss->backward();
        }
        auto t_bwd1 = std::chrono::high_resolution_clock::now();
        double bwd_ms = std::chrono::duration<double, std::milli>(t_bwd1 - t_bwd0).count();
        if (step == 1) print_rss_breakdown("Step 1 After Backward", qwen);

        // Optimizer step every accum_steps or last step
        if (step % accum_steps == 0 || step == num_steps) {
            optimizer.step(trainable_params, lr);
            zero_all_grad();
        }
        ops::autograd::Engine::instance().clear_graph();
        loss.reset(); flat_l.reset(); logits.reset(); h.reset(); spliced.reset(); targets.reset(); full_labels.reset();
        MemoryManager::instance().clear_unused_memory();
        APP_TRIM_MEMORY();
        if (step == 1) print_rss_breakdown("Step 1 After Cleanup & Pool Trim", qwen);

        auto t_step1 = std::chrono::high_resolution_clock::now();
        double total_ms = std::chrono::duration<double, std::milli>(t_step1 - t_step0).count();
        double rss_mb = get_rss_mb();

        std::cout << "[" << get_wall_time_iso() << "] Step " << std::setw(2) << step << "/" << num_steps 
                  << " | Loss = " << std::fixed << std::setprecision(4) << loss_val 
                  << " | RSS = " << std::fixed << std::setprecision(1) << rss_mb << " MB"
                  << " | Fwd = " << std::setw(6) << std::setprecision(1) << fwd_ms << " ms"
                  << " | Bwd = " << std::setw(6) << bwd_ms << " ms"
                  << " | Total = " << std::setw(6) << total_ms << " ms" << std::endl;

        if (val_every > 0 && step % val_every == 0) {
            evaluate_val(step);
        }
    }

    std::cout << "\n========================================================================\n";
    std::cout << "SUCCESS: Multimodal Projector Training Completed Successfully.\n";
    std::cout << "========================================================================\n";
    return 0;
}
