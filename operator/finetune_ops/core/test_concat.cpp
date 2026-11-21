#include "ops.h"
#include "autograd_engine.h"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

using namespace ops;

namespace {

void require(bool ok, const char* message) {
    if (!ok) {
        std::cerr << "FAIL: " << message << std::endl;
        std::exit(1);
    }
}

void require_close(float got, float expected, const char* name) {
    if (std::fabs(got - expected) > 1e-5f) {
        std::cerr << "FAIL: " << name << " got=" << got << " expected=" << expected << std::endl;
        std::exit(1);
    }
}

void test_concat_2d_forward_and_backward() {
    // A: [2, 3]
    auto a = zeros({2, 3}, kFloat32, kCPU);
    float* a_data = a->data<float>();
    for (int i = 0; i < 6; ++i) a_data[i] = static_cast<float>(i + 1); // 1, 2, 3, 4, 5, 6
    a->set_requires_grad(true);

    // B: [2, 2]
    auto b = zeros({2, 2}, kFloat32, kCPU);
    float* b_data = b->data<float>();
    for (int i = 0; i < 4; ++i) b_data[i] = static_cast<float>(10 + i); // 10, 11, 12, 13
    b->set_requires_grad(true);

    // Concat along dim=1 -> [2, 5]
    auto c = concat({a, b}, 1);
    require(c->shape() == std::vector<int64_t>({2, 5}), "concat shape [2,5]");

    const float* c_data = c->data<float>();
    // Row 0: 1, 2, 3, 10, 11
    require_close(c_data[0], 1.0f, "c[0,0]");
    require_close(c_data[1], 2.0f, "c[0,1]");
    require_close(c_data[2], 3.0f, "c[0,2]");
    require_close(c_data[3], 10.0f, "c[0,3]");
    require_close(c_data[4], 11.0f, "c[0,4]");
    // Row 1: 4, 5, 6, 12, 13
    require_close(c_data[5], 4.0f, "c[1,0]");
    require_close(c_data[6], 5.0f, "c[1,1]");
    require_close(c_data[7], 6.0f, "c[1,2]");
    require_close(c_data[8], 12.0f, "c[1,3]");
    require_close(c_data[9], 13.0f, "c[1,4]");

    // Backward pass: loss = sum(c * weight_vector)
    // Create weights for c: [2, 5]
    auto w = zeros({2, 5}, kFloat32, kCPU);
    float* w_data = w->data<float>();
    for (int i = 0; i < 10; ++i) w_data[i] = 2.0f; // gradient is 2.0 everywhere

    auto loss = sum(mul(c, w));
    loss->backward();

    require(a->grad() != nullptr, "a->grad not null");
    require(b->grad() != nullptr, "b->grad not null");

    const float* ga = a->grad()->data<float>();
    for (int i = 0; i < 6; ++i) {
        require_close(ga[i], 2.0f, "ga[i]");
    }

    const float* gb = b->grad()->data<float>();
    for (int i = 0; i < 4; ++i) {
        require_close(gb[i], 2.0f, "gb[i]");
    }

    std::cout << "[PASS] test_concat_2d_forward_and_backward" << std::endl;
}

void test_concat_3d_sequence_splice() {
    // Simulated vision tokens: [B=1, S_vision=4, H=8]
    auto vision = zeros({1, 4, 8}, kFloat32, kCPU);
    vision->set_requires_grad(true);

    // Simulated text tokens: [B=1, S_text=6, H=8]
    auto text = zeros({1, 6, 8}, kFloat32, kCPU);
    text->set_requires_grad(true);

    // Spliced sequence: [1, 10, 8]
    auto spliced = concat({vision, text}, 1);
    require(spliced->shape() == std::vector<int64_t>({1, 10, 8}), "spliced shape [1,10,8]");

    auto loss = sum(spliced);
    loss->backward();

    require(vision->grad() != nullptr, "vision->grad not null");
    require(text->grad() != nullptr, "text->grad not null");
    require(vision->grad()->shape() == std::vector<int64_t>({1, 4, 8}), "vision grad shape");
    require(text->grad()->shape() == std::vector<int64_t>({1, 6, 8}), "text grad shape");

    const float* gv = vision->grad()->data<float>();
    for (int i = 0; i < 32; ++i) {
        require_close(gv[i], 1.0f, "gv[i]");
    }

    std::cout << "[PASS] test_concat_3d_sequence_splice" << std::endl;
}

} // namespace

int main() {
    test_concat_2d_forward_and_backward();
    test_concat_3d_sequence_splice();
    std::cout << "[ALL PASS] Concat operator and autograd tests succeeded." << std::endl;
    return 0;
}
