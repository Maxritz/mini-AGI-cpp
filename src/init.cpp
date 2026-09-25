// src/init.cpp
#include "init.hpp"
#include <cmath>

namespace minagi {

PcgRng::PcgRng(uint64_t seed) {
    state_ = (seed + 0x9E3779B97F35262FULL) * 0xCA52F231E11A209FULL;
    inc_ = 1ULL;
    advance();
    state_ = state_ * 6364136223846793005ULL + inc_;
    advance();
}

void PcgRng::seed(uint64_t s) {
    state_ = (s + 0x9E3779B97F35262FULL) * 0xCA52F231E11A209FULL;
    inc_ = 1ULL;
    advance();
    state_ = state_ * 6364136223846793005ULL + inc_;
    advance();
}

void PcgRng::advance() {
    state_ = state_ * 6364136223846793005ULL + inc_;
    uint32_t xorshift = static_cast<uint32_t>((state_ >> 18u) ^ state_);
    uint32_t rot = static_cast<uint32_t>(state_ >> 59u);
    if (rot == 0) {
        state_ ^= (xorshift >> 1) | (xorshift << 31);
    } else {
        state_ ^= (xorshift >> rot) | (xorshift << (32 - rot));
    }
}

uint32_t PcgRng::u32() {
    advance();
    return static_cast<uint32_t>(state_ >> 33u);
}

float PcgRng::f32() {
    return static_cast<float>(u32() >> 8) * (1.0f / 16777216.0f);
}

double PcgRng::f64() {
    uint64_t a = u32();
    uint64_t b = u32();
    uint64_t combined = (a << 32) | b;
    return static_cast<double>(combined >> 11) * (1.0 / 9007199254740992.0);
}

float normal_sample(PcgRng& rng, float mean, float std) {
    // Box-Muller transform
    float u1 = std::max(rng.f32(), 1e-37f);
    float u2 = rng.f32();
    float z = std::sqrt(-2.0f * std::log(u1)) * std::cos(2.0f * 3.14159265f * u2);
    return mean + std * z;
}

mt::Tensor init_normal(const mt::Shape& s, PcgRng& rng, float mean, float std) {
    mt::Tensor t = mt::make(s, mt::DType::FP32, 0.0f);
    float* p = t.ptr<float>();
    int64_t n = s.numel();
    for (int64_t i = 0; i < n; ++i) {
        p[i] = normal_sample(rng, mean, std);
    }
    return t;
}

mt::Tensor init_zeros(const mt::Shape& s) {
    return mt::make(s, mt::DType::FP32, 0.0f);
}

mt::Tensor init_eye(int rows, int cols) {
    mt::Shape s;
    s.rank = 2;
    s.d[0] = rows;
    s.d[1] = cols;
    mt::Tensor t = mt::make(s, mt::DType::FP32, 0.0f);
    float* p = t.ptr<float>();
    int mn = std::min(rows, cols);
    for (int i = 0; i < mn; ++i) {
        p[i * cols + i] = 1.0f;
    }
    return t;
}

mt::Tensor init_adapter(int d_model) {
    mt::Shape s;
    s.rank = 2;
    s.d[0] = d_model;
    s.d[1] = 2 * d_model;
    mt::Tensor t = mt::make(s, mt::DType::FP32, 0.0f);
    float* p = t.ptr<float>();
    for (int i = 0; i < d_model; ++i) {
        p[i * 2 * d_model + i] = 1.0f;              // h_prev path
        p[i * 2 * d_model + d_model + i] = 1.0f;    // x path
    }
    return t;
}

mt::Tensor init_const(const mt::Shape& s, float val) {
    return mt::make(s, mt::DType::FP32, val);
}

}  // namespace minagi
