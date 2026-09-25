// src/init.hpp
// Weight initialization utilities: PCG32 RNG + tensor initialization
// helpers (normal, zeros, eye, const) matching PyTorch defaults used by
// minagi/recur.py RecurCoder.__init__ and minagi/model.py.
#pragma once
#include "tensor.hpp"
#include <cstdint>
#include <string>

namespace minagi {

// Deterministic PRNG matching torch.manual_seed behavior closely enough for
// reproducible weight generation. Uses PCG32 (32-bit state, 64-bit sequence).
class PcgRng {
public:
    explicit PcgRng(uint64_t seed = 0);
    void seed(uint64_t s);
    uint32_t u32();              // [0, 2^32)
    float f32();                 // [0, 1) via 2^-24 scaling
    double f64();                // [0, 1)
private:
    uint64_t state_ = 0;
    uint64_t inc_ = 0;
    void advance();
};

// Box-Muller transform to Gaussian.
float normal_sample(PcgRng& rng, float mean=0.0f, float std=1.0f);

// Tensor initialization helpers (all return fresh FP32 tensors):
// normal(shape, mean, std)     — Gaussian fill (matches nn.init.normal_)
// zeros(shape)                 — zero fill (matches nn.init.zeros_)
// eye(rows, cols)              — identity (matches torch.eye)
// adapter(d)                   — [d, 2d] cat-mixer init [I|I] (matches
//                               RecurCoder adapter init in minagi/recur.py:
//                               h_new = h_prev + x at init). NOTE: init_eye(d,
//                               2d) is NOT this (it yields [I|0], which zeroes
//                               the x path and kills the whole forward).
// const_tensor(shape, val)     — constant fill (matches nn.init.constant_)
mt::Tensor init_normal(const mt::Shape& s, PcgRng& rng, float mean, float std);
mt::Tensor init_zeros(const mt::Shape& s);
mt::Tensor init_eye(int rows, int cols);
mt::Tensor init_adapter(int d_model);
mt::Tensor init_const(const mt::Shape& s, float val);

}  // namespace minagi
