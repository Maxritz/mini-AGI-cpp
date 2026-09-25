// src/backward.hpp
// CPU-side backward pass for mini-AGI training.
//
// Computes gradients through the forward ops in recur.cpp (cross-entropy,
// matmul, rmsnorm, rope, attention, silu, residual) given the StepOut + caches
// and the target tokens. The forward path in recur.cpp is inference-only (no
// autograd graph), so BackwardCache stores intermediate activations needed for
// gradient computation.
//
// The DX12 backward shaders (src/shaders/backward.hlsl + dx12_dense.cpp)
// mirror this CPU path for GPU training; both are validated and gap #10/#11
// are closed. This CPU path matches the forward in recur.cpp.
#pragma once
#include "tensor.hpp"
#include "recur.hpp"
#include "pool.hpp"
#include "paged.hpp"
#include "model.hpp"
#include <cstdint>
#include <vector>
#include <string>
#include <utility>

namespace minagi::backward {

// Intermediate activations stored during the forward pass for backward use.
struct BackwardCache {
    int T = 0;
    int d = 0;
    int V = 0;
    int M = 0;
    int n_steps = 0;
    int n_recur = 0;
    int H = 0;
    int hd = 0;
    int half = 0;

    // Tok embedding lookup
    std::vector<int32_t> idx_flat;     // [M] token ids

    // Prelude block intermediates
    mt::Tensor pre_x_after_attn;       // [M,d]  input+x_after_attn
    mt::Tensor pre_xn2;                // [M,d]  rms_norm(x_after_attn, ln2)
    mt::Tensor pre_h1;                 // [M,dff]  xn2 @ w1^T
    mt::Tensor pre_h3;                 // [M,dff]  xn2 @ w3^T
    mt::Tensor pre_g;                  // [M,dff]  silu(h1)*h3
    mt::Tensor pre_out;                // [M,d]  prelude block output (residual)

    // Recurrence intermediates (per step)
    std::vector<mt::Tensor> h_before_adapter;  // [N, M, d]  h at start of step n
    std::vector<mt::Tensor> h_after_adapter;   // [N, M, d]  h after adapter
    std::vector<mt::Tensor> yf_per_step;       // [N, M, d]  ln_f(h) per step
    std::vector<mt::Tensor> logits_per_step;   // [N, M, V]  head(yf) per step
    std::vector<mt::Tensor> lam_per_step;      // [N, M]     halting prob per step

    // Recurent block intermediates (per step, per block)
    std::vector<std::vector<mt::Tensor>> rb_xn1;     // [N][n_recur] [M,d]  rms_norm(h, ln1)
    std::vector<std::vector<mt::Tensor>> rb_qkv;    // [N][n_recur] [M,3d]
    std::vector<std::vector<mt::Tensor>> rb_qh;      // [N][n_recur] [H*T, hd]
    std::vector<std::vector<mt::Tensor>> rb_kh;      // [N][n_recur] [H*T, hd]
    std::vector<std::vector<mt::Tensor>> rb_vh;      // [N][n_recur] [H*T, hd]
    std::vector<std::vector<mt::Tensor>> rb_sm;     // [N][n_recur] [H*T, kv_len]
    std::vector<std::vector<mt::Tensor>> rb_out_flat; // [N][n_recur] [M, d]  attention output
    std::vector<std::vector<mt::Tensor>> rb_in;     // [N][n_recur] [M,d]  input to block

    // Pool MLP intermediates
    std::vector<mt::Tensor> rb_xn2;                     // [N] [M,d]  rms_norm(h, ln2)
    std::vector<minagi::PoolBackwardCache> rb_pool_cache; // [N] routing info for backward

    std::vector<float> cum_init;                     // [M] initial cum
    std::vector<int> steps_used;                    // [M]

    // Target tokens for loss computation
    std::vector<int32_t> target_flat;               // [M] next-token targets
};

// Gradient container: param_name -> gradient tensor (same shape as param)
struct Gradients {
    std::vector<std::pair<std::string, mt::Tensor>> grads;
    void add(const std::string& name, const mt::Tensor& grad) {
        for (auto& g : grads) {
            if (g.first == name) {
                // Accumulate
                const float* gp = grad.ptr<float>();
                float* tp = g.second.ptr<float>();
                for (int64_t i = 0; i < g.second.shape.numel(); ++i) {
                    tp[i] += gp[i];
                }
                return;
            }
        }
        grads.emplace_back(name, grad);
    }
    mt::Tensor* get(const std::string& name) {
        for (auto& g : grads) if (g.first == name) return &g.second;
        return nullptr;
    }
};

// Forward with cache: runs the same forward as Coder::forward but stores
// intermediates + targets into BackwardCache for the backward pass.
BackwardCache forward_with_cache(const Config& cfg, Coder& coder,
                                 const mt::Tensor& idx,
                                 const mt::Tensor& targets,
                                 std::vector<model::KVCache>& caches,
                                 int64_t pos_offset);

// Cross-entropy loss: halt-weighted mixture of per-step CE.
// Reads targets from cache.target_flat.
// Returns total loss and fills d_logits [N*M, V] (gradient w.r.t. all-step logits).
float cross_entropy_loss(const BackwardCache& cache,
                         mt::Tensor& d_logits_out);

// Backward pass: given cache + d_logits, compute gradients for all parameters.
Gradients backward(const Config& cfg, Coder& coder,
                   const BackwardCache& cache,
                   const mt::Tensor& d_logits,
                   std::vector<model::KVCache>& caches);

// Full forward + loss + backward in one call. Returns loss.
float compute_loss_and_grads(const Config& cfg, Coder& coder,
                             const mt::Tensor& idx,
                             const mt::Tensor& targets,
                             int64_t pos_offset,
                             std::vector<model::KVCache>& caches,
                             Gradients& grads_out);

}  // namespace minagi::backward
