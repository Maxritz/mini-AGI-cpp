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

    // RoPE tables sliced to the window actually read, [T, half]
    mt::Tensor cos;
    mt::Tensor sin;

    // Prelude block intermediates
    mt::Tensor pre_x;                  // [M,d]  tok_emb(idx)
    mt::Tensor pre_x_after_attn;       // [M,d]  x + proj_out
    mt::Tensor pre_xn1;                // [M,d]  rms_norm(x, ln1)
    mt::Tensor pre_qkv;                // [M,3d] xn1 @ qkv^T
    mt::Tensor pre_qh;                 // [H*T,hd] post-RoPE
    mt::Tensor pre_kc;                 // [H*kv_len,hd] keys the query actually attended
    mt::Tensor pre_vc;                 // [H*kv_len,hd] values likewise
    mt::Tensor pre_sm;                 // [H*T,kv_len] attention weights
    mt::Tensor pre_out_flat;           // [M,d]  attention output before proj
    mt::Tensor pre_xn2;                // [M,d]  rms_norm(x_after_attn, ln2)
    mt::Tensor pre_h1;                 // [M,dff]  xn2 @ w1^T
    mt::Tensor pre_h3;                 // [M,dff]  xn2 @ w3^T
    mt::Tensor pre_g;                  // [M,dff]  silu(h1)*h3
    mt::Tensor pre_out;                // [M,d]  prelude block output (residual)

    // Recurrence intermediates (per step)
    std::vector<mt::Tensor> h_before_adapter;  // [N, M, d]  h at start of step n
    std::vector<mt::Tensor> h_after_adapter;   // [N, M, d]  h after adapter
    std::vector<mt::Tensor> h_end_per_step;    // [N, M, d]  h after the last recur block (ln_f input)
    std::vector<mt::Tensor> yf_per_step;       // [N, M, d]  ln_f(h) per step
    std::vector<mt::Tensor> logits_per_step;   // [N, M, V]  head(yf) per step
    std::vector<mt::Tensor> lam_per_step;      // [N, M]     halting prob per step

    // dL/d(lam_n) per step, [N, M], filled by the halting section of
    // compute_loss_and_grads. Exposed so tests can differentiate the loss with
    // respect to the CACHED lam instead of perturbing a parameter: a parameter
    // perturbation forces a float32 forward, and the halting signal is small
    // enough that the numeric side is then pure noise. A test that re-derives
    // the recursion itself cannot catch a bug in it - that mistake shipped a
    // missing batch mean through a green suite. This field is the library's own
    // value, so a test comparing against it is testing the shipped code.
    // Only populated after compute_loss_and_grads (not by forward_with_cache).
    std::vector<std::vector<float>> d_lam_per_step;  // [N][M]

    // Recurent block intermediates (per step, per block)
    std::vector<std::vector<mt::Tensor>> rb_xn1;     // [N][n_recur] [M,d]  rms_norm(h, ln1)
    std::vector<std::vector<mt::Tensor>> rb_qkv;    // [N][n_recur] [M,3d]
    std::vector<std::vector<mt::Tensor>> rb_qh;      // [N][n_recur] [H*T, hd]
    std::vector<std::vector<mt::Tensor>> rb_kc;      // [N][n_recur] [H*kv_len,hd] attended keys
    std::vector<std::vector<mt::Tensor>> rb_vc;      // [N][n_recur] [H*kv_len,hd] attended values
    std::vector<std::vector<mt::Tensor>> rb_sm;     // [N][n_recur] [H*T, kv_len]
    std::vector<std::vector<mt::Tensor>> rb_out_flat; // [N][n_recur] [M, d]  attention output
    std::vector<std::vector<mt::Tensor>> rb_in;     // [N][n_recur] [M,d]  input to block
    std::vector<std::vector<mt::Tensor>> rb_after_attn;  // [N][n_recur] [M,d] h after attn residual (ln2 input)
    std::vector<std::vector<mt::Tensor>> rb_h1;     // [N][n_recur] [M,dff] dense MLP only
    std::vector<std::vector<mt::Tensor>> rb_h3;     // [N][n_recur] [M,dff] dense MLP only
    std::vector<std::vector<mt::Tensor>> rb_g;      // [N][n_recur] [M,dff] dense MLP only

    // Pool MLP intermediates (per step, per site: each recur site routes
    // independently, so sharing one cache per step would let the last site
    // overwrite earlier sites' routing)
    std::vector<std::vector<mt::Tensor>> rb_xn2;                     // [N][R] [M,d]  rms_norm(h, ln2)
    std::vector<std::vector<minagi::PoolBackwardCache>> rb_pool_cache; // [N][R] routing info for backward

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
    // Const overload: a gradient set is read-only once backward() has filled it,
    // and callers that only inspect it should not need a mutable copy.
    const mt::Tensor* get(const std::string& name) const {
        for (const auto& g : grads) if (g.first == name) return &g.second;
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
                   std::vector<model::KVCache>& caches,
                   std::vector<std::vector<float>>* d_lam_out = nullptr);

// Full forward + loss + backward in one call. Returns loss.
float compute_loss_and_grads(const Config& cfg, Coder& coder,
                             const mt::Tensor& idx,
                             const mt::Tensor& targets,
                             int64_t pos_offset,
                             std::vector<model::KVCache>& caches,
                             Gradients& grads_out,
                             BackwardCache* cache_out = nullptr);

}  // namespace minagi::backward
