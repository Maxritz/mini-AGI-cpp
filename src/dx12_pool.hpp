// src/dx12_pool.hpp
// DirectX 12 acceleration for PooledMLP expert matmul.
//
// The routing/top-k stays on CPU (using the existing pool_mlp_forward dispatch
// logic). The GPU handles only the heavy per-expert matmul kernels:
//   silu(xs @ W1^T) * (xs @ W3^T) @ W2^T
// followed by scatter-add of weighted outputs back to [N, d].
//
// The CPU prepares:
//   - token_input: [total_kept, D] — token rows for kept assignments (slot-major)
//   - assignments: [total_kept * 3] — packed (slot, token_idx, weight) per assignment
//   - w1/w3/w2: resident expert weight tensors (already in GPU memory or uploaded)
//
// The GPU runs 4 passes:
//   1. MatMul W1 + W3  -> scratch_out [total_kept, 2*d_ff]
//   2. SiLU * multiply -> scratch_out [total_kept, d_ff]
//   3. MatMul W2       -> scratch_out [total_kept, D]
//   4. Scatter-add      -> output [N, D] (or per-token weighted output for CPU scatter)
//
// The C++ wrapper produces a tensor equivalent to the CPU path's `y_slots` scatter.
//
#pragma once

#include "tensor.hpp"
#include "pool.hpp"
#include <cstdint>
#include <vector>
#include <string>

namespace minagi {

struct Dx12PoolContext {
  bool valid = false;
#ifdef _WIN32
  void* engine = nullptr;  // dx12::Dx12ComputeEngine* (opaque on non-Windows)
  int pso_matmul_w1w3 = -1;
  int pso_silu_mul = -1;
  int pso_matmul_w2 = -1;
  int pso_scatter = -1;
#endif
};
struct Dx12PoolResult {
    mt::Tensor y;           // [N, D] scattered MLP output (pre-residual-add)
    long long routed;
    long long dropped;
};

// Initialize the DX12 device, pipeline, and root signature.
// Returns nullptr on failure (caller falls back to CPU pool_mlp_forward).
Dx12PoolContext* dx12_pool_create();

// Destroy the DX12 context.
void dx12_pool_destroy(Dx12PoolContext* ctx);

// Run the GPU-accelerated expert matmul given the same routing that CPU
// pool_mlp_forward would have done. The caller (pool_mlp_forward_dx12) must
// replicate the CPU top-k, softmax, capacity, and assignment logic.
//
// Parameters (all pre-computed by CPU routing):
//   x            -- [N, D] input (FP32)
//   router_w     -- [n_experts, D] router weight (FP32)
//   depth_emb    -- [D] depth embedding (FP32)
//   gate         -- [n_experts] gate values (FP32)
//   W1           -- [resident, d_ff, D] (FP32)
//   W3           -- [resident, d_ff, D] (FP32)
//   W2           -- [resident, D, d_ff] (FP32)
//   slots        -- slot -> expert uid mapping (size = resident)
//   rows         -- resident_rows() output (slot -> expert uid, clamped to 0)
//   top_k        -- number of experts to route each token to
//   capacity_factor -- capacity multiplier (0 = no drop)
//
// Returns the [N, D] output tensor, or a rank-0 tensor on error.
Dx12PoolResult dx12_pool_mlp_forward(
    Dx12PoolContext* ctx,
    const mt::Tensor& x,
    const mt::Tensor& router_w,
    const mt::Tensor& depth_emb,
    const mt::Tensor& gate,
    const mt::Tensor& W1,
    const mt::Tensor& W3,
    const mt::Tensor& W2,
    const std::vector<int>& slots,
    const std::vector<int>& rows,
    int top_k,
    double capacity_factor);

// Optional: a wrapper that tries DX12 and falls back to CPU pool_mlp_forward.
// Returns true if DX12 was used.
bool pool_mlp_forward_dx12(Dx12PoolContext* ctx, const Pool& pool,
                           const mt::Tensor& x, const mt::Tensor& router_w,
                           const mt::Tensor& depth_emb, int top_k,
                           double capacity_factor, RouteStats* stats,
                           mt::Tensor& out);

}  // namespace minagi
