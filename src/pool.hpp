// src/pool.hpp
// Shared-expert-pool forward path (minagi/pool.py PooledMLP, inference, fp32):
// the abstract pool floor surface plus the mojito router/dispatch of section 4.
#pragma once
#include "tensor.hpp"
#include <vector>

namespace minagi {

// Abstract floor-pool surface the dispatch routes against. A paged pool
// (PagedPool) implements this over its resident slot tensors.
struct Pool {
  virtual ~Pool() = default;
  virtual int n_experts() const = 0;                                // 6
  virtual int n_routable() const = 0;                               // resident (3)
  virtual int router_rows() const = 0;                              // 6
  virtual const std::vector<int>& slots() const = 0;                // slot -> expert uid (-1 empty), size resident
  virtual const mt::Tensor& gate() const = 0;                       // [n_experts] FP32
  virtual const mt::Tensor& w1() const = 0;                         // [resident, d_ff, d]
  virtual const mt::Tensor& w3() const = 0;
  virtual const mt::Tensor& w2() const = 0;                         // [resident, d, d_ff]
   virtual void note_use(const std::vector<long long>& slot_hits) const = 0;  // per-slot hit counts -> use by uid, age += 1
};

struct RouteStats {
  long long routed = 0;   // assignments the router produced (before any drop)
  long long dropped = 0;  // assignments dropped by the capacity bound
};

// Saved routing info for backward pass through pool_mlp_forward.
// Each entry: (token_row, slot_index, weight, kept)
struct PoolDispatchInfo {
  int64_t token;     // token row index
  int slot;           // resident slot index chosen by routing
  float weight;       // routed weight (after gate)
  bool kept;          // survived capacity drop
};

struct PoolBackwardCache {
  // All assignments (kept only) for reconstructing the scatter/scatter-grad
  std::vector<PoolDispatchInfo> kept;
  // logits [N, n_routable] for softmax backward
  std::vector<float> logits;
  int64_t N = 0;
  int n_routable = 0;
  int top_k = 0;
  // x + (x + depth_emb) for router grad
  std::vector<float> x_shifted;
};

// The mojito dispatch of section 4, exactly. x [T,d] FP32; router_w
// [n_experts,d]; depth_emb [d]; returns [T,d]. top_k and capacity_factor as
// passed. stats is optional (accumulates routed/dropped). Calls pool.note_use.
// If cache != nullptr, saves routing info for pool_mlp_backward.
// Returns a rank-0 tensor on bad input (non-FP32 / non-2D x, or a router_w /
// depth_emb shape mismatch).
mt::Tensor pool_mlp_forward(const Pool& pool, const mt::Tensor& x,
                            const mt::Tensor& router_w, const mt::Tensor& depth_emb,
                            int top_k, double capacity_factor, RouteStats* stats,
                            PoolBackwardCache* cache = nullptr);

// Backward pass for pool_mlp_forward. Given the cache from forward and d_output
// [N, D] (gradient w.r.t. the output), returns d_x [N, D] and fills:
//   - d_router_w [n_experts, D] (gradient w.r.t. router weight)
//   - d_depth_emb [D] (gradient w.r.t. depth embedding)
mt::Tensor pool_mlp_backward(const Pool& pool,
                             const PoolBackwardCache& cache,
                             const mt::Tensor& d_output,
                             const mt::Tensor& router_w,
                             const mt::Tensor& depth_emb,
                             int top_k, double capacity_factor,
                             mt::Tensor* d_router_w,
                             mt::Tensor* d_depth_emb);

}  // namespace minagi