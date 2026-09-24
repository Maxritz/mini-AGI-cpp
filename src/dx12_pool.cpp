// src/dx12_pool.cpp
// DirectX 12 acceleration for PooledMLP expert matmul.
//
// Routing/top-k stays on CPU (replicates pool.cpp:pool_mlp_forward exactly).
// The expert matmul kernels (SiLU gate * value @ W2^T) are dispatched to GPU
// via the HLSL kernels in pool.hlsl. A CPU fallback path is provided when
// DX12 is unavailable, producing bit-identical results.
//
// The HLSL kernels expect:
//   Pass 0 (W1+W3 matmul): g_token_input [total_kept*D], g_w1_weights [resident*d_ff*D],
//                          g_w3_weights [resident*d_ff*D], g_assignments [total_kept*3],
//                          output to g_scratch_out [total_kept*2*d_ff]
//   Pass 1 (SiLU mul):     reads g_scratch_out, writes h = silu(a1)*a3 back to first d_ff
//   Pass 2 (W2 matmul):    reads g_scratch_out h region, g_w2_weights [resident*D*d_ff],
//                          writes to g_scratch_out secondary region [total_kept*D]
//   Pass 3 (scatter-add):  reads assignments + mlp output, accumulates to [N*D]
//
// On Windows, the real DX12 path would call ID3D12GraphicsCommandList::Dispatch.
// Here we use a CPU kernel that computes the same thing for parity/testing.

#include "dx12_pool.hpp"
#include "pool.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <vector>
#include <utility>

#ifdef _WIN32
  #include "dx12_engine.hpp"
  #include "dx12_context.hpp"
  #include <iostream>
#endif

namespace minagi {

namespace {

inline float silu1(float x) {
  if (x == 0.0f) return 0.0f;
  if (x > 0.0f) {
    const float e = std::exp(-x);
    return x / (1.0f + e);
  }
  const float e = std::exp(x);
  return x * e / (1.0f + e);
}

// CPU fallback for the expert matmul kernel.
// This exactly replicates the matmul logic in pool.cpp lines 285-326.
// When DX12 is available, this is replaced by GPU dispatch.
void cpu_expert_matmul(
    const float* xp,             // [N, D] input
    const float* w1p,            // [resident, d_ff, D]
    const float* w3p,            // [resident, d_ff, D]
    const float* w2p,            // [resident, D, d_ff]
    const std::vector<int>& /*rows*/,
    int /*N*/, int D, int dff,
    const std::vector<std::vector<std::pair<int64_t, float>>>& items,
    std::vector<std::vector<float>>& y_slots)  // output per-slot
{
  std::vector<float> h_buf;
  std::vector<float> xrows_buf;

  // Compute max capacity across all slots
  int64_t cap = 0;
  for (const auto& item : items) {
    cap = std::max(cap, static_cast<int64_t>(item.size()));
  }
  int64_t cap_int = static_cast<int64_t>(cap);

  h_buf.assign(static_cast<size_t>(cap_int * dff), 0.0f);
  xrows_buf.assign(static_cast<size_t>(cap_int * D), 0.0f);

  for (size_t s = 0; s < items.size(); ++s) {
    const size_t cs = items[s].size();
    if (cs == 0) continue;

    // Gather token rows into xrows
    for (size_t i = 0; i < cs; ++i) {
      const int64_t t = items[s][i].first;
      std::memcpy(xrows_buf.data() + i * static_cast<size_t>(D),
                  xp + static_cast<size_t>(t) * static_cast<size_t>(D),
                  static_cast<size_t>(D) * sizeof(float));
    }

    // W1 (gate) + W3 (value) matmul + SiLU multiply
    for (size_t i = 0; i < cs; ++i) {
      const float* xrow = xrows_buf.data() + i * static_cast<size_t>(D);
      for (int64_t f = 0; f < dff; ++f) {
        double a1 = 0.0;
        double a3 = 0.0;
        const float* r1 = w1p + (static_cast<size_t>(s) * static_cast<size_t>(dff) + static_cast<size_t>(f)) * static_cast<size_t>(D);
        const float* r3 = w3p + (static_cast<size_t>(s) * static_cast<size_t>(dff) + static_cast<size_t>(f)) * static_cast<size_t>(D);
        for (int64_t dd = 0; dd < D; ++dd) {
          const double xv = static_cast<double>(xrow[dd]);
          a1 += xv * static_cast<double>(r1[dd]);
          a3 += xv * static_cast<double>(r3[dd]);
        }
        h_buf[i * static_cast<size_t>(dff) + f] =
            silu1(static_cast<float>(a1)) * static_cast<float>(a3);
      }
    }

    // W2 (down-proj) matmul: y = h @ W2[s]^T
    std::vector<float>& ys = y_slots[s];
    ys.assign(cs * static_cast<size_t>(D), 0.0f);
    for (size_t i = 0; i < cs; ++i) {
      const float* hrow = h_buf.data() + i * static_cast<size_t>(dff);
      float* yrow = ys.data() + i * static_cast<size_t>(D);
      for (int64_t dd = 0; dd < D; ++dd) {
        double acc = 0.0;
        const float* w2r = w2p + (static_cast<size_t>(s) * static_cast<size_t>(D) + static_cast<size_t>(dd)) * static_cast<size_t>(dff);
        for (int64_t f = 0; f < dff; ++f) {
          acc += static_cast<double>(hrow[f]) * static_cast<double>(w2r[f]);
        }
        yrow[dd] = static_cast<float>(acc);
      }
    }
  }
}

}  // namespace

// Dx12PoolContext: holds the DX12 engine for GPU-accelerated pool dispatch.
// The full struct definition is in dx12_pool.hpp; the engine pointer is
// a void* here to avoid pulling the full engine header at parse time.

Dx12PoolContext* dx12_pool_create() {
#ifdef _WIN32
  // Real implementation: create D3D12 device, compile pool.hlsl,
  // create root signatures and PSOs.
  auto ctx = new Dx12PoolContext();
  auto* eng = new dx12::Dx12ComputeEngine();

   // Try to initialize the engine with the shader directory.
  // Try multiple paths to find the shaders.
  std::string shader_dir;
  const char* paths[] = {
    "src/shaders",
    "../src/shaders",
    "./shaders",
    "shaders",
  };
  for (const char* p : paths) {
    std::string test_path = std::string(p) + "/pool.hlsl";
    std::ifstream f(test_path);
    if (f.good()) {
      shader_dir = p;
      break;
    }
  }
  if (shader_dir.empty()) {
    shader_dir = "src/shaders";  // fallback (will fail to compile)
  }
  std::cout << "DX12_POOL: shader_dir = " << shader_dir << std::endl;
  if (eng->init(shader_dir)) {
    ctx->engine = eng;
    ctx->valid = true;
    // Load PSOs for each pool shader entry point
    ctx->pso_matmul_w1w3 = eng->load_compute_pso("main_pool_matmul_w1w3");
    ctx->pso_silu_mul  = eng->load_compute_pso("main_pool_silu_mul");
    ctx->pso_matmul_w2 = eng->load_compute_pso("main_pool_matmul_w2");
    ctx->pso_scatter   = eng->load_compute_pso("main_pool_scatter_add");
    if (ctx->pso_matmul_w1w3 < 0 || ctx->pso_silu_mul < 0 ||
        ctx->pso_matmul_w2 < 0 || ctx->pso_scatter < 0) {
      ctx->valid = false;
    }
  }
  if (!ctx->valid) {
    std::cout << "DX12_POOL: init failed or PSO compilation failed" << std::endl;
    if (ctx->pso_matmul_w1w3 < 0) std::cout << "DX12_POOL: matmul_w1w3 PSO failed" << std::endl;
    if (ctx->pso_silu_mul < 0) std::cout << "DX12_POOL: silu_mul PSO failed" << std::endl;
    if (ctx->pso_matmul_w2 < 0) std::cout << "DX12_POOL: matmul_w2 PSO failed" << std::endl;
    if (ctx->pso_scatter < 0) std::cout << "DX12_POOL: scatter PSO failed" << std::endl;
    delete static_cast<dx12::Dx12ComputeEngine*>(ctx->engine);
    ctx->engine = nullptr;
    delete ctx;
    return nullptr;
  }
  return ctx;
#else
  return nullptr;
#endif
}

void dx12_pool_destroy(Dx12PoolContext* ctx) {
  if (!ctx) return;
#ifdef _WIN32
  if (ctx->engine) {
    delete static_cast<dx12::Dx12ComputeEngine*>(ctx->engine);
    ctx->engine = nullptr;
  }
#endif
  delete ctx;
}

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
    double capacity_factor)
{
  Dx12PoolResult result;
  result.y = mt::Tensor();  // rank-0 empty on error
  result.routed = 0;
  result.dropped = 0;

  // Validate inputs (same checks as pool_mlp_forward)
  if (x.dtype != mt::DType::FP32 || x.shape.rank != 2 ||
      x.shape.d[0] <= 0 || x.shape.d[1] <= 0) {
    return result;
  }
  if (router_w.dtype != mt::DType::FP32 || router_w.shape.rank != 2) {
    return result;
  }
  if (depth_emb.dtype != mt::DType::FP32 || depth_emb.shape.rank != 1) {
    return result;
  }

  const int64_t N = x.shape.d[0];
  const int64_t D = x.shape.d[1];
  const int n = static_cast<int>(rows.size());
  if (n <= 0) {
    mt::Shape s; s.rank = 2; s.d[0] = N; s.d[1] = D;
    result.y = mt::make(s, mt::DType::FP32, 0.0f);
    return result;
  }
  if (depth_emb.shape.d[0] != D || router_w.shape.d[1] != D) {
    return result;
  }

  const int64_t dff = W1.shape.d[1];

  // Validate weight shapes (same as pool.cpp)
  if (gate.dtype != mt::DType::FP32 || gate.shape.rank != 1 ||
      W1.dtype != mt::DType::FP32 || W1.shape.rank != 3 ||
      W3.dtype != mt::DType::FP32 || W3.shape.rank != 3 ||
      W2.dtype != mt::DType::FP32 || W2.shape.rank != 3 ||
      (int64_t)slots.size() != n || W1.shape.d[0] != n || W3.shape.d[0] != n ||
      W2.shape.d[0] != n || W1.shape.d[2] != D || W3.shape.d[2] != D ||
      W2.shape.d[1] != D) {
    return result;
  }
  if (W2.shape.d[2] != dff) {
    return result;
  }

  const int k = top_k > 0 ? std::min(top_k, n) : 0;
  if (k <= 0) {
    mt::Shape s; s.rank = 2; s.d[0] = N; s.d[1] = D;
    result.y = mt::make(s, mt::DType::FP32, 0.0f);
    return result;
  }

  const float* xp = x.ptr<float>();
  const float* dp = depth_emb.ptr<float>();
  const float* rw = router_w.ptr<float>();
  const float* gp = gate.ptr<float>();
  const float* w1p = W1.ptr<float>();
  const float* w3p = W3.ptr<float>();
  const float* w2p = W2.ptr<float>();

  // ---- Phase 1: CPU routing (same as pool_mlp_forward) ----

  // logits [N, n] = (x + depth_emb) @ w^T
  std::vector<float> logits(static_cast<size_t>(N) * static_cast<size_t>(n));
  for (int64_t r = 0; r < N; ++r) {
    for (int c = 0; c < n; ++c) {
      double acc = 0.0;
      const float* wrow = rw + static_cast<size_t>(rows[c]) * static_cast<size_t>(D);
      for (int64_t dd = 0; dd < D; ++dd) {
        const double xv = static_cast<double>(xp[static_cast<size_t>(r) * D + dd] + dp[dd]);
        acc += xv * static_cast<double>(wrow[dd]);
      }
      logits[static_cast<size_t>(r) * n + c] = static_cast<float>(acc);
    }
  }

  // softmax(logits, -1)
  std::vector<float> probs(static_cast<size_t>(N) * static_cast<size_t>(n));
  for (int64_t r = 0; r < N; ++r) {
    const float* row = logits.data() + static_cast<size_t>(r) * n;
    float mx = -std::numeric_limits<float>::infinity();
    for (int c = 0; c < n; ++c) {
      if (row[c] > mx) mx = row[c];
    }
    double sum = 0.0;
    for (int c = 0; c < n; ++c) {
      const float e = static_cast<float>(std::exp(static_cast<double>(row[c]) - static_cast<double>(mx)));
      probs[static_cast<size_t>(r) * n + c] = e;
      sum += e;
    }
    for (int c = 0; c < n; ++c) {
      probs[static_cast<size_t>(r) * n + c] =
          static_cast<float>(probs[static_cast<size_t>(r) * n + c] / sum);
    }
  }

  // top-k (descending value, ascending index on ties)
  std::vector<float> wv(static_cast<size_t>(N) * static_cast<size_t>(k));
  std::vector<int64_t> idx(static_cast<size_t>(N) * static_cast<size_t>(k));
  for (int64_t r = 0; r < N; ++r) {
    std::vector<std::pair<float, int>> cand(static_cast<size_t>(n));
    for (int c = 0; c < n; ++c) {
      cand[static_cast<size_t>(c)] = {probs[static_cast<size_t>(r) * n + c], c};
    }
    std::partial_sort(cand.begin(), cand.begin() + k, cand.end(),
                      [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                        if (a.first != b.first) return a.first > b.first;
                        return a.second < b.second;
                      });
    for (int i = 0; i < k; ++i) {
      wv[static_cast<size_t>(r) * k + i] = cand[static_cast<size_t>(i)].first;
      idx[static_cast<size_t>(r) * k + i] = cand[static_cast<size_t>(i)].second;
    }
  }

  // wv = wv / wv.sum(-1); wv = wv * routable_gate()[idx]
  for (int64_t r = 0; r < N; ++r) {
    double sum = 0.0;
    for (int i = 0; i < k; ++i) {
      sum += wv[static_cast<size_t>(r) * k + i];
    }
    for (int i = 0; i < k; ++i) {
      float w = wv[static_cast<size_t>(r) * k + i];
      if (sum != 0.0) {
        w = static_cast<float>(w / sum);
      }
      const int64_t sl = idx[static_cast<size_t>(r) * k + i];
      wv[static_cast<size_t>(r) * k + i] = w * gp[rows[sl]];
    }
  }

  // Routing bookkeeping
  std::vector<long long> slot_hits(static_cast<size_t>(n), 0);
  for (size_t j = 0; j < static_cast<size_t>(N) * static_cast<size_t>(k); ++j) {
    slot_hits[static_cast<size_t>(idx[j])]++;
  }

  // ---- Phase 2: Capacity-bounded batched dispatch ----
  const int64_t total = N * static_cast<int64_t>(k);
  struct Assignment {
    int64_t slot = 0;
    int64_t token = 0;
    float weight = 0.0f;
    int64_t in_slot = 0;
    bool kept = true;
  };
  std::vector<Assignment> asn(static_cast<size_t>(total));
  for (int64_t j = 0; j < total; ++j) {
    Assignment& a = asn[static_cast<size_t>(j)];
    a.slot = idx[static_cast<size_t>(j)];
    a.token = j / static_cast<int64_t>(k);
    a.weight = wv[static_cast<size_t>(j)];
  }
  std::stable_sort(asn.begin(), asn.end(),
                   [](const Assignment& a, const Assignment& b) {
                     return a.slot < b.slot;
                   });

  std::vector<int64_t> counts(static_cast<size_t>(n), 0);
  int64_t cap = 0;
  for (const Assignment& a : asn) {
    counts[static_cast<size_t>(a.slot)]++;
    cap = std::max(cap, counts[static_cast<size_t>(a.slot)]);
  }

  // Per-slot position of each assignment, in sorted order
  {
    std::vector<int64_t> run(static_cast<size_t>(n), 0);
    for (Assignment& a : asn) {
      a.in_slot = run[static_cast<size_t>(a.slot)]++;
    }
  }

  int64_t dropped = 0;
  if (capacity_factor != 0.0) {
    const int64_t limit = std::max<int64_t>(1, static_cast<int64_t>(
        std::ceil(capacity_factor * static_cast<double>(total) / static_cast<double>(n))));
    if (cap > limit) {
      for (Assignment& a : asn) {
        a.kept = a.in_slot < limit;
        if (!a.kept) ++dropped;
      }
    } else {
      for (Assignment& a : asn) {
        a.kept = true;
      }
    }
  }
  result.routed = total;
  result.dropped = dropped;

  // Build per-slot items (slot-major, kept only)
  std::vector<std::vector<std::pair<int64_t, float>>> items(static_cast<size_t>(n));
  for (const Assignment& a : asn) {
    if (a.kept) {
      items[static_cast<size_t>(a.slot)].push_back({a.token, a.weight});
    }
  }

  // ---- Phase 3: Expert matmul (GPU or CPU fallback) ----
  std::vector<std::vector<float>> y_slots(static_cast<size_t>(n));

#ifdef _WIN32
  if (ctx && ctx->valid && ctx->engine) {
    // GPU path: prepare buffers and dispatch shaders
    // Flatten assignments into a tight buffer: {slot:u32, token:u32, weight:f32, pad:u32}
    std::vector<uint32_t> assignment_buf(static_cast<size_t>(total) * 4, 0);
    for (int64_t j = 0; j < total; ++j) {
      const size_t base = static_cast<size_t>(j) * 4;
      assignment_buf[base]     = static_cast<uint32_t>(asn[static_cast<size_t>(j)].slot);
      assignment_buf[base + 1] = static_cast<uint32_t>(asn[static_cast<size_t>(j)].token);
      std::memcpy(&assignment_buf[base + 2], &asn[static_cast<size_t>(j)].weight, sizeof(float));
    }

    // Gather token_input [total_kept, D]
    int64_t total_kept = total - dropped;
    std::vector<float> token_input(static_cast<size_t>(total_kept) * static_cast<size_t>(D), 0.0f);
    int64_t out_idx = 0;
    for (size_t s = 0; s < items.size(); ++s) {
      const size_t cs = items[s].size();
      for (size_t i = 0; i < cs; ++i) {
        const int64_t t = items[s][i].first;
        std::memcpy(token_input.data() + out_idx * static_cast<size_t>(D),
                    xp + static_cast<size_t>(t) * static_cast<size_t>(D),
                    static_cast<size_t>(D) * sizeof(float));
        ++out_idx;
      }
    }

    dx12::Dx12ComputeEngine* eng = static_cast<dx12::Dx12ComputeEngine*>(ctx->engine);

    // Create GPU buffers — all as UAV since compute shaders read via UAV
    auto buf_token    = eng->create_buffer(static_cast<uint64_t>(total) * D * sizeof(float), true);
    auto buf_w1       = eng->create_buffer(static_cast<uint64_t>(n) * dff * D * sizeof(float), true);
    auto buf_w3       = eng->create_buffer(static_cast<uint64_t>(n) * dff * D * sizeof(float), true);
    auto buf_w2       = eng->create_buffer(static_cast<uint64_t>(n) * D * dff * sizeof(float), true);
    auto buf_scratch  = eng->create_buffer(static_cast<uint64_t>(total) * (2 * dff + D) * sizeof(float), true);
    auto buf_assign   = eng->create_buffer(static_cast<uint64_t>(total) * 4 * sizeof(uint32_t), true);
    auto buf_scatter  = eng->create_buffer(static_cast<uint64_t>(N) * D * sizeof(float), true);

    // Upload data
    eng->upload_buffer(buf_token, mt::Tensor());  // placeholder, real upload below
    // For real upload we need to create a temporary tensor
    {
      mt::Shape sh; sh.rank = 1; sh.d[0] = total * D;
      mt::Tensor tmp = mt::make(sh, mt::DType::FP32, 0.0f);
      std::memcpy(tmp.ptr<float>(), token_input.data(), token_input.size() * sizeof(float));
      eng->upload_buffer(buf_token, tmp);
    }
    {
      mt::Shape sh; sh.rank = 1; sh.d[0] = total * 4;
      mt::Tensor tmp = mt::make(sh, mt::DType::FP32, 0.0f);
      std::memcpy(tmp.ptr<float>(), assignment_buf.data(), assignment_buf.size() * sizeof(float));
      eng->upload_buffer(buf_assign, tmp);
    }
    {
      mt::Tensor w1t = mt::to_fp32(W1);
      eng->upload_buffer(buf_w1, w1t);
    }
    {
      mt::Tensor w3t = mt::to_fp32(W3);
      eng->upload_buffer(buf_w3, w3t);
    }
    {
      mt::Tensor w2t = mt::to_fp32(W2);
      eng->upload_buffer(buf_w2, w2t);
    }

    // Zero the scatter buffer
    {
      std::vector<float> zeros(static_cast<size_t>(N) * D, 0.0f);
      mt::Shape sh; sh.rank = 1; sh.d[0] = N * D;
      mt::Tensor zt = mt::make(sh, mt::DType::FP32, 0.0f);
      std::memcpy(zt.ptr<float>(), zeros.data(), zeros.size() * sizeof(float));
      eng->upload_buffer(buf_scatter, zt);
    }

    // Reset command list
    eng->reset();

    // Pass 1: Set PSO for W1+W3 matmul
    eng->set_pso(ctx->pso_matmul_w1w3);
    // Root param 0: descriptor table (12 UAV/SRV slots)
    // We bind buffers via root descriptors (SetComputeRootUnorderedAccessView/SetComputeRootShaderResourceView)
    eng->set_uav(0, buf_token);    // token_input
    eng->set_uav(1, buf_w1);       // w1_weights
    eng->set_uav(2, buf_w3);       // w3_weights
    eng->set_uav(4, buf_scratch);  // scratch_out
    eng->set_uav(5, buf_assign);   // assignments

    // Push constants for pool shader
    struct PoolPushConstants {
      uint32_t dispatch_id;
      uint32_t D;
      uint32_t d_ff;
      uint32_t N;
      uint32_t total_kept;
      uint32_t resident;
      uint32_t mode;
      uint32_t pad0;
    } ppc{};
    ppc.dispatch_id = 0;
    ppc.D = static_cast<uint32_t>(D);
    ppc.d_ff = static_cast<uint32_t>(dff);
    ppc.N = static_cast<uint32_t>(N);
    ppc.total_kept = static_cast<uint32_t>(total_kept);
    ppc.resident = static_cast<uint32_t>(n);

    eng->push_constants(&ppc, sizeof(ppc) / sizeof(uint32_t));

    // Dispatch W1+W3 matmul
    eng->dispatch((static_cast<uint32_t>(total_kept) * static_cast<uint32_t>(dff) + 63) / 64, 1, 1);

    // Pass 2: SiLU mul
    eng->set_pso(ctx->pso_silu_mul);
    eng->push_constants(&ppc, sizeof(ppc) / sizeof(uint32_t));
    eng->dispatch((static_cast<uint32_t>(total_kept) * static_cast<uint32_t>(dff) + 63) / 64, 1, 1);

    // Pass 3: W2 matmul
    eng->set_pso(ctx->pso_matmul_w2);
    eng->set_uav(3, buf_w2);       // w2_weights
    eng->set_uav(5, buf_assign);   // assignments (for slot lookup)
    eng->push_constants(&ppc, sizeof(ppc) / sizeof(uint32_t));
    eng->dispatch((static_cast<uint32_t>(total_kept) * static_cast<uint32_t>(D) + 63) / 64, 1, 1);

    // Pass 4: Scatter-add
    eng->set_pso(ctx->pso_scatter);
    eng->set_uav(6, buf_scatter);  // scatter_add output
    eng->set_uav(5, buf_assign);   // assignments (for token_idx, weight)
    eng->push_constants(&ppc, sizeof(ppc) / sizeof(uint32_t));
    eng->dispatch((static_cast<uint32_t>(total_kept) * static_cast<uint32_t>(D) + 63) / 64, 1, 1);

    // Flush and read back
    eng->flush();

    // Read back scatter_add buffer
    mt::Shape sh; sh.rank = 2; sh.d[0] = N; sh.d[1] = D;
    result.y = eng->readback_buffer(buf_scatter, sh, mt::DType::FP32);
  } else
#endif
  {
    // CPU fallback / actual computation
    cpu_expert_matmul(xp, w1p, w3p, w2p, rows,
                      static_cast<int>(N), static_cast<int>(D), static_cast<int>(dff),
                      items, y_slots);

    // ---- Phase 4: Scatter-add weighted back to [N, D] ----
    mt::Shape s2; s2.rank = 2; s2.d[0] = N; s2.d[1] = D;
    result.y = mt::make(s2, mt::DType::FP32, 0.0f);
    float* rp = result.y.ptr<float>();

    // Use double accumulation for scatter-add parity (matches pool.cpp)
    std::vector<double> acc_rows(static_cast<size_t>(N) * static_cast<size_t>(D), 0.0);
    for (int i = 0; i < n; ++i) {
      const size_t cs = items[static_cast<size_t>(i)].size();
      const float* ys = y_slots[static_cast<size_t>(i)].data();
      for (size_t j = 0; j < cs; ++j) {
        const int64_t t = items[static_cast<size_t>(i)][j].first;
        const float w = items[static_cast<size_t>(i)][j].second;
        double* outrow = acc_rows.data() + static_cast<size_t>(t) * static_cast<size_t>(D);
        for (int64_t dd = 0; dd < D; ++dd) {
          outrow[dd] += static_cast<double>(w) * static_cast<double>(ys[j * static_cast<size_t>(D) + static_cast<size_t>(dd)]);
        }
      }
    }
    for (int64_t j = 0; j < N * D; ++j) {
      rp[j] = static_cast<float>(acc_rows[static_cast<size_t>(j)]);
    }
  }

  return result;
}

}  // namespace minagi
