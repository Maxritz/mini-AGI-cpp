// src/dx12_dense.hpp
// DirectX 12 compute shader wrapper for mini-AGI dense transformer forward pass.
// Mirrors the CPU path: tensor.cpp (matmul, rms_norm, softmax, rope, silu, gelu)
// and model.cpp/recur.cpp (embedding_gather, attention, mlp, residuals).
//
// Standard 12-binding DSL:
//   binding 0 = activation (read/write, used as input to each kernel)
//   binding 1 = qkv_w / weight A
//   binding 2 = proj_w / weight B
//   binding 3 = ln1_w / ln2_w / 1D weight
//   binding 4 = kv_cache_k
//   binding 5 = kv_cache_v
//   binding 6 = cos (RoPE)
//   binding 7 = sin (RoPE)
//   binding 8 = output logits
//   binding 9 = scratch
//   binding 10 = mlp_w1 / w3 / additional weight
//   binding 11 = mlp_w2 / head_w / additional weight
//
// Push constants (32-bit root constants): M, N, K, T, H, hd, pos_offset,
// kv_len, cached_T, eps, inv_sqrt_hd, stride3, d_model, d_ff, dispatch_id, mode

#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <type_traits>
#include "model.hpp"
#include "tensor.hpp"

#ifdef _WIN32
#include <d3d12.h>

namespace dx12_dense {

// Push constant block (32-bit values, matches HLSL cbuffer PushConstants).
// On the C++ side we pass this via ID3D12GraphicsCommandList::SetComputeRoot32BitConstants
// with root parameter index 1 (root param 0 = descriptor table).
struct alignas(4) DensePushConstants
{
    uint32_t dispatch_id;    // which shader to run
    uint32_t M;              // rows of activation
    uint32_t N;              // cols / output dim
    uint32_t K;              // inner dim
    uint32_t T;              // sequence length
    uint32_t H;              // num heads
    uint32_t hd;             // head dim
    uint32_t pos_offset;     // position offset for RoPE
    uint32_t kv_len;         // KV cache length
    uint32_t cached_T;       // cached tokens
    uint32_t head_dim;       // = hd
    uint32_t d_model;        // model dim
    uint32_t d_ff;           // MLP intermediate
    uint32_t stride3;        // 3 * d_model
    float    eps;            // RMSNorm eps
    float    inv_sqrt_hd;    // 1/sqrt(hd)
    uint32_t mode;           // 0=plain, 1=with cache
    uint32_t _pad0;
    uint32_t _pad1;
    uint32_t _pad2;
};
static_assert(sizeof(DensePushConstants) == 20 * sizeof(uint32_t),
              "DensePushConstants must be 20 uint32_t for root constants");

// Dispatch IDs for the push constant dispatch_id field.
// Matches HLSL kernel numbering.
enum class DispatchId : uint32_t
{
    MatMulFP32      = 0,
    RMSNorm         = 1,
    Softmax         = 2,
    RoPE            = 3,
    EmbeddingGather = 4,
    SiLU            = 5,
    GELU            = 6,
    SiLUGate        = 7,
    Add             = 8,
    AttnScores      = 9,
    AttnWeightedSum = 10,
    ElementwiseCopy = 11,
};

// Simple GPU buffer wrapper: opaque handle to an ID3D12Resource with its
// GPU virtual address. The real buffer management lives in dx12_engine.hpp.
struct GpuBuffer
{
    ID3D12Resource* resource = nullptr;
    D3D12_GPU_VIRTUAL_ADDRESS gpu_va = 0;
    uint32_t byte_size = 0;
    bool is_uav = false;
};

// CPU-side dispatch wrapper. Each function mirrors the CPU math exactly.
// Uses root descriptor binding (SetComputeRootShaderResourceView /
// SetComputeRootUnorderedAccessView) per dispatch for immediate binding.
class DenseComputeContext
{
public:
    DenseComputeContext() = default;
    ~DenseComputeContext() = default;

    // Bind a GPU buffer to a slot (0-11).
    void bind_buffer(int slot, const GpuBuffer& buf);

    // Set push constants for the next dispatch
    void set_push_constants(const DensePushConstants& pc);

    // Dispatch matmul: C = A * B^T
    // A from binding 1 [M,K], B from binding 2 [N,K], output to binding 0 [M,N]
    void dispatch_matmul(uint32_t M, uint32_t N, uint32_t K,
                         ID3D12GraphicsCommandList* cmd_list);

    // RMSNorm: x from binding 0, weight from binding 3, output to binding 0
    void dispatch_rmsnorm(uint32_t rows, uint32_t cols, float eps,
                          ID3D12GraphicsCommandList* cmd_list);

    // Softmax: row-wise on binding 0 [M,N], output to binding 0
    void dispatch_softmax(uint32_t rows, uint32_t cols,
                          ID3D12GraphicsCommandList* cmd_list);

    // RoPE: x from binding 0 [B*H*T, hd], cos/sin from bindings 6/7 [T, hd/2]
    void dispatch_rope(uint32_t B, uint32_t H, uint32_t T, uint32_t hd,
                       uint32_t pos_offset, ID3D12GraphicsCommandList* cmd_list);

    // Embedding gather: emb from binding 1 [V,d], idx from binding 0 [B*T], output to binding 0
    void dispatch_embedding_gather(uint32_t V, uint32_t d, uint32_t M,
                                   ID3D12GraphicsCommandList* cmd_list);

    // SiLU elementwise: x from binding 0, output to binding 0
    void dispatch_silu(uint32_t n, ID3D12GraphicsCommandList* cmd_list);

    // GELU elementwise: x from binding 0, output to binding 0
    void dispatch_gelu(uint32_t n, ID3D12GraphicsCommandList* cmd_list);

    // SiLU * Gate: x from binding 0, gate from binding 3, output to binding 0
    void dispatch_silu_gate(uint32_t n, ID3D12GraphicsCommandList* cmd_list);

    // Add residual: a from binding 0, b from binding 3, output to binding 0
    void dispatch_add(uint32_t n, ID3D12GraphicsCommandList* cmd_list);

    // Attention scores: Q from binding 0 [B*H*T, hd], K from binding 1 [B*H*kv_len, hd]
    // scores to binding 9 [B*H*T, kv_len], causal masked
    void dispatch_attn_scores(uint32_t M, uint32_t kv_len, uint32_t T,
                              uint32_t hd, float inv_sqrt_hd,
                              ID3D12GraphicsCommandList* cmd_list);

    // Attention weighted sum: scores from binding 9, V from binding 1
    // output to binding 0 [B*H*T, hd]
    void dispatch_attn_wsum(uint32_t M, uint32_t kv_len, uint32_t hd,
                            ID3D12GraphicsCommandList* cmd_list);

    // Elementwise copy: binding 0 -> binding 0
    void dispatch_copy(uint32_t n, ID3D12GraphicsCommandList* cmd_list);

    // ---- High-level composite operations ----

    // Attention residual: mirrors recur.cpp::attn_residual
    void dispatch_attn_residual(
        mt::Tensor& x,
        const mt::Tensor& qkv_w,
        const mt::Tensor& proj_w,
        const mt::Tensor& ln1_w,
        const mt::Tensor& cos,
        const mt::Tensor& sin,
        model::KVCache& cache,
        uint32_t T, uint32_t H, uint32_t hd, uint32_t d,
        uint32_t pos_offset, ID3D12GraphicsCommandList* cmd_list);

    // MLP block: mirrors recur.cpp dense_mlp_impl
    void dispatch_mlp(
        mt::Tensor& x,
        const mt::Tensor& w1, const mt::Tensor& w3, const mt::Tensor& w2,
        const mt::Tensor& ln2_w,
        uint32_t T, uint32_t d, uint32_t dff,
        ID3D12GraphicsCommandList* cmd_list);

    // Engine / device access (set externally)
    void set_device(ID3D12Device* dev) { device_ = dev; }

private:
    GpuBuffer buffers_[12];
    DensePushConstants push_constants_{};
    ID3D12Device* device_ = nullptr;
};

// Build compiled shader from .hlsl file
std::vector<uint8_t> compile_hlsl(const std::string& hlsl_path,
                                  const std::string& entry_point,
                                  const std::string& target = "cs_5_0");

} // namespace dx12_dense

#else // !_WIN32

// On non-Windows platforms, provide empty stubs to keep headers parseable.
namespace dx12_dense {
struct DensePushConstants {};
enum class DispatchId : uint32_t {};
struct GpuBuffer {};
class DenseComputeContext {
public:
    void bind_buffer(int, const GpuBuffer&) {}
    void set_push_constants(const DensePushConstants&) {}
    void dispatch_matmul(uint32_t, uint32_t, uint32_t, void*) {}
    void dispatch_rmsnorm(uint32_t, uint32_t, float, void*) {}
    void dispatch_softmax(uint32_t, uint32_t, void*) {}
    void dispatch_rope(uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, void*) {}
    void dispatch_embedding_gather(uint32_t, uint32_t, uint32_t, void*) {}
    void dispatch_silu(uint32_t, void*) {}
    void dispatch_gelu(uint32_t, void*) {}
    void dispatch_silu_gate(uint32_t, void*) {}
    void dispatch_add(uint32_t, void*) {}
    void dispatch_attn_scores(uint32_t, uint32_t, uint32_t, uint32_t, float, void*) {}
    void dispatch_attn_wsum(uint32_t, uint32_t, uint32_t, void*) {}
    void dispatch_copy(uint32_t, void*) {}
    void dispatch_attn_residual(mt::Tensor&, const mt::Tensor&, const mt::Tensor&,
                                const mt::Tensor&, const mt::Tensor&, const mt::Tensor&,
                                model::KVCache&, uint32_t, uint32_t, uint32_t, uint32_t,
                                uint32_t, void*) {}
    void dispatch_mlp(mt::Tensor&, const mt::Tensor&, const mt::Tensor&,
                      const mt::Tensor&, const mt::Tensor&, uint32_t, uint32_t,
                      uint32_t, void*) {}
};
}

#endif // _WIN32
