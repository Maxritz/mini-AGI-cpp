// src/dx12_dense.cpp
// DirectX 12 compute shader wrapper for mini-AGI dense transformer forward pass.
// Mirrors: tensor.cpp (matmul, rms_norm, softmax, rope, silu, gelu),
//          model.cpp (embedding_gather, attn, mlp, forward),
//          recur.cpp:79-251 (attn_residual, dense_mlp_impl).
//
// Build under #ifdef _WIN32 only (portable tests excluded on non-Windows).

#ifdef _WIN32

#include "dx12_dense.hpp"
#include "dx12_engine.hpp"
#include "tensor.hpp"
#include "model.hpp"

#include <d3d12.h>
#include <d3dcompiler.h>
#include <wrl/client.h>
#include <cmath>
#include <cstring>
#include <stdexcept>

using Microsoft::WRL::ComPtr;

namespace minagi {
using namespace dx12_dense;
}

namespace dx12_dense {

// ==================== Push constants ====================

static inline DensePushConstants make_pc(DispatchId id)
{
    DensePushConstants pc{};
    pc.dispatch_id = static_cast<uint32_t>(id);
    return pc;
}

// ==================== Buffer management ====================

void DenseComputeContext::bind_buffer(int slot, const GpuBuffer& buf)
{
    if (slot >= 0 && slot < 12)
        buffers_[slot] = buf;
}

void DenseComputeContext::set_push_constants(const DensePushConstants& pc)
{
    push_constants_ = pc;
}

// Helper: push 32-bit constants to root param 1 via the compute command list.
static void set_root_constants(ID3D12GraphicsCommandList* cmd,
                               const DensePushConstants& pc)
{
    cmd->SetComputeRoot32BitConstants(1, sizeof(DensePushConstants) / sizeof(uint32_t),
                                      &pc, 0);
}

// ==================== MatMul (FP32) ====================
// C = A * B^T; A [M,K] in binding 1, B [N,K] in binding 2, out [M,N] to binding 0
void DenseComputeContext::dispatch_matmul(uint32_t M, uint32_t N, uint32_t K,
                                           ID3D12GraphicsCommandList* cmd)
{
    auto pc = make_pc(DispatchId::MatMulFP32);
    pc.M = M; pc.N = N; pc.K = K;
    set_push_constants(pc);

    // A is SRV from binding 1, B is SRV from binding 2, output is UAV in binding 0
    cmd->SetComputeRootShaderResourceView(1, buffers_[1].gpu_va);
    cmd->SetComputeRootShaderResourceView(2, buffers_[2].gpu_va);
cmd->SetComputeRootUnorderedAccessView(0, buffers_[0].gpu_va);
    set_root_constants(cmd, pc);
    cmd->Dispatch((M + 7) / 8, (N + 7) / 8, 1);
}

// ==================== RMSNorm ====================
// x from binding 0, weight from binding 3, output to binding 0
void DenseComputeContext::dispatch_rmsnorm(uint32_t rows, uint32_t cols, float eps,
                                             ID3D12GraphicsCommandList* cmd)
{
    auto pc = make_pc(DispatchId::RMSNorm);
    pc.M = rows; pc.N = cols; pc.eps = eps;
    set_push_constants(pc);

    // x in/out via UAV (binding 0), weight via SRV (binding 3)
    cmd->SetComputeRootUnorderedAccessView(0, buffers_[0].gpu_va);
    cmd->SetComputeRootShaderResourceView(3, buffers_[3].gpu_va);
    set_root_constants(cmd, pc);
    cmd->Dispatch((rows + 255) / 256, 1, 1);
}

// ==================== Softmax (row-wise) ====================
void DenseComputeContext::dispatch_softmax(uint32_t rows, uint32_t cols,
                                             ID3D12GraphicsCommandList* cmd)
{
    auto pc = make_pc(DispatchId::Softmax);
    pc.M = rows; pc.N = cols;
    set_push_constants(pc);

    cmd->SetComputeRootUnorderedAccessView(0, buffers_[0].gpu_va);
    cmd->SetComputeRootUnorderedAccessView(9, buffers_[9].gpu_va);// scratch
    set_root_constants(cmd, pc);
    cmd->Dispatch(rows, (cols + 15) / 16, 1);
}

// ==================== RoPE ====================
void DenseComputeContext::dispatch_rope(uint32_t B, uint32_t H, uint32_t T,
                                          uint32_t hd, uint32_t pos_offset,
                                          ID3D12GraphicsCommandList* cmd)
{
    auto pc = make_pc(DispatchId::RoPE);
    pc.M = B * H * T;
    pc.T = T; pc.H = H; pc.hd = hd; pc.pos_offset = pos_offset;
    set_push_constants(pc);

    cmd->SetComputeRootUnorderedAccessView(0, buffers_[0].gpu_va);
    cmd->SetComputeRootShaderResourceView(6, buffers_[6].gpu_va);  // cos
    cmd->SetComputeRootShaderResourceView(7, buffers_[7].gpu_va);  // sin
    set_root_constants(cmd, pc);
    cmd->Dispatch((B * H * T + 255) / 256, 1, 1);
}

// ==================== Embedding Gather ====================
void DenseComputeContext::dispatch_embedding_gather(uint32_t V, uint32_t d, uint32_t M,
                                                      ID3D12GraphicsCommandList* cmd)
{
    auto pc = make_pc(DispatchId::EmbeddingGather);
    pc.M = M; pc.N = d; pc.K = V;
    set_push_constants(pc);

    // idx from binding 0 (UAV read as SRV), emb from binding 1, output to binding 0
    cmd->SetComputeRootShaderResourceView(0, buffers_[0].gpu_va);
    cmd->SetComputeRootShaderResourceView(1, buffers_[1].gpu_va);
    cmd->SetComputeRootUnorderedAccessView(0, buffers_[0].gpu_va);
    set_root_constants(cmd, pc);
    cmd->Dispatch((M + 255) / 256, 1, 1);
}

// ==================== SiLU ====================
void DenseComputeContext::dispatch_silu(uint32_t n,
                                          ID3D12GraphicsCommandList* cmd)
{
    auto pc = make_pc(DispatchId::SiLU);
    pc.M = n;
    set_push_constants(pc);

    cmd->SetComputeRootUnorderedAccessView(0, buffers_[0].gpu_va);
    set_root_constants(cmd, pc);
    cmd->Dispatch((n + 255) / 256, 1, 1);
}

// ==================== GELU ====================
void DenseComputeContext::dispatch_gelu(uint32_t n,
                                          ID3D12GraphicsCommandList* cmd)
{
    auto pc = make_pc(DispatchId::GELU);
    pc.M = n;
    set_push_constants(pc);

    cmd->SetComputeRootUnorderedAccessView(0, buffers_[0].gpu_va);
    set_root_constants(cmd, pc);
    cmd->Dispatch((n + 255) / 256, 1, 1);
}

// ==================== SiLU * Gate (SwiGLU) ====================
void DenseComputeContext::dispatch_silu_gate(uint32_t n,
                                               ID3D12GraphicsCommandList* cmd)
{
    auto pc = make_pc(DispatchId::SiLUGate);
    pc.M = n;
    set_push_constants(pc);

    // x from binding 0 (UAV), gate from binding 3 (SRV)
    cmd->SetComputeRootUnorderedAccessView(0, buffers_[0].gpu_va);
    cmd->SetComputeRootShaderResourceView(3, buffers_[3].gpu_va);
    set_root_constants(cmd, pc);
    cmd->Dispatch((n + 255) / 256, 1, 1);
}

// ==================== Add (residual) ====================
void DenseComputeContext::dispatch_add(uint32_t n,
                                         ID3D12GraphicsCommandList* cmd)
{
    auto pc = make_pc(DispatchId::Add);
    pc.M = n;
    set_push_constants(pc);

    // a from binding 0 (UAV), b from binding 3 (SRV)
    cmd->SetComputeRootUnorderedAccessView(0, buffers_[0].gpu_va);
    cmd->SetComputeRootShaderResourceView(3, buffers_[3].gpu_va);
    set_root_constants(cmd, pc);
    cmd->Dispatch((n + 255) / 256, 1, 1);
}

// ==================== Attention Scores ====================
void DenseComputeContext::dispatch_attn_scores(uint32_t M, uint32_t kv_len,
                                                  uint32_t T, uint32_t hd,
                                                  float inv_sqrt_hd,
                                                  ID3D12GraphicsCommandList* cmd)
{
    auto pc = make_pc(DispatchId::AttnScores);
    pc.M = M; pc.kv_len = kv_len; pc.T = T; pc.hd = hd; pc.inv_sqrt_hd = inv_sqrt_hd;
    set_push_constants(pc);

    // Q from binding 0 [B*H*T, hd], K from binding 1 [B*H*kv_len, hd], scores to binding 9
    cmd->SetComputeRootShaderResourceView(0, buffers_[0].gpu_va);  // Q as SRV
    cmd->SetComputeRootShaderResourceView(1, buffers_[1].gpu_va);  // K as SRV
    cmd->SetComputeRootUnorderedAccessView(9, buffers_[9].gpu_va);// scores output
    set_root_constants(cmd, pc);
    cmd->Dispatch((M + 7) / 8, (kv_len + 7) / 8, 1);
}

// ==================== Attention Weighted Sum ====================
void DenseComputeContext::dispatch_attn_wsum(uint32_t M, uint32_t kv_len,
                                               uint32_t hd,
                                               ID3D12GraphicsCommandList* cmd)
{
    auto pc = make_pc(DispatchId::AttnWeightedSum);
    pc.M = M; pc.kv_len = kv_len; pc.hd = hd;
    set_push_constants(pc);

    // scores from binding 9, V from binding 1, output to binding 0
    cmd->SetComputeRootShaderResourceView(9, buffers_[9].gpu_va);  // scores
    cmd->SetComputeRootShaderResourceView(1, buffers_[1].gpu_va);  // V
    cmd->SetComputeRootUnorderedAccessView(0, buffers_[0].gpu_va);// output
    set_root_constants(cmd, pc);
    cmd->Dispatch((M + 7) / 8, (hd + 31) / 32, 1);
}

// ==================== Elementwise Copy ====================
void DenseComputeContext::dispatch_copy(uint32_t n,
                                          ID3D12GraphicsCommandList* cmd)
{
    auto pc = make_pc(DispatchId::ElementwiseCopy);
    pc.M = n;
    set_push_constants(pc);

    cmd->SetComputeRootShaderResourceView(0, buffers_[0].gpu_va);
    cmd->SetComputeRootUnorderedAccessView(0, buffers_[0].gpu_va);
    set_root_constants(cmd, pc);
    cmd->Dispatch((n + 255) / 256, 1, 1);
}

// ==================== Composite: Attention Residual ====================
// Mirrors recur.cpp::attn_residual (lines 79-251)
void DenseComputeContext::dispatch_attn_residual(
    mt::Tensor& x,
    const mt::Tensor& qkv_w,
    const mt::Tensor& proj_w,
    const mt::Tensor& ln1_w,
    const mt::Tensor& cos,
    const mt::Tensor& sin,
    model::KVCache& cache,
    uint32_t T, uint32_t H, uint32_t hd, uint32_t d,
    uint32_t pos_offset, ID3D12GraphicsCommandList* cmd)
{
    const uint32_t B = 1;
    const uint32_t M = B * T;
    const float inv_sqrt = 1.0f / std::sqrt(static_cast<float>(hd));

    // Step 1: rms_norm(x, ln1_w, eps) -> binding 0
    bind_buffer(3, buffers_[3]);  // ln1_w already bound by caller
    dispatch_rmsnorm(M, d, 1e-6f, cmd);

    // Step 2: matmul(xn, qkv_w^T) -> [M, 3d] to binding 0
    {
        auto pc = make_pc(DispatchId::MatMulFP32);
        pc.M = M; pc.N = 3 * d; pc.K = d;
        set_push_constants(pc);
        cmd->SetComputeRootShaderResourceView(1, buffers_[1].gpu_va);  // xn (output of rms_norm)
        cmd->SetComputeRootShaderResourceView(2, buffers_[2].gpu_va);  // qkv_w as B[N,K]
        cmd->SetComputeRootUnorderedAccessView(0, buffers_[0].gpu_va);// output
        set_root_constants(cmd, pc);
        cmd->Dispatch((M + 7) / 8, ((3 * d) + 7) / 8, 1);
    }

    // Step 3-6: RoPE applied in-place to QKV via stride3 indexing.
    // The GPU shader applies RoPE per-head directly from the QKV buffer.
    {
        auto pc = make_pc(DispatchId::RoPE);
        pc.M = B * H * T;
        pc.T = T; pc.H = H; pc.hd = hd; pc.pos_offset = pos_offset;
        pc.stride3 = 3 * d;
        set_push_constants(pc);
        cmd->SetComputeRootUnorderedAccessView(0, buffers_[0].gpu_va);// QKV
        cmd->SetComputeRootShaderResourceView(6, buffers_[6].gpu_va);   // cos
        cmd->SetComputeRootShaderResourceView(7, buffers_[7].gpu_va);   // sin
        set_root_constants(cmd, pc);
        cmd->Dispatch((B * H * T + 255) / 256, 1, 1);
    }

    // Step 7: attention scores (Q @ K^T) with causal mask
    // After RoPE, Q is at [B*H*T, hd] and K is at [B*H*T, hd] within the QKV buffer.
    // K from binding 1 (qkv_w repurposed as KV pointer), scores to binding 9.
    dispatch_attn_scores(M * H, T, T, hd, inv_sqrt, cmd);

    // Step 8: softmax on scores (binding 9)
    dispatch_softmax(M * H, T, cmd);

    // Step 9: weighted sum: scores @ V -> binding 0
    dispatch_attn_wsum(M * H, T, hd, cmd);

    // Step 10: proj matmul(out_flat, proj_w^T) -> [M, d]
    {
        auto pc = make_pc(DispatchId::MatMulFP32);
        pc.M = M; pc.N = d; pc.K = d;
        set_push_constants(pc);
        cmd->SetComputeRootShaderResourceView(1, buffers_[1].gpu_va);  // attn output
        cmd->SetComputeRootShaderResourceView(2, buffers_[2].gpu_va);  // proj_w
        cmd->SetComputeRootUnorderedAccessView(0, buffers_[0].gpu_va);// output
        set_root_constants(cmd, pc);
        cmd->Dispatch((M + 7) / 8, (d + 7) / 8, 1);
    }

    // Step 11: residual add (proj output + original x in binding 3)
    bind_buffer(3, buffers_[3]);  // x (original)
    dispatch_add(M * d, cmd);
}

// ==================== Composite: MLP Block ====================
// Mirrors recur.cpp dense_mlp_impl and model.cpp::mlp
void DenseComputeContext::dispatch_mlp(
    mt::Tensor& x,
    const mt::Tensor& w1, const mt::Tensor& w3, const mt::Tensor& w2,
    const mt::Tensor& ln2_w,
    uint32_t T, uint32_t d, uint32_t dff,
    ID3D12GraphicsCommandList* cmd)
{
    const uint32_t M = T;

    // 1: rms_norm(x, ln2_w) -> binding 0
    bind_buffer(3, buffers_[3]);
    dispatch_rmsnorm(M, d, 1e-6f, cmd);

    // 2: matmul(xn, w1^T) -> h1 [M, dff] (to scratch binding 9)
    {
        auto pc = make_pc(DispatchId::MatMulFP32);
        pc.M = M; pc.N = dff; pc.K = d;
        set_push_constants(pc);
        cmd->SetComputeRootShaderResourceView(1, buffers_[0].gpu_va);  // xn
        cmd->SetComputeRootShaderResourceView(2, buffers_[10].gpu_va); // w1
        cmd->SetComputeRootUnorderedAccessView(9, buffers_[9].gpu_va); // h1
        set_root_constants(cmd, pc);
        cmd->Dispatch((M + 7) / 8, (dff + 7) / 8, 1);
    }

    // 3: matmul(xn, w3^T) -> h3 [M, dff] (to binding 0, overwriting xn)
    {
        auto pc = make_pc(DispatchId::MatMulFP32);
        pc.M = M; pc.N = dff; pc.K = d;
        set_push_constants(pc);
        cmd->SetComputeRootShaderResourceView(1, buffers_[0].gpu_va);  // xn (reuse)
        cmd->SetComputeRootShaderResourceView(2, buffers_[11].gpu_va); // w3
        cmd->SetComputeRootUnorderedAccessView(0, buffers_[0].gpu_va);// h3 overwrites xn
        set_root_constants(cmd, pc);
        cmd->Dispatch((M + 7) / 8, (dff + 7) / 8, 1);
    }

    // 4: silu(h1) * h3 = gated -> to binding 0
    // h1 in binding 9, h3 in binding 0
    {
        auto pc = make_pc(DispatchId::SiLUGate);
        pc.M = M * dff;
        set_push_constants(pc);
        cmd->SetComputeRootShaderResourceView(3, buffers_[9].gpu_va);  // h1 as gate
        cmd->SetComputeRootUnorderedAccessView(0, buffers_[0].gpu_va);// h3 in, result out
        set_root_constants(cmd, pc);
        cmd->Dispatch((M * dff + 255) / 256, 1, 1);
    }

    // 5: matmul(g, w2^T) -> y [M, d]
    {
        auto pc = make_pc(DispatchId::MatMulFP32);
        pc.M = M; pc.N = d; pc.K = dff;
        set_push_constants(pc);
        cmd->SetComputeRootShaderResourceView(1, buffers_[0].gpu_va);  // g
        cmd->SetComputeRootShaderResourceView(2, buffers_[11].gpu_va); // w2
        cmd->SetComputeRootUnorderedAccessView(9, buffers_[9].gpu_va);// y to scratch
        set_root_constants(cmd, pc);
        cmd->Dispatch((M + 7) / 8, (d + 7) / 8, 1);
    }

    // 6: residual: binding 9 (y) + binding 3 (original x) -> binding 0
    {
        auto pc = make_pc(DispatchId::Add);
        pc.M = M * d;
        set_push_constants(pc);
        cmd->SetComputeRootShaderResourceView(0, buffers_[9].gpu_va);  // y
        cmd->SetComputeRootUnorderedAccessView(0, buffers_[0].gpu_va);// result
        cmd->SetComputeRootShaderResourceView(3, buffers_[3].gpu_va);  // x
        set_root_constants(cmd, pc);
        cmd->Dispatch((M * d + 255) / 256, 1, 1);
    }
}

// ==================== Shader compilation ====================
// (Delegates to dx12_engine.cpp compile_hlsl which has the full implementation.)

} // namespace dx12_dense

#endif // _WIN32
