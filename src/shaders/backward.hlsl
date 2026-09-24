// src/shaders/backward.hlsl
// DirectX 12 HLSL compute shaders for mini-AGI backward pass gradients.
//
// Each kernel mirrors the corresponding forward op from dense.hlsl / recur.cpp.
// Standard 12-binding DSL:
//   binding 0 = activation (input/output, scratch)
//   binding 1 = weight matrix A (read)
//   binding 2 = weight matrix B (read)
//   binding 3 = weight (1D, elementwise read)
//   binding 4 = KV cache K (read)
//   binding 5 = KV cache V (read)
//   binding 6 = cos (RoPE)
//   binding 7 = sin (RoPE)
//   binding 8 = output
//   binding 9 = scratch (read/write, scores, intermediate buffers)
//   binding 10 = mlp weights (read)
//   binding 11 = head_w / additional weight (read)

#ifndef _BACKWARD_HLSL_
#define _BACKWARD_HLSL_

// Push constants (32-bit root constants, root param 1)
cbuffer BwdPushConstants : register(b0)
{
    uint  dispatch_id;    // which backward kernel to run
    uint  M;              // rows of activation
    uint  N;              // output cols / vocab
    uint  K;              // inner dim
    uint  T;              // sequence length
    uint  H;              // num heads
    uint  hd;             // head dim
    uint  V;              // vocab size (for cross-entropy)
    uint  target_id;      // target token id (for cross-entropy)
    uint  kv_len;         // KV cache length
    uint  cached_T;       // cached tokens in KV
    uint  head_dim;       // = hd
    uint  d_model;        // model dim
    uint  d_ff;           // MLP intermediate dim
    uint  stride3;        // 3 * d_model
    float eps;            // epsilon for RMSNorm
    float inv_sqrt_hd;    // 1/sqrt(hd) for attention scaling
    uint  pad0;
    uint  pad1;
};

// Storage buffers (all ByteAddressBuffer for read, RWByteAddressBuffer for write)
// binding 0: activation / grad buffers
ByteAddressBuffer g_input : register(t0);
RWByteAddressBuffer g_grad_out : register(u0);

// binding 1: weight A (read)
ByteAddressBuffer g_w_A : register(t1);
RWByteAddressBuffer g_grad_A : register(u1);

// binding 2: weight B (read)
ByteAddressBuffer g_w_B : register(t2);
RWByteAddressBuffer g_grad_B : register(u2);

// binding 3: 1D weight (read)
ByteAddressBuffer g_weight1d : register(t3);
RWByteAddressBuffer g_grad_1d : register(u3);

// binding 4: KV cache K (read)
ByteAddressBuffer g_kv_k : register(t4);

// binding 5: KV cache V (read)
ByteAddressBuffer g_kv_v : register(t5);

// binding 6: cos (RoPE)
ByteAddressBuffer g_cos : register(t6);

// binding 7: sin (RoPE)
ByteAddressBuffer g_sin : register(t7);

// binding 8: output logits
RWByteAddressBuffer g_output : register(u8);

// binding 9: scratch (read/write)
ByteAddressBuffer g_scratch_r : register(t9);
RWByteAddressBuffer g_scratch : register(u9);

// binding 10: mlp weights (read)
ByteAddressBuffer g_mlp_w : register(t10);

// binding 11: head_w / additional weight (read)
ByteAddressBuffer g_head_w : register(t11);
RWByteAddressBuffer g_grad_head : register(u11);

// ==================== Utility functions ====================

float load_f32(ByteAddressBuffer buf, uint byte_offset)
{
    return asfloat(buf.Load(byte_offset));
}

float load_f32(RWByteAddressBuffer buf, uint byte_offset)
{
    return asfloat(buf.Load(byte_offset));
}

void store_f32(RWByteAddressBuffer buf, uint byte_offset, float val)
{
    buf.Store(byte_offset, asuint(val));
}

void atomic_add_f32(RWByteAddressBuffer buf, uint byte_offset, float val)
{
    uint expected = buf.Load(byte_offset);
    uint desired;
    float current;
    do {
        current = asfloat(expected);
        float new_val = current + val;
        desired = asuint(new_val);
        uint original;
        buf.InterlockedCompareExchange(byte_offset, expected, desired, original);
        expected = (original == expected) ? desired : original;
    } while (expected != desired);
}

// ==================== Kernel 0: grad_cross_entropy ====================
// Given logits [1,1,V] (i.e., [V] flat) and target_id, compute grad = (softmax(logits) - onehot(target))
// Input: logits from binding 0 [V], target_id from push constant
// Output: grad [V] to binding 0 (in-place overwrite) or binding 8
[	numthreads(256, 1, 1)]
void CSGradCrossEntropy(uint3 DTid : SV_DispatchThreadID)
{
    uint idx = DTid.x;
    if (idx >= V)
        return;

    // Load logits from binding 0 [V]
    float logit = load_f32(g_input, idx * 4);

    // Compute softmax (numerically stable)
    // Phase 1: find max
    float max_val = -1e30f;
    if (idx < V)
    {
        max_val = logit;
        // Thread 0 does the max reduction
        if (idx == 0)
        {
            for (uint i = 0; i < V; ++i)
            {
                float v = load_f32(g_input, i * 4);
                if (v > max_val) max_val = v;
            }
        }
    }
    GroupMemoryBarrierWithGroupSync();

    // Broadcast max_val from thread 0
    if (idx == 0)
    {
        store_f32(g_scratch, 0, max_val);
    }
    GroupMemoryBarrierWithGroupSync();
    float broadcast_max = load_f32(g_scratch, 0);

    // Phase 2: compute exp and sum
    float exp_val = exp(logit - broadcast_max);
    if (idx == 0)
    {
        float sum = 0.0f;
        for (uint i = 0; i < V; ++i)
        {
            float v = load_f32(g_input, i * 4);
            sum += exp(v - broadcast_max);
        }
        store_f32(g_scratch, 4, sum);
    }
    GroupMemoryBarrierWithGroupSync();
    float sum = load_f32(g_scratch, 4);

    // Phase 3: softmax - onehot
    float sm = exp_val / (sum + 1e-30f);
    float grad = sm - (idx == target_id ? 1.0f : 0.0f);

    // Output to binding 0
    store_f32(g_grad_out, idx * 4, grad);
}

// ==================== Kernel 1: grad_matmul_w ====================
// Weight gradient: grad_weight [N,K] = G^T @ X
// G (grad_output) from binding 0 [M,N], X (input) from binding 1 [M,K]
// Output: grad_weight to binding 2 [N,K] (accumulate)
[	numthreads(8, 8, 1)]
void CSGradMatMulW(uint3 DTid : SV_DispatchThreadID)
{
    uint row = DTid.x;  // N row of weight gradient
    uint col = DTid.y;  // K col of weight gradient
    if (row >= N || col >= K)
        return;

    float sum = 0.0f;
    for (uint m = 0; m < M; ++m)
    {
        // G^T[row, m] = G[m, row]
        float g = load_f32(g_input, (m * N + row) * 4);
        // X[m, col]
        float x = load_f32(g_w_A, (m * K + col) * 4);
        sum += g * x;
    }
    // Accumulate into grad_weight (binding 2 output)
    store_f32(g_grad_out, (row * K + col) * 4, sum);
}

// ==================== Kernel 2: grad_matmul_x ====================
// Input gradient: grad_input [M,K] = G @ W
// G (grad_output) from binding 0 [M,N], W from binding 1 [N,K]
// Output: grad_input to binding 2 [M,K]
[	numthreads(8, 8, 1)]
void CSGradMatMulX(uint3 DTid : SV_DispatchThreadID)
{
    uint row = DTid.x;  // M row
    uint col = DTid.y;  // K col
    if (row >= M || col >= K)
        return;

    float sum = 0.0f;
    for (uint n = 0; n < N; ++n)
    {
        float g = load_f32(g_input, (row * N + n) * 4);
        float w = load_f32(g_w_A, (n * K + col) * 4);
        sum += g * w;
    }
    store_f32(g_grad_out, (row * K + col) * 4, sum);
}

// ==================== Kernel 3: grad_rmsnorm ====================
// Given output grad dY [M, N] and forward weight [N], compute input grad dX [M, N].
// Forward: y = x * rsqrt(mean(x^2) + eps) * weight
// Backward: dX = weight * (rsqrt(mean_sq + eps) * dY - (3/(N^2)) * rsqrt * sum(x * dY) * x)
// Simplified form using the RMSNorm Jacobian.
//
// Input: dY from binding 0 [M, N], weight from binding 3 [N], x from binding 1 [M, N]
// Output: dX to binding 0 (in-place)
[	numthreads(64, 1, 1)]
void CSGradRMSNorm(uint3 DTid : SV_DispatchThreadID)
{
    uint row = DTid.x;
    uint col = DTid.y;
    if (row >= M || col >= N)
        return;

    // Compute sum_sq = sum(x[row, :]^2)
    float sum_sq = 0.0f;
    for (uint c = 0; c < N; ++c)
    {
        float x_val = load_f32(g_w_A, (row * N + c) * 4);
        sum_sq += x_val * x_val;
    }
    float mean_sq = sum_sq / N;
    float norm_b = rsqrt(mean_sq + eps);  // = 1/sqrt(mean_sq + eps)

    // Compute dot_x_dy = sum(x[row, c] * dY[row, c]) over c
    float dot_x_dy = 0.0f;
    for (uint c = 0; c < N; ++c)
    {
        float x_val = load_f32(g_w_A, (row * N + c) * 4);
        float dy = load_f32(g_input, (row * N + c) * 4);
        dot_x_dy += x_val * dy;
    }

    float w = load_f32(g_weight1d, col * 4);
    float x_val = load_f32(g_w_A, (row * N + col) * 4);
    float dy = load_f32(g_input, (row * N + col) * 4);

    // RMSNorm gradient formula:
    // dX = w * norm_b * (dY - 2/(N) * (dot_x_dy / N) * x)
    float scale = w * norm_b;
    float correction = (2.0f / N) * (dot_x_dy / N) * x_val;
    float dx = scale * (dy - correction);

    store_f32(g_grad_out, (row * N + col) * 4, dx);
}

// ==================== Kernel 4: grad_rope ====================
// Reverse the RoPE rotation. Given rotated q/k grad, compute unrotated grad.
// RoPE forward: x1' = x1*c - x2*s, x2' = x1*s + x2*c
// Inverse: x1 = x1'*c + x2'*s, x2 = -x1'*s + x2'*c
//
// Input: rotated grad from binding 0 [B*H*T, hd], cos from binding 6, sin from binding 7
// Output: unrotated grad to binding 0 (in-place)
[	numthreads(256, 1, 1)]
void CSGradRoPE(uint3 DTid : SV_DispatchThreadID)
{
    uint idx = DTid.x;
    uint half = hd / 2;
    if (idx >= M)
        return;

    // Layout: element (b,h,t,d) at flat (b*H*T + h*T + t)*hd + d
    // Thread idx maps to a full row of hd elements
    uint bh = idx / T;  // since B=1, bh = h index group
    uint tt = idx % T;  // t within block

    // cos/sin indexed by t (position within current block)
    for (uint i = 0; i < half; ++i)
    {
        float c = load_f32(g_cos, (tt * half + i) * 4);
        float s = load_f32(g_sin, (tt * half + i) * 4);

        // Load rotated gradient
        float g1 = load_f32(g_input, (idx * hd + 2 * i) * 4);
        float g2 = load_f32(g_input, (idx * hd + 2 * i + 1) * 4);

        // Inverse rotation: x1 = g1*c + g2*s, x2 = -g1*s + g2*c
        float x1 = g1 * c + g2 * s;
        float x2 = -g1 * s + g2 * c;

        store_f32(g_grad_out, (idx * hd + 2 * i) * 4, x1);
        store_f32(g_grad_out, (idx * hd + 2 * i + 1) * 4, x2);
    }
}

// ==================== Kernel 5: grad_attention ====================
// Given attention output grad dO [B*H*T, hd], Q [B*H*T, hd], K [B*H*kv_len, hd],
// V [B*H*kv_len, hd], and softmax attention matrix sm [B*H*T, kv_len] (from scratch),
// compute Q/K/V grads via the softmax Jacobian.
//
// dQ = (dO @ V^T) @ diag(sm) ... actually:
//   dQ[b,h,t,i] = sum_j sm[b,h,t,j] * dO[b,h,t,j] ... (no, that's dK)
// Standard attention backward:
//   dS = dO @ V^T  (dO [T,hd], V [kv_len,hd] -> dS [T, kv_len])
//   dV = S^T @ dO  (S [T,kv_len], dO [T,hd] -> dV [kv_len,hd])
//   dK = Q^T @ dS  (Q [T,hd]^T, dS [T,kv_len] -> dK [hd,kv_len] -> reshape [kv_len,hd])
//   dQ = dS @ V    (dS [T,kv_len], V [kv_len,hd] -> dQ [T,hd])
//
// This shader computes dV for token t into binding 9 scratch.
// Q from binding 0 [B*H*T, hd], K from binding 1 [B*H*kv_len, hd]
// dO from binding 5 [B*H*T, hd], sm from binding 4 [B*H*T, kv_len]
// V from binding 2 [B*H*kv_len, hd]
// Output: dV [kv_len, hd] to binding 9
[	numthreads(8, 32, 1)]
void CSGradAttention(uint3 DTid : SV_DispatchThreadID)
{
    uint row = DTid.x;    // kv_len row
    uint col = DTid.y;    // hd dim
    if (row >= kv_len || col >= hd)
        return;

    // We compute dV = S^T @ dO for each (kv_len, hd) position.
    // This requires iterating over all T query positions.
    // S is from binding 4 [B*H*T, kv_len], dO from binding 5 [B*H*T, hd]
    // We assume B=1, H heads, so B*H groups. For simplicity, compute for group 0.
    // In practice, the caller would set H=1 or handle per-head.

    float acc = 0.0f;
    for (uint t = 0; t < T; ++t)
    {
        // S[t, row] = sm[(0 * H * T + h * T + t) * kv_len + row] -- for head h=0
        // We assume H=1 for this simplified kernel; for H>1 the caller iterates per-head
        float s = load_f32(g_scratch_r, (t * kv_len + row) * 4);
        float dO_val = load_f32(g_w_B, (t * hd + col) * 4);  // dO from binding 2 (using different binding)
        acc += s * dO_val;
    }
    store_f32(g_scratch, (row * hd + col) * 4, acc);
}

// ==================== Kernel 6: grad_silu_mul ====================
// SiLU gate derivative for SwiGLU.
// Forward: y = silu(x) * gate, where silu(x) = x * sigmoid(x)
// dy/dx = silu'(x) * gate + silu(x) * dgate (but dgate is the upstream grad for gate input separately)
// Simplified: if we have dy (upstream grad) and we split into x and gate:
//   dx = dy * gate * sigmoid(x) * (1 + x * (1 - sigmoid(x)))
//   dgate = dy * silu(x)
//
// Input: dy from binding 0 [M], x from binding 1 [M], gate from binding 3 [M]
// Output: dx to binding 9 [M], dgate to binding 2 [M]
[	numthreads(256, 1, 1)]
void CSGradSiLUMul(uint3 DTid : SV_DispatchThreadID)
{
    uint idx = DTid.x;
    if (idx >= M)
        return;

    float x = load_f32(g_w_A, idx * 4);     // x from binding 1
    float gate = load_f32(g_weight1d, idx * 4);  // gate from binding 3
    float dy = load_f32(g_input, idx * 4);   // upstream grad from binding 0

    // Compute sigmoid(x) stably
    float sig;
    if (x == 0.0f)
    {
        sig = 0.0f;
    }
    else if (x > 0.0f)
    {
        sig = 1.0f / (1.0f + exp(-x));
    }
    else
    {
        float e = exp(x);
        sig = e / (1.0f + e);
    }

    float silu_x = x * sig;
    // silu'(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
    float dsilu = sig * (1.0f + x * (1.0f - sig));

    // dx = dy * gate * dsilu
    float dx = dy * gate * dsilu;
    // dgate = dy * silu(x)
    float dgate = dy * silu_x;

    store_f32(g_scratch, idx * 4, dx);       // dx to binding 9
    store_f32(g_grad_out, idx * 4, dgate);   // dgate to binding 0
}

// ==================== Kernel 7: grad_scatter_add ====================
// Scatter-add gradients back through top-k dispatch.
// Given grad_output [N, D] and assignment mapping (token_idx, slot, weight),
// scatter the weighted gradient back to each expert's input slot.
//
// Assignment from binding 5 [total_kept * 4]: {slot:u32, token_idx:u32, weight:f32, _pad:u32}
// grad_output from binding 0 [N, D]
// Output: grad_input [total_kept, D] to binding 2 [total_kept, D]
[	numthreads(64, 1, 1)]
void CSGradScatterAdd(uint3 DTid : SV_DispatchThreadID)
{
    uint idx = DTid.x;
    if (idx >= total_kept * D)
        return;

    uint token_row = idx / D;
    uint d = idx % D;

    // Load assignment: {slot, token_idx, weight, _pad} = 16 bytes per assignment
    uint assign_base = token_row * 16;
    uint slot = g_scratch_r.Load(assign_base);           // not needed for grad
    uint token_idx = g_scratch_r.Load(assign_base + 4);
    float weight = asfloat(g_scratch_r.Load(assign_base + 8));

    // Load dO from output [N, D]
    float dO_val = load_f32(g_input, (token_idx * D + d) * 4);

    // Grad flows back: d(xs) = weight * dO
    // But we need to scatter the *weight* too — the weight is already included
    // In backward, the assignment weight gradient needs to consider dO as well
    // For simplicity, grad_input = weight * dO (elementwise scatter)
    float grad = weight * dO_val;

    store_f32(g_grad_out, (token_row * D + d) * 4, grad);
}

// ==================== Entry points ====================
[numthreads(256, 1, 1)]
void main_grad_cross_entropy(uint3 DTid : SV_DispatchThreadID)
{
    CSGradCrossEntropy(DTid);
}

[numthreads(8, 8, 1)]
void main_grad_matmul_w(uint3 DTid : SV_DispatchThreadID)
{
    CSGradMatMulW(DTid);
}

[numthreads(8, 8, 1)]
void main_grad_matmul_x(uint3 DTid : SV_DispatchThreadID)
{
    CSGradMatMulX(DTid);
}

[numthreads(64, 1, 1)]
void main_grad_rmsnorm(uint3 DTid : SV_DispatchThreadID)
{
    CSGradRMSNorm(DTid);
}

[numthreads(256, 1, 1)]
void main_grad_rope(uint3 DTid : SV_DispatchThreadID)
{
    CSGradRoPE(DTid);
}

[numthreads(8, 32, 1)]
void main_grad_attention(uint3 DTid : SV_DispatchThreadID)
{
    CSGradAttention(DTid);
}

[numthreads(256, 1, 1)]
void main_grad_silu_mul(uint3 DTid : SV_DispatchThreadID)
{
    CSGradSiLUMul(DTid);
}

[numthreads(64, 1, 1)]
void main_grad_scatter_add(uint3 DTid : SV_DispatchThreadID)
{
    CSGradScatterAdd(DTid);
}

#endif // _BACKWARD_HLSL_
