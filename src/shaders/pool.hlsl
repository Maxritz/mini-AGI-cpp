// src/shaders/pool.hlsl
// DirectX 12 HLSL compute shader for PooledMLP expert matmul.
// Port of pool.cpp:pool_mlp_forward expert-eval loop (section 4, lines 285-326).
//
// The CPU routing (top-k, softmax, capacity-bounded slot assignment) stays on
// CPU and produces a flattened dispatch table. The GPU handles only the heavy
// per-expert matmul: silu(xs@W1^T) * (xs@W3^T) @ W2^T, then scatter-add weighted.
//
// Standard 12-binding DSL (same as dense.hlsl):
//   binding 0  = token_input   [total_kept * D]        (xs rows, flat per assignment)
//   binding 1  = w1_weights   [resident * d_ff * D]    (gate, [slot, f, d] row-major)
//   binding 2  = w3_weights   [resident * d_ff * D]    (value, [slot, f, d] row-major)
//   binding 3  = w2_weights   [resident * D * d_ff]     (down-proj, [slot, d, f] row-major)
//   binding 4  = scratch_out   [total_kept * (2*d_ff + D)]  (a1[a+dff], a3[dff], y[D])
//   binding 5  = assignments   [total_kept * 4]         (slot:u32, token:u32, weight:f32, _pad:u32)
//   binding 6  = scatter_add   [N * D]                  (atomic accumulate mlp_out * w per token)
//   binding 7  = reserved
//   binding 8  = reserved
//   binding 9  = scratch
//   binding 10 = reserved
//   binding 11 = reserved

#ifndef _POOL_HLSL_
#define _POOL_HLSL_

cbuffer PoolPushConstants : register(b0)
{
    uint  dispatch_id;    // which shader to run (0=W1W3, 1=SiLUMul, 2=W2, 3=ScatterAdd)
    uint  D;              // model dimension (input/output)
    uint  d_ff;           // MLP intermediate dimension
    uint  N;              // number of token rows (sequence length)
    uint  total_kept;     // total kept assignments (after capacity drop)
    uint  resident;       // number of resident expert slots
    uint  mode;           // unused (dispatch_id drives the entry point)
    uint  pad0;
};

// Storage buffers — all bound as UAV (u0-u11) since compute shaders can read UAVs
RWByteAddressBuffer g_token_input  : register(u0);   // [total_kept, D]
RWByteAddressBuffer g_w1_weights   : register(u1);   // [resident * d_ff * D]
RWByteAddressBuffer g_w3_weights   : register(u2);   // [resident * d_ff * D]
RWByteAddressBuffer g_w2_weights   : register(u3);   // [resident * D * d_ff]
RWByteAddressBuffer g_scratch_out  : register(u4);   // [total_kept * (2*d_ff + D)]
RWByteAddressBuffer g_assignments  : register(u5);   // [total_kept * 4] packed {slot, token, weight}
RWByteAddressBuffer g_scatter_add  : register(u6);   // [N * D] atomic scatter
RWByteAddressBuffer g_scratch      : register(u9);   // general scratch

// ==================== Utility functions ====================

float load_f32(RWByteAddressBuffer buf, uint byte_offset)
{
    return asfloat(buf.Load(byte_offset));
}

void store_f32(RWByteAddressBuffer buf, uint byte_offset, float val)
{
    buf.Store(byte_offset, asuint(val));
}

// Atomic add on a float in a RWByteAddressBuffer
void atomic_add_f32(RWByteAddressBuffer buf, uint byte_offset, float val)
{
    // DX12 HLSL supports atomic operations on integer types only.
    // We use a compare exchange loop on the bit representation.
    // InterlockedCompareExchange takes 4 args: (offset, comparison, exchange, original)
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

// ==================== Kernel 0: Batched Expert MatMul (W1 and W3) ====================
// Computes: a1 = xs @ W1^T (gate), a3 = xs @ W3^T (value), for each kept token row.
//
// Thread idx = token_row * d_ff + f  (one thread per (token, feature) pair)
//
// Input layout:
//   g_token_input:  [total_kept, D] — token row data (flat index = assignment position)
//   g_w1_weights:   [resident * d_ff * D] — W1[s][f][d] at (s*d_ff + f)*D + d
//   g_w3_weights:   [resident * d_ff * D] — W3[s][f][d] at (s*d_ff + f)*D + d
//   g_assignments:  [total_kept * 4] — {slot:u32, token:u32, weight:f32, _pad:u32}
//
// Output:
//   g_scratch_out: [total_kept, 2*d_ff] — a1[first d_ff] then a3[next d_ff] per token

[numthreads(64, 1, 1)]
void CSPoolMatMulW1W3(uint3 DTid : SV_DispatchThreadID)
{
    uint idx = DTid.x;
    if (idx >= total_kept * d_ff)
        return;

    uint token_row = idx / d_ff;    // 0..total_kept-1
    uint f = idx % d_ff;            // 0..d_ff-1

    // Load slot from assignment
    uint assign_base = token_row * 4;  // 4 bytes per field, but stored as 4 uints = 16 bytes
    // Actually: each assignment is {slot(4B), token(4B), weight(4B float), pad(4B)} = 16 bytes
    uint slot = g_assignments.Load(assign_base);          // slot at byte offset token_row*16

    // Token input: xs[token_row, :] at offset token_row * D
    uint xs_base = token_row * D;

    // W1[s, f, d] at (slot * d_ff + f) * D + d
    uint w1_base = (slot * d_ff + f) * D;
    // W3[s, f, d] at (slot * d_ff + f) * D + d
    uint w3_base = (slot * d_ff + f) * D;

    float a1 = 0.0f;
    float a3 = 0.0f;
    for (uint dd = 0; dd < D; ++dd)
    {
        float xv = load_f32(g_token_input, (xs_base + dd) * 4);
        a1 += xv * load_f32(g_w1_weights, (w1_base + dd) * 4);
        a3 += xv * load_f32(g_w3_weights, (w3_base + dd) * 4);
    }

    // Store: a1 at [token_row * 2*d_ff + f], a3 at [token_row * 2*d_ff + d_ff + f]
    uint scratch_base = token_row * (d_ff * 2);
    store_f32(g_scratch_out, (scratch_base + f) * 4, a1);
    store_f32(g_scratch_out, (scratch_base + d_ff + f) * 4, a3);
}

// ==================== Kernel 1: SiLU(Gate) * Value ====================
// h[f] = silu(a1[f]) * a3[f], for each token row.
// Thread idx = token_row * d_ff + f
// Reads a1 and a3 from g_scratch_out, writes h back to a1 region.

[numthreads(64, 1, 1)]
void CSPoolSiLUMul(uint3 DTid : SV_DispatchThreadID)
{
    uint idx = DTid.x;
    if (idx >= total_kept * d_ff)
        return;

    uint token_row = idx / d_ff;
    uint f = idx % d_ff;

    uint scratch_base = token_row * (d_ff * 2);
    float a1 = load_f32(g_scratch_out, (scratch_base + f) * 4);
    float a3 = load_f32(g_scratch_out, (scratch_base + d_ff + f) * 4);

    // silu(x) = x * sigmoid(x), stable form matching pool.cpp::silu1
    float sig;
    if (a1 == 0.0f)
    {
        sig = 0.0f;
    }
    else if (a1 > 0.0f)
    {
        sig = 1.0f / (1.0f + exp(-a1));
    }
    else
    {
        float e = exp(a1);
        sig = e / (1.0f + e);
    }
    float h = a1 * sig * a3;

    // Write h back to a1 region (first d_ff of the 2*d_ff block)
    store_f32(g_scratch_out, (scratch_base + f) * 4, h);
}

// ==================== Kernel 2: Batched Expert Down-Proj MatMul (W2) ====================
// y[d] = sum_f(h[f] * W2[s][d][f])
//
// W2 layout from paged.cpp: [resident, D, d_ff] row-major
//   W2[s][dd][f] at (slot * D * d_ff) + dd * d_ff + f
//
// h[f] is in scratch a1 region: [token_row * 2*d_ff + f]
// Output y[dd] to secondary scratch region: [total_kept * 2*d_ff + token_row * D + dd]

[numthreads(64, 1, 1)]
void CSPoolMatMulW2(uint3 DTid : SV_DispatchThreadID)
{
    uint idx = DTid.x;
    if (idx >= total_kept * D)
        return;

    uint token_row = idx / D;
    uint dd = idx % D;

    // Load slot from assignment
    uint assign_base = token_row * 16;
    uint slot = g_assignments.Load(assign_base);

    // W2[s, dd, f] at (slot * D * d_ff + dd * d_ff + f)
    uint w2_row_offset = slot * D * d_ff + dd * d_ff;

    float acc = 0.0f;
    for (uint f = 0; f < d_ff; ++f)
    {
        float hf = load_f32(g_scratch_out, (token_row * (d_ff * 2) + f) * 4);
        float w = load_f32(g_w2_weights, (w2_row_offset + f) * 4);
        acc += hf * w;
    }

    // Store to secondary region: [total_kept * 2 * d_ff + token_row * D + dd]
    uint out_offset = total_kept * (d_ff * 2) + token_row * D + dd;
    store_f32(g_scratch_out, out_offset * 4, acc);
}

// ==================== Kernel 3: Scatter-Add Weighted ====================
// For each kept token: scatter_add[token_idx, d] += weight * y[d]
// Thread idx = token_row * D + dd
//
// g_assignments: {slot:u32, token:u32, weight:f32, _pad:u32}
// g_scratch_out: mlp output y at [total_kept * 2*d_ff + token_row * D + dd]
// g_scatter_add: [N * D] — atomic accumulate (multiple token_rows may map to same token)

[numthreads(64, 1, 1)]
void CSPoolScatterAdd(uint3 DTid : SV_DispatchThreadID)
{
    uint idx = DTid.x;
    if (idx >= total_kept * D)
        return;

    uint token_row = idx / D;
    uint dd = idx % D;

    // Load assignment: {slot, token, weight} — 4 bytes each, 16 bytes per assignment
    uint assign_base = token_row * 16;
    uint slot = g_assignments.Load(assign_base);          // not needed for scatter
    uint token_idx = g_assignments.Load(assign_base + 4); // byte offset 4
    float weight = asfloat(g_assignments.Load(assign_base + 8)); // byte offset 8

    // Load mlp output y
    uint mlp_offset = total_kept * (d_ff * 2) + token_row * D + dd;
    float val = load_f32(g_scratch_out, mlp_offset * 4);
    float weighted = val * weight;

    // Atomic add to scatter buffer
    uint out_offset = (token_idx * D + dd) * 4;
    atomic_add_f32(g_scatter_add, out_offset, weighted);
}

// ==================== Entry points for DX12 ====================

[numthreads(64, 1, 1)]
void main_pool_matmul_w1w3(uint3 DTid : SV_DispatchThreadID)
{
    CSPoolMatMulW1W3(DTid);
}

[numthreads(64, 1, 1)]
void main_pool_silu_mul(uint3 DTid : SV_DispatchThreadID)
{
    CSPoolSiLUMul(DTid);
}

[numthreads(64, 1, 1)]
void main_pool_matmul_w2(uint3 DTid : SV_DispatchThreadID)
{
    CSPoolMatMulW2(DTid);
}

[numthreads(64, 1, 1)]
void main_pool_scatter_add(uint3 DTid : SV_DispatchThreadID)
{
    CSPoolScatterAdd(DTid);
}

#endif // _POOL_HLSL_
