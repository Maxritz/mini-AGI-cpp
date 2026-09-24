// src/shaders/dense.hlsl
// DirectX 12 HLSL compute shaders for mini-AGI dense transformer.
// Standard 12-binding DSL:
//   binding 0 = activation (input/output, scratch)
//   binding 1 = qkv_w
//   binding 2 = proj_w
//   binding 3 = ln1_w / ln2_w / weight (elementwise)
//   binding 4 = kv_cache_k
//   binding 5 = kv_cache_v
//   binding 6 = cos (RoPE)
//   binding 7 = sin (RoPE)
//   binding 8 = output
//   binding 9 = scratch
//   binding 10 = ln2_w / mlp weights
//   binding 11 = mlp_w / head_w / etc

#ifndef _DENSE_HLSL_
#define _DENSE_HLSL_

// Use std430-equivalent scalar layout (matches VAiT scalar convention).
// HLSL structured buffer with scalar layout is natural.

// Push constants (32-bit root constants in DX12)
cbuffer PushConstants : register(b0)
{
    uint  dispatch_id;    // which shader to run (0=matmul, 1=rmsnorm, etc.)
    uint  M;              // rows of activation
    uint  N;              // cols of activation / output cols
    uint  K;              // inner dim for matmul
    uint  T;              // sequence length
    uint  H;              // num heads
    uint  hd;             // head dim
    uint  pos_offset;     // position offset for RoPE
    uint  kv_len;         // KV cache length (cached_T + T)
    uint  cached_T;       // cached tokens in KV
    uint  head_dim;       // alias for hd
    uint  d_model;        // model dim
    uint  d_ff;           // MLP intermediate dim
    uint  stride3;        // 3 * d_model
    float eps;            // epsilon for RMSNorm
    float inv_sqrt_hd;    // 1/sqrt(hd) for attention scaling
    uint  mode;           // 0=plain, 1=with cache concat
    uint  pad0;
    uint  pad1;
    uint  pad2;
};

// Storage buffers (std430 scalar layout)
// binding 0: activation buffer (read/write scratch/output)
ByteAddressBuffer g_activation : register(t0);
RWByteAddressBuffer g_activation_out : register(u0);

// binding 1: qkv_w or weight matrix A for matmul
ByteAddressBuffer g_qkv_w : register(t1);
RWByteAddressBuffer g_qkv_out : register(u1);

// binding 2: proj_w or weight matrix B for matmul
ByteAddressBuffer g_proj_w : register(t2);
RWByteAddressBuffer g_proj_out : register(u2);

// binding 3: ln1_w / weight (1D, d_model elements)
ByteAddressBuffer g_weight : register(t3);
RWByteAddressBuffer g_weight_out : register(u3);

// binding 4: KV cache K key
ByteAddressBuffer g_kv_k : register(t4);
RWByteAddressBuffer g_kv_k_out : register(u4);

// binding 5: KV cache V value
ByteAddressBuffer g_kv_v : register(t5);
RWByteAddressBuffer g_kv_v_out : register(u5);

// binding 6: cos (RoPE)
ByteAddressBuffer g_cos : register(t6);
RWByteAddressBuffer g_cos_out : register(u6);

// binding 7: sin (RoPE)
ByteAddressBuffer g_sin : register(t7);
RWByteAddressBuffer g_sin_out : register(u7);

// binding 8: output logits
RWByteAddressBuffer g_output : register(u8);

// binding 9: scratch buffer (general purpose)
RWByteAddressBuffer g_scratch : register(u9);

// binding 10: ln2_w or mlp w1/w3/w2 weights
ByteAddressBuffer g_mlp_w1 : register(t10);
RWByteAddressBuffer g_mlp_w1_out : register(u10);

// binding 11: head_w or additional weight
ByteAddressBuffer g_head_w : register(t11);
RWByteAddressBuffer g_head_w_out : register(u11);

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

// ==================== Kernel 0: MatMul FP32 ====================
// C = A * B^T where A is [M,K], B is [N,K] stored as [N,K], output [M,N]
// A read from g_qkv_w (binding 1), B read from g_proj_w (binding 2)
// Output to g_activation_out (binding 0)
[numthreads(8, 8, 1)]
void CSMatMulFP32(uint3 DTid : SV_DispatchThreadID)
{
    uint row = DTid.x;
    uint col = DTid.y;
    if (row >= M || col >= N)
        return;

    float sum = 0.0f;
    for (uint k = 0; k < K; ++k)
    {
        float a = load_f32(g_qkv_w, (row * K + k) * 4);
        float b = load_f32(g_proj_w, (col * K + k) * 4);
        sum += a * b;
    }
    store_f32(g_activation_out, (row * N + col) * 4, sum);
}

// ==================== Kernel 1: RMSNorm ====================
// y = rms_norm(x, weight, eps)
// x read from g_activation (binding 0), weight from g_weight (binding 3)
// output to g_activation_out (binding 0)
[numthreads(256, 1, 1)]
void CSRMSNorm(uint3 DTid : SV_DispatchThreadID)
{
    uint row = DTid.x;
    if (row >= M)
        return;

    float sum_sq = 0.0f;
    for (uint c = 0; c < N; ++c)
    {
        float val = load_f32(g_activation, (row * N + c) * 4);
        sum_sq += val * val;
    }
    float mean_sq = sum_sq / N;
    float scale = rsqrt(mean_sq + eps);

    for (uint c = 0; c < N; ++c)
    {
        float val = load_f32(g_activation, (row * N + c) * 4);
        float w = load_f32(g_weight, c * 4);
        float result = val * scale * w;
        store_f32(g_activation_out, (row * N + c) * 4, result);
    }
}

// ==================== Kernel 2: Softmax (row-wise) ====================
// Row-wise softmax on g_activation [M, N], output to g_activation_out
[numthreads(1, 16, 1)]
void CSSoftmax(uint3 DTid : SV_DispatchThreadID)
{
    uint row = DTid.x;
    uint lane = DTid.y;
    if (row >= M)
        return;

    // Phase 1: find max
    float max_val = -1e30f;
    if (lane < N)
    {
        max_val = load_f32(g_activation, (row * N + lane) * 4);
    }
    // Barrier within thread group for reduction
    GroupMemoryBarrierWithGroupSync();

    for (uint stride = N / 2; stride > 0; stride /= 2)
    {
        if (lane < stride && lane + stride < N)
        {
            float other = load_f32(g_activation, (row * N + lane + stride) * 4);
            if (other > max_val) max_val = other;
        }
        GroupMemoryBarrierWithGroupSync();
    }

    // Phase 2: compute exp and sum
    float exp_val = 0.0f;
    if (lane < N)
    {
        float val = load_f32(g_activation, (row * N + lane) * 4);
        exp_val = exp(val - max_val);
    }
    GroupMemoryBarrierWithGroupSync();

    for (uint stride = N / 2; stride > 0; stride /= 2)
    {
        if (lane < stride && lane + stride < N)
        {
            exp_val += load_f32(g_scratch, (lane + stride) * 4);
        }
        store_f32(g_scratch, lane * 4, asuint(exp_val));
        GroupMemoryBarrierWithGroupSync();
    }
    float sum = load_f32(g_scratch, 0);

    // Phase 3: normalize
    if (lane < N)
    {
        float val = load_f32(g_activation, (row * N + lane) * 4);
        float result = exp(val - max_val) / (sum + 1e-30f);
        store_f32(g_activation_out, (row * N + lane) * 4, result);
    }
}

// ==================== Kernel 3: RoPE (rotary position embedding) ====================
// x [B*H*T, hd] with cos/sin [T, hd/2]
// cos from g_cos (binding 6), sin from g_sin (binding 7)
// input x from g_activation (binding 0), output to g_activation_out
[numthreads(256, 1, 1)]
void CSRoPE(uint3 DTid : SV_DispatchThreadID)
{
    uint idx = DTid.x;
    if (idx >= M)
        return;

    uint half = hd / 2;
    uint t = idx / (H * T); // B=1 so bh = idx, t = idx / H... need reshape
    // Actually: layout is [B*H*T, hd], element (b,h,t,d) at flat (b*H*T + h*T + t)*hd + d
    // So thread idx maps to a full vector of hd elements
    uint bh = idx / T;  // since B=1, bh = h
    uint tt = idx % T;  // t within block

    uint row = tt;  // position within segment (not pos_offset+t)
    for (uint i = 0; i < half; ++i)
    {
        float c = load_f32(g_cos, (row * half + i) * 4);
        float s = load_f32(g_sin, (row * half + i) * 4);
        float x1 = load_f32(g_activation, (idx * hd + 2 * i) * 4);
        float x2 = load_f32(g_activation, (idx * hd + 2 * i + 1) * 4);
        store_f32(g_activation_out, (idx * hd + 2 * i) * 4, x1 * c - x2 * s);
        store_f32(g_activation_out, (idx * hd + 2 * i + 1) * 4, x1 * s + x2 * c);
    }
}

// ==================== Kernel 4: Embedding Gather ====================
// emb [V, d_model] from g_qkv_w (binding 1), idx [B*T] from g_activation (binding 0)
// output [B*T, d_model] to g_activation_out
[numthreads(256, 1, 1)]
void CSEmbeddingGather(uint3 DTid : SV_DispatchThreadID)
{
    uint bt = DTid.x;
    if (bt >= M)
        return;

    int32_t id = (int32_t)load_f32(g_activation, bt * 4);
    if (id < 0 || id >= K) // K holds V (vocab_size) here
    {
        for (uint d = 0; d < N; ++d)
            store_f32(g_activation_out, (bt * N + d) * 4, 0.0f);
    }
    else
    {
        for (uint d = 0; d < N; ++d)
            store_f32(g_activation_out, (bt * N + d) * 4,
                      load_f32(g_qkv_w, (id * N + d) * 4));
    }
}

// ==================== Kernel 5: SiLU (elementwise) ====================
// y = silu(x) = x * sigmoid(x)
[numthreads(256, 1, 1)]
void CSSiLU(uint3 DTid : SV_DispatchThreadID)
{
    uint idx = DTid.x;
    if (idx >= M)
        return;
    float x = load_f32(g_activation, idx * 4);
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
    store_f32(g_activation_out, idx * 4, x * sig);
}

// ==================== Kernel 6: GELU (elementwise) ====================
// y = gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
[numthreads(256, 1, 1)]
void CSGELU(uint3 DTid : SV_DispatchThreadID)
{
    uint idx = DTid.x;
    if (idx >= M)
        return;
    float x = load_f32(g_activation, idx * 4);
    float inner = 0.7978845608f * (x + 0.044715f * x * x * x);
    float val = 0.5f * x * (1.0f + tanh(inner));
    store_f32(g_activation_out, idx * 4, val);
}

// ==================== Kernel 7: Silu * Gate (elementwise) ====================
// y = silu(x) * y_gate  (for SwiGLU: x from g_activation, gate from g_weight)
[numthreads(256, 1, 1)]
void CSSiLUGate(uint3 DTid : SV_DispatchThreadID)
{
    uint idx = DTid.x;
    if (idx >= M)
        return;
    float x = load_f32(g_activation, idx * 4);
    float gate = load_f32(g_weight, idx * 4);
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
    store_f32(g_activation_out, idx * 4, x * sig * gate);
}

// ==================== Kernel 8: Add (residual elementwise) ====================
// y = a + b; a from g_activation, b from g_weight, output to g_activation_out
[numthreads(256, 1, 1)]
void CSAdd(uint3 DTid : SV_DispatchThreadID)
{
    uint idx = DTid.x;
    if (idx >= M)
        return;
    float a = load_f32(g_activation, idx * 4);
    float b = load_f32(g_weight, idx * 4);
    store_f32(g_activation_out, idx * 4, a + b);
}

// ==================== Kernel 9: Attention Scores (matmul Q x K^T) ====================
// Computes Q @ K^T with causal masking.
// Q from g_activation (binding 0) [B*H*T, hd]
// K from g_qkv_w (binding 1) [B*H*kv_len, hd]
// scores output to g_scratch (binding 9) [B*H*T, kv_len]
[numthreads(8, 8, 1)]
void CSAttnScores(uint3 DTid : SV_DispatchThreadID)
{
    uint bh_t = DTid.x;  // (bh * T + t) index
    uint j = DTid.y;     // key index
    if (bh_t >= M || j >= kv_len)
        return;

    float dot = 0.0f;
    for (uint d = 0; d < hd; ++d)
    {
        float q = load_f32(g_activation, (bh_t * hd + d) * 4);
        float k = load_f32(g_qkv_w, (j * hd + d) * 4);
        dot += q * k;
    }
    dot *= inv_sqrt_hd;

    // Causal mask: extract t (within block) from bh_t
    uint t = bh_t % T;  // position within the current block
    int P = (int)kv_len - (int)T;  // cached_T
    if ((int)j > (int)t + P)
        dot = -1e30f;

    store_f32(g_scratch, (bh_t * kv_len + j) * 4, dot);
}

// ==================== Kernel 10: Attention Weighted Sum ====================
// out = scores @ V
// scores from g_scratch (binding 9) [B*H*T, kv_len]
// V from g_qkv_w (binding 1) [B*H*kv_len, hd]
// output to g_activation_out (binding 0) [B*H*T, hd]
[numthreads(8, 32, 1)]
void CSAttnWeightedSum(uint3 DTid : SV_DispatchThreadID)
{
    uint bh_t = DTid.x;  // (bh * T + t) index
    uint d = DTid.y;     // head dim
    if (bh_t >= M || d >= hd)
        return;

    float acc = 0.0f;
    for (uint j = 0; j < kv_len; ++j)
    {
        float s = load_f32(g_scratch, (bh_t * kv_len + j) * 4);
        float v = load_f32(g_qkv_w, (j * hd + d) * 4);
        acc += s * v;
    }
    store_f32(g_activation_out, (bh_t * hd + d) * 4, acc);
}

// ==================== Kernel 11: Elementwise Copy ====================
// out_buf[gid] = in_buf[gid]
[numthreads(256, 1, 1)]
void CSElementwiseCopy(uint3 DTid : SV_DispatchThreadID)
{
    uint gid = DTid.x;
    if (gid >= M)
        return;
    float val = load_f32(g_activation, gid * 4);
    store_f32(g_activation_out, gid * 4, val);
}

// ==================== Entry point for DX12 ====================
// Each entry point maps to a specific kernel for DX12 dispatch

[numthreads(8, 8, 1)]
void main_matmul(uint3 DTid : SV_DispatchThreadID)
{
    CSMatMulFP32(DTid);
}

[numthreads(256, 1, 1)]
void main_rmsnorm(uint3 DTid : SV_DispatchThreadID)
{
    CSRMSNorm(DTid);
}

[numthreads(1, 16, 1)]
void main_softmax(uint3 DTid : SV_DispatchThreadID)
{
    CSSoftmax(DTid);
}

[numthreads(256, 1, 1)]
void main_rope(uint3 DTid : SV_DispatchThreadID)
{
    CSRoPE(DTid);
}

[numthreads(256, 1, 1)]
void main_embedding_gather(uint3 DTid : SV_DispatchThreadID)
{
    CSEmbeddingGather(DTid);
}

[numthreads(256, 1, 1)]
void main_silu(uint3 DTid : SV_DispatchThreadID)
{
    CSSiLU(DTid);
}

[numthreads(256, 1, 1)]
void main_gelu(uint3 DTid : SV_DispatchThreadID)
{
    CSGELU(DTid);
}

[numthreads(256, 1, 1)]
void main_silu_gate(uint3 DTid : SV_DispatchThreadID)
{
    CSSiLUGate(DTid);
}

[numthreads(256, 1, 1)]
void main_add(uint3 DTid : SV_DispatchThreadID)
{
    CSAdd(DTid);
}

[numthreads(8, 8, 1)]
void main_attn_scores(uint3 DTid : SV_DispatchThreadID)
{
    CSAttnScores(DTid);
}

[numthreads(8, 32, 1)]
void main_attn_wsum(uint3 DTid : SV_DispatchThreadID)
{
    CSAttnWeightedSum(DTid);
}

[numthreads(256, 1, 1)]
void main_copy(uint3 DTid : SV_DispatchThreadID)
{
    CSElementwiseCopy(DTid);
}

#endif // _DENSE_HLSL_
