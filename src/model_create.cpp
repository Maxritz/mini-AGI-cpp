// src/model_create.cpp
#include "model_create.hpp"
#include "tensor.hpp"
#include "init.hpp"
#include <cmath>

namespace minagi {

// Truncated normal: sample normal, clamp to [-2*std, 2*std], repeat up to
// 10 tries. This matches minagi's use of torch.nn.init.normal_ with manual
// truncation applied by PyTorch's default for Embedding and Linear.
static float trunc_normal(PcgRng& rng, float mean, float std) {
    for (int retry = 0; retry < 10; ++retry) {
        float v = normal_sample(rng, mean, std);
        if (v >= mean - 2*std && v <= mean + 2*std) return v;
    }
    return normal_sample(rng, mean, std);
}

// Normal init with same std as Python:
//   nn.init.normal_(m.weight, 0.0, 0.02)
// then depth-scaled for proj/w2:
//   nn.init.normal_(p, 0.0, 0.02 / sqrt(2 * depth))
static mt::Tensor init_w(const mt::Shape& s, PcgRng& rng, float std) {
    mt::Tensor t = mt::make(s, mt::DType::FP32, 0.0f);
    float* p = t.ptr<float>();
    int64_t n = s.numel();
    for (int64_t i = 0; i < n; ++i) {
        p[i] = trunc_normal(rng, 0.0f, std);
    }
    return t;
}

std::map<std::string, mt::Tensor> build_model(const Config& cfg, PcgRng& rng) {
    std::map<std::string, mt::Tensor> out;

    const int V = cfg.vocab_size;
    const int d = cfg.d_model;
    const int dff = cfg.d_ff;
    const int n_experts = cfg.pool_experts;
    const int pool_dff = cfg.pool_d_ff;
    const int pool_depth = cfg.pool_depth;
    const int total_depth = cfg.n_prelude + cfg.n_recur + cfg.n_coda;
    const float init_scale = 0.02f;
    const float proj_scale = init_scale / std::sqrt(2.0f * static_cast<float>(total_depth));

    // --- tok_emb ---
    // nn.init.normal_(m.weight, 0.0, 0.02)
    mt::Shape emb_shape;
    emb_shape.rank = 2;
    emb_shape.d[0] = V;
    emb_shape.d[1] = d;
    out["tok_emb.weight"] = init_w(emb_shape, rng, init_scale);

    // --- Block builder ---
    auto add_dense_block = [&](const std::string& prefix) {
        // ln1, ln2, attn.qkv, attn.proj, mlp.w1, mlp.w3, mlp.w2
        mt::Shape s1d;
        s1d.rank = 1;
        s1d.d[0] = d;
        out[prefix + "ln1.weight"] = init_const(s1d, 1.0f);  // RMSNorm = 1.0
        out[prefix + "ln2.weight"] = init_const(s1d, 1.0f);

        mt::Shape qkv_shape;
        qkv_shape.rank = 2;
        qkv_shape.d[0] = 3 * d;
        qkv_shape.d[1] = d;
        out[prefix + "attn.qkv.weight"] = init_w(qkv_shape, rng, init_scale);

        mt::Shape proj_shape;
        proj_shape.rank = 2;
        proj_shape.d[0] = d;
        proj_shape.d[1] = d;
        out[prefix + "attn.proj.weight"] = init_w(proj_shape, rng, proj_scale);

        mt::Shape mlp_shape;
        mlp_shape.rank = 2;
        mlp_shape.d[0] = dff;
        mlp_shape.d[1] = d;
        out[prefix + "mlp.w1.weight"] = init_w(mlp_shape, rng, proj_scale);
        out[prefix + "mlp.w3.weight"] = init_w(mlp_shape, rng, proj_scale);

        mt::Shape w2_shape;
        w2_shape.rank = 2;
        w2_shape.d[0] = d;
        w2_shape.d[1] = dff;
        out[prefix + "mlp.w2.weight"] = init_w(w2_shape, rng, proj_scale);
    };

    auto add_pooled_block = [&](const std::string& prefix) {
        // ln1, ln2, attn.qkv, attn.proj, mlp.router.weight, mlp.depth_emb
        mt::Shape s1d;
        s1d.rank = 1;
        s1d.d[0] = d;
        out[prefix + "ln1.weight"] = init_const(s1d, 1.0f);
        out[prefix + "ln2.weight"] = init_const(s1d, 1.0f);

        mt::Shape qkv_shape;
        qkv_shape.rank = 2;
        qkv_shape.d[0] = 3 * d;
        qkv_shape.d[1] = d;
        out[prefix + "attn.qkv.weight"] = init_w(qkv_shape, rng, init_scale);

        mt::Shape proj_shape;
        proj_shape.rank = 2;
        proj_shape.d[0] = d;
        proj_shape.d[1] = d;
        out[prefix + "attn.proj.weight"] = init_w(proj_shape, rng, proj_scale);

        mt::Shape router_shape;
        router_shape.rank = 2;
        router_shape.d[0] = n_experts;
        router_shape.d[1] = d;
        // router.weight: nn.Linear default init (normal 0.02), depth_emb: zeros
        out[prefix + "mlp.router.weight"] = init_w(router_shape, rng, init_scale);
        out[prefix + "mlp.depth_emb"] = init_zeros(s1d);
    };

    // --- prelude blocks (dense MLP) ---
    for (int i = 0; i < cfg.n_prelude; ++i) {
        add_dense_block("prelude." + std::to_string(i) + ".");
    }

    // --- recurrent blocks (pooled MLP) ---
    for (int i = 0; i < cfg.n_recur; ++i) {
        add_pooled_block("recur." + std::to_string(i) + ".");
    }

    // --- coda blocks (pooled MLP) ---
    for (int i = 0; i < cfg.n_coda; ++i) {
        add_pooled_block("coda." + std::to_string(i) + ".");
    }

    // --- adapter.weight = [I | I] ---
    mt::Shape adapter_shape;
    adapter_shape.rank = 2;
    adapter_shape.d[0] = d;
    adapter_shape.d[1] = 2 * d;
    out["adapter.weight"] = init_adapter(d);

    // --- ln_f.weight = 1.0 ---
    mt::Shape lnf_shape;
    lnf_shape.rank = 1;
    lnf_shape.d[0] = d;
    out["ln_f.weight"] = init_const(lnf_shape, 1.0f);

    // --- head.weight (if not tied) ---
    if (!cfg.tie_embeddings) {
        out["head.weight"] = init_w(emb_shape, rng, init_scale);
    }

    // --- halt.weight * 0.01, halt.bias = -2.0 ---
    mt::Shape halt_w_shape;
    halt_w_shape.rank = 2;
    halt_w_shape.d[0] = 1;
    halt_w_shape.d[1] = d;
    {
        mt::Tensor w = init_w(halt_w_shape, rng, init_scale);
        float* p = w.ptr<float>();
        for (int64_t i = 0; i < d; ++i) p[i] *= 0.01f;
        out["halt.weight"] = std::move(w);
    }

    mt::Shape halt_b_shape;
    halt_b_shape.rank = 1;
    halt_b_shape.d[0] = 1;
    out["halt.bias"] = init_const(halt_b_shape, -2.0f);

    // --- pool wiring ---
    if (cfg.use_pool) {
        // pool.gate = 1.0
        mt::Shape gate_shape;
        gate_shape.rank = 1;
        gate_shape.d[0] = n_experts;
        out["pool.gate"] = init_const(gate_shape, 1.0f);

        // pool.segment_router.weight ~ normal(0, 0.02)
        mt::Shape seg_shape;
        seg_shape.rank = 2;
        seg_shape.d[0] = n_experts;
        seg_shape.d[1] = d;
        out["pool.segment_router.weight"] = init_w(seg_shape, rng, init_scale);

        // --- expert weights ---
        // Each expert: w1 [pool_dff, d], w3 [pool_dff, d], w2 [d, pool_dff]
        // If pool_depth > 1, also blocks.{i}.{w1,w3,w2}
        for (int eid = 0; eid < n_experts; ++eid) {
            std::string prefix = "pool.experts." + std::to_string(eid) + ".";
            mt::Shape w1_shape, w2_shape;
            w1_shape.rank = 2;
            w1_shape.d[0] = pool_dff;
            w1_shape.d[1] = d;
            w2_shape.rank = 2;
            w2_shape.d[0] = d;
            w2_shape.d[1] = pool_dff;

            out[prefix + "w1.weight"] = init_w(w1_shape, rng, init_scale);
            out[prefix + "w3.weight"] = init_w(w1_shape, rng, init_scale);
            out[prefix + "w2.weight"] = init_w(w2_shape, rng, proj_scale);

            // additional depth blocks (blocks.1, blocks.2, ...)
            for (int blk = 1; blk < pool_depth; ++blk) {
                std::string bprefix = prefix + "blocks." + std::to_string(blk) + ".";
                out[bprefix + "w1.weight"] = init_w(w1_shape, rng, init_scale);
                out[bprefix + "w3.weight"] = init_w(w1_shape, rng, init_scale);
                out[bprefix + "w2.weight"] = init_w(w2_shape, rng, proj_scale);
            }
        }
    }

    return out;
}

std::map<std::string, mininpz::Array> tensors_to_state(
    const std::map<std::string, mt::Tensor>& model_tensors) {
    std::map<std::string, mininpz::Array> state;
    for (const auto& kv : model_tensors) {
        const mt::Tensor& t = kv.second;
        mininpz::Array arr;
        arr.dtype = mininpz::DType::F4;
        arr.shape.rank = t.shape.rank;
        for (size_t i = 0; i < t.shape.rank && i < 4; ++i) {
            arr.shape.d[i] = static_cast<size_t>(t.shape.d[i]);
        }
        arr.bytes = t.data_;
        state[kv.first] = arr;
    }
    return state;
}

}  // namespace minagi
