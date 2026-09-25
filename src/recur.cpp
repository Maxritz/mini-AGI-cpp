// src/recur.cpp
// Inference forward of the latent-recurrence model (minagi/recur.py RecurCoder
// with targets=None). The dense transformer blocks are reimplemented here to
// match src/model.cpp bit-for-bit (model.cpp is a provided library and is NOT
// modified) -- the recurrence steps a SHARED set of weights over multiple KV
// cache positions, which model.cpp's single-stack forward does not do.
#include "recur.hpp"
#include "mininpz.hpp"
#include "init.hpp"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <iostream>
#include <map>
#include <sstream>

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

mt::Tensor zeros2d(int64_t r, int64_t c) {
    mt::Shape s;
    s.rank = 2;
    s.d[0] = r;
    s.d[1] = c;
    return mt::make(s, mt::DType::FP32, 0.0f);
}

mt::Shape shape1(int64_t d0) { mt::Shape s; s.rank = 1; s.d[0] = d0; return s; }
mt::Shape shape2(int64_t r0, int64_t r1) { mt::Shape s; s.rank = 2; s.d[0] = r0; s.d[1] = r1; return s; }

mt::Tensor make_rank0() {
    mt::Shape s;
    s.rank = 0;
    return mt::make(s, mt::DType::FP32, 0.0f);
}

mt::Tensor transpose_2d(const mt::Tensor& a) {
    const int64_t M = a.shape.d[0];
    const int64_t N = a.shape.d[1];
    mt::Tensor out = zeros2d(N, M);
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            out.set_flat(j * M + i, a.at_flat(i * N + j));
        }
    }
    return out;
}

// y [T,d] = x [T,d] + silu(x@ln2) gated mlp; prelude block only.
mt::Tensor dense_mlp_impl(const mt::Tensor& x, const mt::Tensor& w1,
                          const mt::Tensor& w3, const mt::Tensor& w2,
                          const mt::Tensor& ln2_w, int64_t T, int64_t d,
                          int64_t dff) {
    mt::Tensor xn = mt::rms_norm(x, ln2_w, 1e-6f);
    mt::Tensor h1 = mt::matmul(xn, transpose_2d(w1));
    mt::Tensor h3 = mt::matmul(xn, transpose_2d(w3));
    mt::Tensor g = zeros2d(T, dff);
    float* gp = g.ptr<float>();
    for (int64_t i = 0; i < T * dff; ++i) {
        gp[i] = silu1(h1.atf(i)) * h3.atf(i);
    }
    mt::Tensor y = mt::matmul(g, transpose_2d(w2));
    // residual
    for (int64_t i = 0; i < T * d; ++i) {
        y.set_flat(i, x.at_flat(i) + y.at_flat(i));
    }
    return y;
}

// Single transformer block attention (cache-aware), mirroring model.cpp::attn.
// Returns x + proj(attn(rms_norm(x))) using caches[ci] (grown in place).
void attn_residual(mt::Tensor& x, const mt::Tensor& ln1_w,
                   const mt::Tensor& qkv_w, const mt::Tensor& proj_w,
                   std::vector<model::KVCache>& caches, int ci,
                   int64_t T, int64_t H, int64_t hd, int64_t d,
                    int64_t pos_offset, const mt::Tensor& cos,
                    const mt::Tensor& sin) {
    (void)pos_offset;
    const int64_t B = 1;
    const int64_t M = B * T;
    mt::Tensor xn = mt::rms_norm(x, ln1_w, 1e-6f);               // [M, d]
    mt::Tensor qkv = mt::matmul(xn, transpose_2d(qkv_w));         // [M, 3d]
    mt::Tensor q = zeros2d(M, d);
    mt::Tensor k = zeros2d(M, d);
    mt::Tensor v = zeros2d(M, d);
    const int64_t stride3 = 3 * d;
    for (int64_t bt = 0; bt < M; ++bt) {
        for (int64_t h = 0; h < H; ++h) {
            for (int64_t dd = 0; dd < hd; ++dd) {
                const int64_t src = bt * d + h * hd + dd;
                q.set_flat(src, qkv.at_flat(bt * stride3 + h * hd + dd));
                k.set_flat(src, qkv.at_flat(bt * stride3 + d + h * hd + dd));
                v.set_flat(src, qkv.at_flat(bt * stride3 + 2 * d + h * hd + dd));
            }
        }
    }
    // reshape q,k,v [M,d] into qh/kh/vh laid out as [B*H*T, hd] = [B,H,T,hd]
    // (head-major), i.e. element (b,h,t,dd) at flat (b*H*T + h*T + t)*hd + dd.
    mt::Tensor qh = zeros2d(B * H * T, hd);
    mt::Tensor kh = zeros2d(B * H * T, hd);
    mt::Tensor vh = zeros2d(B * H * T, hd);
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t t = 0; t < T; ++t) {
            const int64_t bt = b * T + t;
            for (int64_t h = 0; h < H; ++h) {
                for (int64_t dd = 0; dd < hd; ++dd) {
                    const int64_t src = bt * d + h * hd + dd;
                    const int64_t dst = (b * (H * T) + h * T + t) * hd + dd;
                    qh.set_flat(dst, q.at_flat(src));
                    kh.set_flat(dst, k.at_flat(src));
                    vh.set_flat(dst, v.at_flat(src));
                }
            }
        }
    }
    int64_t half = hd / 2;
    // apply rope to q,k,v. cos/sin are pre-sliced to [T, half] covering
    // positions pos_offset..pos_offset+T-1, so index by t (not pos_offset+t).
    auto rot = [&](mt::Tensor& tt) {
        for (int64_t b = 0; b < B; ++b) {
            for (int64_t h = 0; h < H; ++h) {
                for (int64_t t = 0; t < T; ++t) {
                    const int64_t row = t;
                    for (int64_t i = 0; i < half; ++i) {
                        const float c = cos.atf(row * half + i);
                        const float s = sin.atf(row * half + i);
                        const float x1 = tt.atf(((b * H + h) * T + t) * hd + 2 * i);
                        const float x2 = tt.atf(((b * H + h) * T + t) * hd + 2 * i + 1);
                        tt.set_flat(((b * H + h) * T + t) * hd + 2 * i, x1 * c - x2 * s);
                        tt.set_flat(((b * H + h) * T + t) * hd + 2 * i + 1, x1 * s + x2 * c);
                    }
                }
            }
        }
    };
    rot(qh);
    rot(kh);

    int64_t kv_len = T;
    mt::Tensor k_concat = kh;
    mt::Tensor v_concat = vh;
    if (caches[ci].active && caches[ci].k.shape.d[2] > 0) {
        const int64_t cached_T = caches[ci].k.shape.d[2];
        kv_len = cached_T + T;
        const int64_t BH = B * H;
        const int64_t hd_total = hd;
        // k_concat/v_concat are bh-major [B*H*kv_len, hd]. The persistent
        // cache mirrors Python's [B, n_head, kv_len, head_dim] layout
        // (bh-major, flat (bh*cached_T + pos)*hd + d), so reads/writes map
        // bh-major <-> bh-major directly (B=1).
        k_concat = zeros2d(BH * kv_len, hd);
        v_concat = zeros2d(BH * kv_len, hd);
        for (int64_t bh = 0; bh < BH; ++bh) {
            for (int64_t cc = 0; cc < cached_T; ++cc) {
                for (int64_t cd = 0; cd < hd_total; ++cd) {
                    k_concat.set_flat((bh * kv_len + cc) * hd + cd,
                                      caches[ci].k.at_flat((bh * cached_T + cc) * hd + cd));
                    v_concat.set_flat((bh * kv_len + cc) * hd + cd,
                                      caches[ci].v.at_flat((bh * cached_T + cc) * hd + cd));
                }
            }
            for (int64_t t = 0; t < T; ++t) {
                for (int64_t dd = 0; dd < hd_total; ++dd) {
                    k_concat.set_flat((bh * kv_len + cached_T + t) * hd + dd,
                                      kh.at_flat((bh * T + t) * hd + dd));
                    v_concat.set_flat((bh * kv_len + cached_T + t) * hd + dd,
                                      vh.at_flat((bh * T + t) * hd + dd));
                }
            }
        }
    }

    const int64_t BH_T = B * H * T;
    mt::Tensor scores = zeros2d(BH_T, kv_len);
    const float inv_sqrt = 1.0f / std::sqrt(static_cast<float>(hd));
    const int64_t P = kv_len - T;
    float* sp = scores.ptr<float>();
    for (int64_t bh = 0; bh < B * H; ++bh) {
        for (int64_t i = 0; i < T; ++i) {
            for (int64_t j = 0; j < kv_len; ++j) {
                double dot = 0.0;
                for (int64_t ddi = 0; ddi < hd; ++ddi) {
                    dot += static_cast<double>(qh.atf((bh * T + i) * hd + ddi)) *
                           static_cast<double>(k_concat.atf((bh * kv_len + j) * hd + ddi));
                }
                dot *= inv_sqrt;
                if (j > i + P) dot = -1e30;
                sp[(bh * T + i) * kv_len + j] = static_cast<float>(dot);
            }
        }
    }
    mt::Tensor sm = mt::softmax(scores, 1);
    // weighted sum -> [B,H,T,hd]
    mt::Tensor out_h = zeros2d(B * H * T, hd);
    float* op = out_h.ptr<float>();
    for (int64_t bh = 0; bh < B * H; ++bh) {
        for (int64_t t = 0; t < T; ++t) {
            for (int64_t dd = 0; dd < hd; ++dd) {
                double acc = 0.0;
                for (int64_t j = 0; j < kv_len; ++j) {
                    acc += static_cast<double>(sm.atf((bh * T + t) * kv_len + j)) *
                           static_cast<double>(v_concat.atf((bh * kv_len + j) * hd + dd));
                }
                op[(bh * T + t) * hd + dd] = static_cast<float>(acc);
            }
        }
    }
    // back to B,T,d (interleaving heads). out_h is laid out [B,H,T,hd]
    // (flat (b*H*T + h*T + t)*hd + dd); gather head h of token t into slot h*hd.
    mt::Tensor out_flat = zeros2d(M, d);
    float* fp = out_flat.ptr<float>();
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t t = 0; t < T; ++t) {
            const int64_t bt = b * T + t;
            for (int64_t h = 0; h < H; ++h) {
                for (int64_t dd = 0; dd < hd; ++dd) {
                    fp[bt * d + h * hd + dd] = out_h.atf((b * (H * T) + h * T + t) * hd + dd);
                }
            }
        }
    }
    mt::Tensor proj_out = mt::matmul(out_flat, transpose_2d(proj_w)); // [M, d]

    // write cache back as [B, H, kv_len, hd] (Python [B, n_head, kv_len, head_dim],
    // bh-major flat). k_concat is bh-major, so a direct copy is correct.
    mt::Shape csh;
    csh.rank = 4;
    csh.d[0] = B;
    csh.d[1] = H;
    csh.d[2] = kv_len;
    csh.d[3] = hd;
    caches[ci].k = zeros2d(B * H * kv_len, hd);
    caches[ci].v = zeros2d(B * H * kv_len, hd);
    caches[ci].k.shape = csh;
    caches[ci].v.shape = csh;
    caches[ci].active = true;
    std::memcpy(caches[ci].k.data_.data(), k_concat.data_.data(),
                static_cast<size_t>(B * H * kv_len) * static_cast<size_t>(hd) * sizeof(float));
    std::memcpy(caches[ci].v.data_.data(), v_concat.data_.data(),
                static_cast<size_t>(B * H * kv_len) * static_cast<size_t>(hd) * sizeof(float));

    // residual (B=1, M = T)
    for (int64_t i = 0; i < M * d; ++i) {
        x.set_flat(i, x.at_flat(i) + proj_out.at_flat(i));
    }
}

}  // namespace

// ---- Config ----

bool Config::from_manifest(const mini::JsonValue& cfg_obj) {
    if (cfg_obj.t != mini::JsonValue::Type::Obj) return false;
    auto num = [&cfg_obj](const char* key, bool& ok) -> double {
        ok = false;
        const mini::JsonValue* v = mini::json_find(cfg_obj, key);
        if (!v) return 0.0;
        double d;
        if (!mini::json_try_num(*v, d)) return 0.0;
        ok = true;
        return d;
    };
    bool ok;
    double d;
    if ((d = num("vocab_size", ok), ok)) vocab_size = static_cast<int>(d);
    if ((d = num("n_layer", ok), ok)) n_layer = static_cast<int>(d);
    if ((d = num("n_head", ok), ok)) n_head = static_cast<int>(d);
    if ((d = num("d_model", ok), ok)) d_model = static_cast<int>(d);
    if ((d = num("block", ok), ok)) block = static_cast<int>(d);
    if ((d = num("d_ff", ok), ok)) d_ff = static_cast<int>(d);
    if ((d = num("rope_theta", ok), ok)) rope_theta = d;
    {
        const mini::JsonValue* v = mini::json_find(cfg_obj, "tie_embeddings");
        if (v) tie_embeddings = v->b;
    }
    if ((d = num("use_pool", ok), ok)) use_pool = static_cast<bool>(d > 0);
    if ((d = num("pool_experts", ok), ok)) pool_experts = static_cast<int>(d);
    if ((d = num("pool_d_ff", ok), ok)) pool_d_ff = static_cast<int>(d);
    if ((d = num("pool_depth", ok), ok)) pool_depth = static_cast<int>(d);
    if ((d = num("pool_top_k", ok), ok)) pool_top_k = static_cast<int>(d);
    if ((d = num("pool_capacity_factor", ok), ok)) pool_capacity_factor = d;
    if ((d = num("pool_max", ok), ok)) pool_max = static_cast<int>(d);
    if ((d = num("pool_aux", ok), ok)) pool_aux = d;
    if ((d = num("n_prelude", ok), ok)) n_prelude = static_cast<int>(d);
    if ((d = num("n_recur", ok), ok)) n_recur = static_cast<int>(d);
    if ((d = num("n_coda", ok), ok)) n_coda = static_cast<int>(d);
    if ((d = num("max_steps", ok), ok)) max_steps = static_cast<int>(d);
    if ((d = num("min_steps", ok), ok)) min_steps = static_cast<int>(d);
    if ((d = num("train_steps_mean", ok), ok)) train_steps_mean = d;
    if ((d = num("bptt_window", ok), ok)) bptt_window = static_cast<int>(d);
    if ((d = num("ponder_beta", ok), ok)) ponder_beta = d;
    if ((d = num("halt_prior", ok), ok)) halt_prior = d;
    if ((d = num("halt_thresh", ok), ok)) halt_thresh = d;
    if ((d = num("pool_resident", ok), ok)) pool_resident = static_cast<int>(d);
    return true;
}

// ---- Coder ----

Coder::Coder(const Config& cfg) : cfg_(cfg), n_slots_(cfg.n_prelude +
             cfg.max_steps * (cfg.n_recur + cfg.n_coda)) {
    const int d = cfg_.d_model;
    // one distinct weight set per (prelude + recur + coda) block-application
    // identity; weights are shared across the max_steps applications per step.
    const int n_blocks = cfg_.n_prelude + cfg_.n_recur + cfg_.n_coda;
    blocks_.resize(static_cast<size_t>(n_blocks));
    model::build_rope(cfg_.block, d / cfg_.n_head, cfg_.rope_theta, rope_cos_, rope_sin_);
}

bool Coder::loaded() const { return loaded_; }

const Config& Coder::cfg() const { return cfg_; }

minagi::paged::PagedPool* Coder::pool() { return pool_; }
const minagi::paged::PagedPool* Coder::pool() const { return pool_; }
void Coder::set_pool(minagi::paged::PagedPool* p) { pool_ = p; }

void Coder::random_init(unsigned seed) {
    PcgRng rng(static_cast<uint64_t>(seed));
    const int d = cfg_.d_model;
    const int V = cfg_.vocab_size;
    const int dff = cfg_.d_ff;
    const int n_blocks = cfg_.n_prelude + cfg_.n_recur + cfg_.n_coda;

    tok_emb_ = init_normal(shape2(V, d), rng, 0.0f, 0.02f);
    adapter_w_ = init_eye(d, 2 * d);
    ln_f_w_ = init_const(shape1(d), 1.0f);
    halt_w_ = init_normal(shape2(1, d), rng, 0.0f, 0.01f);
    halt_b_ = init_const(shape1(1), -2.0f);

    blocks_.resize(static_cast<size_t>(n_blocks));
    for (int i = 0; i < n_blocks; ++i) {
        auto& b = blocks_[i];
        b.ln1 = init_const(shape1(d), 1.0f);
        b.qkv = init_normal(shape2(3 * d, d), rng, 0.0f, 0.02f / std::sqrt(static_cast<float>(d)));
        b.proj = init_const(shape2(d, d), 0.0f);
        b.ln2 = init_const(shape1(d), 1.0f);
        b.w1 = init_normal(shape2(dff, d), rng, 0.0f, 0.02f / std::sqrt(static_cast<float>(dff)));
        b.w3 = init_normal(shape2(dff, d), rng, 0.0f, 0.02f / std::sqrt(static_cast<float>(dff)));
        b.w2 = init_normal(shape2(d, dff), rng, 0.0f, 0.02f / std::sqrt(static_cast<float>(dff)));
        b.router = init_const(shape2(cfg_.pool_experts, d), 0.0f);
        b.depth_emb = init_const(shape1(d), 0.0f);
    }
    loaded_ = true;
}

mt::Tensor* Coder::get_param(const std::string& name) {
    if (name == "tok_emb.weight") return &tok_emb_;
    if (name == "ln_f.weight") return &ln_f_w_;
    if (name == "halt.weight") return &halt_w_;
    if (name == "halt.bias") return &halt_b_;
    if (name == "adapter.weight") return &adapter_w_;

    for (int i = 0; i < static_cast<int>(blocks_.size()); ++i) {
        const auto& b = blocks_[i];
        std::string prefix;
        if (i == 0) prefix = "prelude.0.";
        else prefix = "recur." + std::to_string(i - cfg_.n_prelude) + ".";

        if (name == prefix + "ln1.weight") return const_cast<mt::Tensor*>(&b.ln1);
        if (name == prefix + "attn.qkv.weight") return const_cast<mt::Tensor*>(&b.qkv);
        if (name == prefix + "attn.proj.weight") return const_cast<mt::Tensor*>(&b.proj);
        if (name == prefix + "ln2.weight") return const_cast<mt::Tensor*>(&b.ln2);
        if (name == prefix + "mlp.w1.weight") return const_cast<mt::Tensor*>(&b.w1);
        if (name == prefix + "mlp.w3.weight") return const_cast<mt::Tensor*>(&b.w3);
        if (name == prefix + "mlp.w2.weight") return const_cast<mt::Tensor*>(&b.w2);
        if (name == prefix + "mlp.router.weight") return const_cast<mt::Tensor*>(&b.router);
        if (name == prefix + "mlp.depth_emb") return const_cast<mt::Tensor*>(&b.depth_emb);
    }
    return nullptr;
}

const mt::Tensor* Coder::get_param(const std::string& name) const {
    return const_cast<Coder*>(this)->get_param(name);
}

int Coder::n_slots() const { return n_slots_; }

std::vector<model::KVCache> Coder::empty_caches() const {
    std::vector<model::KVCache> c(static_cast<size_t>(n_slots_));
    return c;
}

bool Coder::load_weights(std::vector<std::pair<std::string, mt::Tensor>>& sd,
                         paged::PagedPool& pool) {
    auto find = [&](const std::string& key) -> const mt::Tensor* {
        for (auto& kv : sd) if (kv.first == key) return &kv.second;
        return nullptr;
    };
    auto get2 = [&](const std::string& key, int64_t r0, int64_t r1) -> mt::Tensor {
        const mt::Tensor* t = find(key);
        if (!t) return mt::Tensor();
        if (t->dtype != mt::DType::FP32) return mt::to_fp32(*t);
        if (t->shape.rank != 2 || t->shape.d[0] != r0 || t->shape.d[1] != r1)
            return mt::Tensor();
        return *t;
    };
    auto get1 = [&](const std::string& key, int64_t r0) -> mt::Tensor {
        const mt::Tensor* t = find(key);
        if (!t) return mt::Tensor();
        if (t->dtype != mt::DType::FP32) return mt::to_fp32(*t);
        if (t->shape.rank != 1 || t->shape.d[0] != r0) return mt::Tensor();
        return *t;
    };

    const int d = cfg_.d_model;
    const int V = cfg_.vocab_size;
    const int dff = cfg_.d_ff;

    tok_emb_ = get2("tok_emb.weight", V, d);
    if (tok_emb_.shape.rank == 0) return false;

    // prelude block (dense mlp)
    auto& pre = blocks_[0];
    pre.ln1 = get1("prelude.0.ln1.weight", d);
    pre.qkv = get2("prelude.0.attn.qkv.weight", 3 * d, d);
    pre.proj = get2("prelude.0.attn.proj.weight", d, d);
    pre.ln2 = get1("prelude.0.ln2.weight", d);
    pre.w1 = get2("prelude.0.mlp.w1.weight", dff, d);
    pre.w3 = get2("prelude.0.mlp.w3.weight", dff, d);
    pre.w2 = get2("prelude.0.mlp.w2.weight", d, dff);
    if (pre.ln1.shape.rank == 0 || pre.qkv.shape.rank == 0 || pre.proj.shape.rank == 0 ||
        pre.ln2.shape.rank == 0 || pre.w1.shape.rank == 0 || pre.w3.shape.rank == 0 ||
        pre.w2.shape.rank == 0)
        return false;

    // recurrent site blocks -- pooled mlp (router/depth_emb) when use_pool is
    // set, otherwise a dense SwiGLU (w1/w3/w2) mirroring the prelude block.
    for (int i = 0; i < cfg_.n_recur; ++i) {
        std::string b = "recur." + std::to_string(i) + ".";
        auto& rb = blocks_[cfg_.n_prelude + i];
        rb.ln1 = get1(b + "ln1.weight", d);
        rb.qkv = get2(b + "attn.qkv.weight", 3 * d, d);
        rb.proj = get2(b + "attn.proj.weight", d, d);
        rb.ln2 = get1(b + "ln2.weight", d);
        if (cfg_.use_pool) {
            rb.router = get2(b + "mlp.router.weight", cfg_.pool_experts, d);
            rb.depth_emb = get1(b + "mlp.depth_emb", d);
        } else {
            rb.w1 = get2(b + "mlp.w1.weight", dff, d);
            rb.w3 = get2(b + "mlp.w3.weight", dff, d);
            rb.w2 = get2(b + "mlp.w2.weight", d, dff);
        }
        if (rb.ln1.shape.rank == 0 || rb.qkv.shape.rank == 0 || rb.proj.shape.rank == 0 ||
            rb.ln2.shape.rank == 0)
            return false;
        if (cfg_.use_pool &&
            (rb.router.shape.rank == 0 || rb.depth_emb.shape.rank == 0))
            return false;
        if (!cfg_.use_pool &&
            (rb.w1.shape.rank == 0 || rb.w3.shape.rank == 0 || rb.w2.shape.rank == 0))
            return false;
    }

    adapter_w_ = get2("adapter.weight", d, 2 * d);
    if (adapter_w_.shape.rank == 0) return false;
    ln_f_w_ = get1("ln_f.weight", d);
    if (ln_f_w_.shape.rank == 0) return false;
    halt_w_ = get2("halt.weight", 1, d);
    halt_b_ = get1("halt.bias", 1);
    if (halt_w_.shape.rank == 0 || halt_b_.shape.rank == 0) return false;
    // head tied to tok_emb; only load tok_emb (already did).

    // pool wiring: gate + segment_router (router.weight/depth_emb live per-site,
    // loaded above into the recurrent blocks).
    if (cfg_.use_pool) {
        pool_ = &pool;
        const mt::Tensor* g = find("pool.gate");
        if (!g || g->dtype != mt::DType::FP32 || g->shape.rank != 1 ||
            g->shape.d[0] != cfg_.pool_experts)
            return false;
        pool.set_gate(*g);
        const mt::Tensor* sr = find("pool.segment_router.weight");
        if (!sr || sr->dtype != mt::DType::FP32 || sr->shape.rank != 2 ||
            sr->shape.d[0] != cfg_.pool_experts || sr->shape.d[1] != d)
            return false;
        pool.set_segment_router(*sr);
    }
    loaded_ = true;
    return true;
}

StepOut Coder::forward(const mt::Tensor& idx,
                       std::vector<model::KVCache>& caches,
                       int64_t pos_offset) {
    const int d = cfg_.d_model;
    const int H = cfg_.n_head;
    const int hd = d / H;
    const int V = cfg_.vocab_size;

    if (idx.dtype != mt::DType::FP32 && idx.dtype != mt::DType::BF16) {
        return StepOut{make_rank0(), make_rank0(), make_rank0()};
    }
    if (idx.shape.rank != 2 || idx.shape.d[0] != 1 || idx.shape.d[1] <= 0) {
        return StepOut{make_rank0(), make_rank0(), make_rank0()};
    }
    const int64_t B = 1;
    const int64_t T = idx.shape.d[1];

    // Section 6: reading past the rotary tables -> empty StepOut (no crash).
    if (pos_offset + T > cfg_.block) {
        return StepOut{make_rank0(), make_rank0(), make_rank0()};
    }
    if (static_cast<int>(caches.size()) < static_cast<size_t>(n_slots_)) {
        return StepOut{make_rank0(), make_rank0(), make_rank0()};
    }

    // x = tok_emb(idx) [M, d]
    const int64_t M = B * T;
    mt::Tensor x;
    {
        const int64_t Vrows = tok_emb_.shape.d[0];
        const float* ep = tok_emb_.ptr<float>();
        x = zeros2d(M, d);
        float* xp = x.ptr<float>();
        for (int64_t bt = 0; bt < M; ++bt) {
            int32_t id = static_cast<int32_t>(idx.at_flat(bt));
            if (id >= 0 && id < Vrows) {
                std::memcpy(xp + bt * d, ep + id * d, static_cast<size_t>(d) * sizeof(float));
            }
         }
    }

    // cos/sin positions pos_offset..pos_offset+T-1
    mt::Tensor cos = zeros2d(T, hd / 2);
    mt::Tensor sin = zeros2d(T, hd / 2);
    {
        const float* cp = rope_cos_.ptr<float>();
        const float* sp = rope_sin_.ptr<float>();
        float* cpo = cos.ptr<float>();
        float* spo = sin.ptr<float>();
        for (int64_t t = 0; t < T; ++t) {
            const int64_t row = pos_offset + t;
            std::memcpy(cpo + t * (hd / 2), cp + row * (hd / 2),
                        static_cast<size_t>(hd / 2) * sizeof(float));
            std::memcpy(spo + t * (hd / 2), sp + row * (hd / 2),
                        static_cast<size_t>(hd / 2) * sizeof(float));
        }
    }

    // prelude loop
    int ci = 0;
    {
        const Coder::Block& b = blocks_[0];
        // prelude block: attention (cache-aware) residual + dense SwiGLU residual.
        attn_residual(x, b.ln1, b.qkv, b.proj, caches, ci, T, H, hd, d, pos_offset, cos, sin);
        ++ci;
        mt::Tensor xn2 = mt::rms_norm(x, b.ln2, 1e-6f);
        mt::Tensor h1 = mt::matmul(xn2, transpose_2d(b.w1));
        mt::Tensor h3 = mt::matmul(xn2, transpose_2d(b.w3));
        mt::Tensor g = zeros2d(M, cfg_.d_ff);
        float* gp = g.ptr<float>();
        for (int64_t i = 0; i < M * cfg_.d_ff; ++i) {
            gp[i] = silu1(h1.atf(i)) * h3.atf(i);
        }
        mt::Tensor y = mt::matmul(g, transpose_2d(b.w2));
        for (int64_t i = 0; i < M * d; ++i) {
            x.set_flat(i, x.at_flat(i) + y.at_flat(i));
        }
    }

    // observe: pool.summary = mean over dims (0,1) of prelude output x.
    if (pool_ && cfg_.use_pool) {
        mt::Tensor obs;
        mt::Shape os;
        os.rank = 1;
        os.d[0] = d;
        obs = zeros2d(M, d);  // temporary reuse; we compute per-column mean
        // compute mean over M rows for each column
        mt::Tensor mean = mt::make(os, mt::DType::FP32, 0.0f);
        float* mp = mean.ptr<float>();
        const float* xp = x.ptr<float>();
        for (int64_t col = 0; col < d; ++col) {
            double acc = 0.0;
            for (int64_t t = 0; t < M; ++t) {
                acc += static_cast<double>(xp[static_cast<size_t>(t) * d + col]);
            }
            mp[col] = static_cast<float>(acc / static_cast<double>(M));
        }
        (void)obs;
        pool_->observe(mean);
    }

    // recurrence
    mt::Tensor h = zeros2d(M, d);            // h = zeros_like(x)
    std::vector<float> cum(static_cast<size_t>(M), 1.0f);
    std::vector<char> halted(static_cast<size_t>(M), 0);
    mt::Tensor halted_logits;                // [M, V]
    std::vector<int> steps_used(static_cast<size_t>(M), 1);
    std::vector<float> halt_p(static_cast<size_t>(M), 0.0f);

    const mt::Tensor& head_w = tok_emb_;     // tied
    const mt::Tensor& halt_w = halt_w_;      // [1, d]
    const mt::Tensor& halt_b = halt_b_;      // [1]
    const float hb = halt_b.atf(0);
    const float* hw = halt_w.ptr<float>();

    const int n_steps = cfg_.max_steps;
    const int min_steps = cfg_.min_steps;
    const float thresh = static_cast<float>(cfg_.halt_thresh);
    const int n_prelude = cfg_.n_prelude;
    const int n_recur = cfg_.n_recur;

    for (int n = 0; n < n_steps; ++n) {
        // h = adapter(cat([h, x], last dim))  -> [M, 2d] @ adapter_w^T -> [M, d]
        {
            mt::Tensor hin = zeros2d(M, 2 * d);
            float* hp = hin.ptr<float>();
            const float* hp_src = h.ptr<float>();
            const float* xp_src = x.ptr<float>();
            for (int64_t r = 0; r < M; ++r) {
                for (int64_t c = 0; c < d; ++c) {
                    hp[r * (2 * d) + c] = hp_src[r * d + c];       // h row
                    hp[r * (2 * d) + d + c] = xp_src[r * d + c];   // x row
                }
            }
            // h = hin @ adapter_w^T  ; adapter_w [d, 2d] -> transpose [2d, d]
            mt::Tensor hnew = mt::matmul(hin, transpose_2d(adapter_w_));
            std::memcpy(h.data_.data(), hnew.data_.data(),
                        static_cast<size_t>(M) * static_cast<size_t>(d) * sizeof(float));
        }
        // recurrent blocks (pooled mlp)
        for (int r = 0; r < n_recur; ++r) {
            const Coder::Block& b = blocks_[n_prelude + r];
            attn_residual(h, b.ln1, b.qkv, b.proj, caches, ci, T, H, hd, d,
                          pos_offset, cos, sin);
             ++ci;
            mt::Tensor hn2 = mt::rms_norm(h, b.ln2, 1e-6f);
            mt::Tensor mlp_out;
            if (cfg_.use_pool) {
                mlp_out = pool_mlp_forward(*pool_, hn2, b.router, b.depth_emb,
                                           cfg_.pool_top_k,
                                           cfg_.pool_capacity_factor, nullptr);
            } else {
                // dense SwiGLU at the recur site (no pool; mirrors the prelude)
                mt::Tensor h1 = mt::matmul(hn2, transpose_2d(b.w1));
                mt::Tensor h3 = mt::matmul(hn2, transpose_2d(b.w3));
                mt::Tensor g = zeros2d(M, cfg_.d_ff);
                float* gp = g.ptr<float>();
                for (int64_t i = 0; i < M * cfg_.d_ff; ++i) {
                    gp[i] = silu1(h1.atf(i)) * h3.atf(i);
                }
                mlp_out = mt::matmul(g, transpose_2d(b.w2));
            }
            const float* mp = mlp_out.ptr<float>();
            float* hp = h.ptr<float>();
            for (int64_t i = 0; i < M * d; ++i) {
                hp[i] = hp[i] + mp[i];
            }
        }
        // coda blocks (n_coda; zero here)
        // (omitted: n_coda == 0 in the golden; a CODA block would be a pooled mlp
        // site like RECUR, with its own cache slot.)

        mt::Tensor yf = mt::rms_norm(h, ln_f_w_, 1e-6f);          // [M, d]
        mt::Tensor logits_n = mt::matmul(yf, transpose_2d(head_w));  // [M, V]
        // lam = sigmoid(halt(yf)): halt linear [1,d], bias [1]
        std::vector<float> lam(static_cast<size_t>(M));
        for (int64_t m = 0; m < M; ++m) {
            const float* yrow = yf.ptr<float>() + m * d;
            double dot = 0.0;
            for (int64_t dd = 0; dd < d; ++dd) {
                dot += static_cast<double>(hw[dd]) * static_cast<double>(yrow[dd]);
            }
            dot += hb;
            lam[m] = 1.0f / (1.0f + std::exp(static_cast<float>(-dot)));
        }
        if (n == n_steps - 1) {
            for (int64_t m = 0; m < M; ++m) lam[m] = 1.0f;
        } else if (n < min_steps - 1) {
            for (int64_t m = 0; m < M; ++m) lam[m] = 0.0f;
        }

        for (int64_t m = 0; m < M; ++m) {
            float old_cum = cum[m];
            cum[m] = old_cum * (1.0f - lam[m]);  // cum = cum * (1 - lam)
            if (halted_logits.shape.rank == 0) {
                halted_logits = mt::make(logits_n.shape, mt::DType::FP32, 0.0f);
                for (int64_t i = 0; i < M * V; ++i) halted_logits.set_flat(i, logits_n.at_flat(i));
            }
            // newly halted: (1 - cum) >= thresh and not already halted
            float one_minus = 1.0f - cum[m];
            if (!halted[m] && one_minus >= thresh) {
                // copy this step's logits row for position m
                for (int64_t v = 0; v < V; ++v) {
                    halted_logits.set_flat(m * V + v, logits_n.at_flat(m * V + v));
                }
                steps_used[m] = n + 1;
                halt_p[m] = one_minus;
                halted[m] = 1;
            }
        }
        // (all steps run regardless of halting -- quality bar (c))
    }

    // Build StepOut: logits [1,T,V] = halted_logits reshaped, halt_row [1,T], halt_p [1,T].
    StepOut out;
    {
        mt::Shape ls;
        ls.rank = 3;
        ls.d[0] = 1;
        ls.d[1] = T;
        ls.d[2] = V;
        out.logits = mt::make(ls, mt::DType::FP32, 0.0f);
        if (halted_logits.shape.rank == 2) {
            for (int64_t i = 0; i < M * V; ++i)
                out.logits.set_flat(i, halted_logits.at_flat(i));
        }
        mt::Shape hs;
        hs.rank = 2;
        hs.d[0] = 1;
        hs.d[1] = T;
        out.halt_row = mt::make(hs, mt::DType::FP32, 0.0f);
        out.halt_p = mt::make(hs, mt::DType::FP32, 0.0f);
        float* hr = out.halt_row.ptr<float>();
        float* hp2 = out.halt_p.ptr<float>();
        for (int64_t t = 0; t < T; ++t) {
            hr[t] = static_cast<float>(steps_used[static_cast<size_t>(t)]);
            hp2[t] = halt_p[static_cast<size_t>(t)];
        }
    }
    return out;
}

int Coder::begin_segment() {
    if (!pool_ || !cfg_.use_pool) return 0;
    return pool_->swap_to(pool_->choose());
}

void Coder::end_segment(const mt::Tensor& x) {
    if (!pool_ || !cfg_.use_pool) return;
    const int64_t M = x.shape.d[0];
    const int64_t d = x.shape.d[1];
    mt::Tensor mean;
    mt::Shape os;
    os.rank = 1;
    os.d[0] = d;
    mean = mt::make(os, mt::DType::FP32, 0.0f);
    const float* xp = x.ptr<float>();
    float* mp = mean.ptr<float>();
    for (int64_t col = 0; col < d; ++col) {
        double acc = 0.0;
        for (int64_t t = 0; t < M; ++t)
            acc += static_cast<double>(xp[static_cast<size_t>(t) * d + col]);
        mp[col] = static_cast<float>(acc / static_cast<double>(M));
    }
    pool_->observe(mean);
}

// ---- minagi::load ----

bool load(const std::string& wdir, Config& cfg_out,
          paged::PagedPool& pool_out, Coder& coder_out) {
    ::store::Manifest man;
    std::map<std::string, mininpz::Array> raw;
    if (!::store::load(wdir, man, raw)) return false;
    if (!man.paged) return false;

    // cfg object: prefer the parsed JSON (typed); fall back to string map.
    mini::JsonValue cfg_obj;
    if (man.cfg_obj.t == mini::JsonValue::Type::Obj) {
        cfg_obj = man.cfg_obj;
    } else {
        std::map<std::string, mini::JsonValue> o;
        for (const auto& kv : man.cfg) {
            o[kv.first] = mini::JsonValue::make_str(kv.second);
        }
        cfg_obj = mini::JsonValue::make_obj(o);
    }
    if (!cfg_out.from_manifest(cfg_obj)) return false;

    // read_only / pool_ram come from the manifest top-level (defaults true / 4).
    const bool read_only = man.read_only;
    const int pool_ram = man.pool_ram;

    // sd: flat array -> Tensor (fp32). Experts' w1/w3/w2 come from the per-expert
    // files already flattened into the state by store::load.
    std::vector<std::pair<std::string, mt::Tensor>> sd;
    for (const auto& kv : raw) {
        sd.emplace_back(kv.first, paged::array_to_tensor(kv.second));
    }

    const int n = cfg_out.pool_experts;
    const int d = cfg_out.d_model;
    const int dff = cfg_out.pool_d_ff;
    mt::Shape oneshape;
    oneshape.rank = 1;
    oneshape.d[0] = n;
    mt::Tensor ones = mt::make(oneshape, mt::DType::FP32, 1.0f);
    paged::PagedPool fresh(n, d, dff, cfg_out.pool_resident, pool_ram,
                           cfg_out.explore, cfg_out.margin, cfg_out.dwell,
                           read_only, ones);
    fresh.set_experts_dir(::store::expert_dir(wdir));
    // NOTE: minagi::load deliberately does NOT load_telemetry (end state).

    Coder coder(cfg_out);
    if (!coder.load_weights(sd, fresh)) {
        return false;
    }

    pool_out = std::move(fresh);
    coder_out = std::move(coder);
    coder_out.set_pool(&pool_out);
    return true;
}

}  // namespace minagi
