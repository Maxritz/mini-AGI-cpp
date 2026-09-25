// src/backward.cpp
// CPU-side backward pass for mini-AGI training.
#include "backward.hpp"
#include "pool.hpp"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <limits>
#include <iostream>

namespace minagi::backward {

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

inline float silu_deriv(float x) {
    float sig = 1.0f / (1.0f + std::exp(-x));
    return sig * (1.0f + x * (1.0f - sig));
}

float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

mt::Shape shape0() { mt::Shape s; s.rank = 0; return s; }
mt::Shape shape1(int64_t d0) { mt::Shape s; s.rank = 1; s.d[0] = d0; return s; }
mt::Shape shape2(int64_t r0, int64_t r1) { mt::Shape s; s.rank = 2; s.d[0] = r0; s.d[1] = r1; return s; }

mt::Tensor zeros2d(int64_t r, int64_t c) {
    return mt::make(shape2(r, c), mt::DType::FP32, 0.0f);
}

mt::Tensor transpose_2d(const mt::Tensor& a) {
    const int64_t M = a.shape.d[0];
    const int64_t N = a.shape.d[1];
    mt::Tensor out = zeros2d(N, M);
    for (int64_t i = 0; i < M; ++i)
        for (int64_t j = 0; j < N; ++j)
            out.set_flat(j * M + i, a.at_flat(i * N + j));
    return out;
}

}  // namespace

// ---- Forward with cache ----

BackwardCache forward_with_cache(const Config& cfg, Coder& coder,
                                 const mt::Tensor& idx,
                                 const mt::Tensor& targets,
                                 std::vector<model::KVCache>& caches,
                                 int64_t pos_offset) {

    const int d = cfg.d_model;
    const int H = cfg.n_head;
    const int hd = d / H;
    const int V = cfg.vocab_size;
    const int T = static_cast<int>(idx.shape.d[1]);
    const int M = T;
    const int dff = cfg.d_ff;
    const int half = hd / 2;
    const int n_steps = cfg.max_steps;
    const int n_recur = cfg.n_recur;
    const int n_prelude = cfg.n_prelude;

    BackwardCache cache;
    cache.T = T; cache.d = d; cache.V = V; cache.M = M;
    cache.n_steps = n_steps; cache.n_recur = n_recur;
    cache.H = H; cache.hd = hd; cache.half = half;

    // Store token ids
    cache.idx_flat.resize(static_cast<size_t>(M));
    for (int64_t i = 0; i < M; ++i) {
        cache.idx_flat[static_cast<size_t>(i)] = static_cast<int32_t>(idx.at_flat(i));
    }

    cache.target_flat.resize(static_cast<size_t>(M));
    for (int64_t i = 0; i < M; ++i) {
        cache.target_flat[static_cast<size_t>(i)] = static_cast<int32_t>(targets.at_flat(i));
    }

    // cos/sin
    mt::Tensor cos = zeros2d(T, half);
    mt::Tensor sin = zeros2d(T, half);
    {
        const float* cp = coder.rope_cos().ptr<float>();
        const float* sp = coder.rope_sin().ptr<float>();
        float* cpo = cos.ptr<float>();
        float* spo = sin.ptr<float>();
        for (int64_t t = 0; t < T; ++t) {
            const int64_t row = pos_offset + t;
            std::memcpy(cpo + t * half, cp + row * half, static_cast<size_t>(half) * sizeof(float));
            std::memcpy(spo + t * half, sp + row * half, static_cast<size_t>(half) * sizeof(float));
        }
    }
    cache.cos = cos;
    cache.sin = sin;

    // x = tok_emb(idx) [M, d]
    mt::Tensor x;
    {
        const int64_t Vrows = coder.tok_emb().shape.d[0];
        const float* ep = coder.tok_emb().ptr<float>();
        x = zeros2d(M, d);
        float* xp = x.ptr<float>();
        for (int64_t bt = 0; bt < M; ++bt) {
            int32_t id = static_cast<int32_t>(idx.at_flat(bt));
            if (id >= 0 && id < Vrows) {
                std::memcpy(xp + bt * d, ep + id * d, static_cast<size_t>(d) * sizeof(float));
            }
        }
    }
    cache.pre_x = x;

    // Prelude block
    int ci = 0;
    const auto& pre = coder.blocks()[0];

    // Attention: rn1 = rms_norm(x, ln1), qkv = rn1 @ qkv^T
    mt::Tensor rn1 = mt::rms_norm(x, pre.ln1, 1e-6f);
    cache.pre_xn1 = rn1;
    mt::Tensor qkv_t = mt::matmul(rn1, transpose_2d(pre.qkv));
    cache.pre_qkv = qkv_t;
    // Split qkv
    mt::Tensor q = zeros2d(M, d), k = zeros2d(M, d), v = zeros2d(M, d);
    for (int64_t bt = 0; bt < M; ++bt) {
        for (int64_t h = 0; h < H; ++h) {
            for (int64_t dd = 0; dd < hd; ++dd) {
                const int64_t src = bt * d + h * hd + dd;
                const int64_t s3 = bt * 3 * d;
                q.set_flat(src, qkv_t.at_flat(s3 + h * hd + dd));
                k.set_flat(src, qkv_t.at_flat(s3 + d + h * hd + dd));
                v.set_flat(src, qkv_t.at_flat(s3 + 2 * d + h * hd + dd));
            }
        }
    }
    // Reshape to head-major [H*T, hd]
    mt::Tensor qh = zeros2d(H * T, hd);
    mt::Tensor kh = zeros2d(H * T, hd);
    mt::Tensor vh = zeros2d(H * T, hd);
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t h = 0; h < H; ++h) {
            for (int64_t dd = 0; dd < hd; ++dd) {
                const int64_t src = t * d + h * hd + dd;
                const int64_t dst = (h * T + t) * hd + dd;
                qh.set_flat(dst, q.at_flat(src));
                kh.set_flat(dst, k.at_flat(src));
                vh.set_flat(dst, v.at_flat(src));
            }
        }
    }
    // RoPE
    auto rot = [&](mt::Tensor& tt) {
        for (int64_t h = 0; h < H; ++h) {
            for (int64_t t = 0; t < T; ++t) {
                const int64_t row = t;
                for (int64_t i = 0; i < half; ++i) {
                    const float c = cos.atf(row * half + i);
                    const float s = sin.atf(row * half + i);
                    const int64_t idx0 = (h * T + t) * hd + 2 * i;
                    const int64_t idx1 = (h * T + t) * hd + 2 * i + 1;
                    const float x1 = tt.atf(idx0);
                    const float x2 = tt.atf(idx1);
                    tt.set_flat(idx0, x1 * c - x2 * s);
                    tt.set_flat(idx1, x1 * s + x2 * c);
                }
            }
        }
    };
    rot(qh);
    rot(kh);
    cache.pre_qh = qh;

    // KV cache handling
    int64_t kv_len = T;
    mt::Tensor k_concat = kh;
    mt::Tensor v_concat = vh;
    if (caches[ci].active && caches[ci].k.shape.d[2] > 0) {
        const int64_t cached_T = caches[ci].k.shape.d[2];
        kv_len = cached_T + T;
        k_concat = zeros2d(H * kv_len, hd);
        v_concat = zeros2d(H * kv_len, hd);
        for (int64_t h = 0; h < H; ++h) {
            for (int64_t cc = 0; cc < cached_T; ++cc) {
                for (int64_t cd = 0; cd < hd; ++cd) {
                    k_concat.set_flat((h * kv_len + cc) * hd + cd,
                                      caches[ci].k.at_flat((h * cached_T + cc) * hd + cd));
                    v_concat.set_flat((h * kv_len + cc) * hd + cd,
                                      caches[ci].v.at_flat((h * cached_T + cc) * hd + cd));
                }
            }
            for (int64_t t = 0; t < T; ++t) {
                for (int64_t cd = 0; cd < hd; ++cd) {
                    k_concat.set_flat((h * kv_len + cached_T + t) * hd + cd,
                                      kh.at_flat((h * T + t) * hd + cd));
                    v_concat.set_flat((h * kv_len + cached_T + t) * hd + cd,
                                      vh.at_flat((h * T + t) * hd + cd));
                }
            }
        }
    }

    // Scores [H*T, kv_len]
    mt::Tensor scores = zeros2d(H * T, kv_len);
    const float inv_sqrt = 1.0f / std::sqrt(static_cast<float>(hd));
    const int64_t P = kv_len - T;
    float* sp = scores.ptr<float>();
    for (int64_t h = 0; h < H; ++h) {
        for (int64_t i = 0; i < T; ++i) {
            for (int64_t j = 0; j < kv_len; ++j) {
                double dot = 0.0;
                for (int64_t ddi = 0; ddi < hd; ++ddi) {
                    dot += static_cast<double>(qh.atf((h * T + i) * hd + ddi)) *
                           static_cast<double>(k_concat.atf((h * kv_len + j) * hd + ddi));
                }
                dot *= inv_sqrt;
                if (j > i + P) dot = -1e30f;
                sp[(h * T + i) * kv_len + j] = static_cast<float>(dot);
            }
        }
    }
    mt::Tensor sm = mt::softmax(scores, 1);
    // The keys/values the query actually attended (the concat, not the window)
    // plus the resulting weights. Cached positions belong to earlier chunks,
    // which are constant inputs here, so their gradients are not produced.
    cache.pre_kc = k_concat;
    cache.pre_vc = v_concat;
    cache.pre_sm = sm;
    // Attention output: out_h [H*T, hd] = sm @ v_concat
    mt::Tensor out_h = zeros2d(H * T, hd);
    {
        float* op = out_h.ptr<float>();
        const float* sp2 = sm.ptr<float>();
        const float* vp = v_concat.ptr<float>();
        for (int64_t h = 0; h < H; ++h) {
            for (int64_t t = 0; t < T; ++t) {
                for (int64_t dd = 0; dd < hd; ++dd) {
                    double acc = 0.0;
                    for (int64_t j = 0; j < kv_len; ++j) {
                        acc += static_cast<double>(sp2[(h * T + t) * kv_len + j]) *
                               static_cast<double>(vp[(h * kv_len + j) * hd + dd]);
                    }
                    op[(h * T + t) * hd + dd] = static_cast<float>(acc);
                }
            }
        }
    }
    // Back to [M, d]
    mt::Tensor out_flat = zeros2d(M, d);
    {
        float* fp = out_flat.ptr<float>();
        for (int64_t b = 0; b < 1; ++b) {
            for (int64_t t2 = 0; t2 < T; ++t2) {
                const int64_t bt = b * T + t2;
                for (int64_t h = 0; h < H; ++h) {
                    for (int64_t dd = 0; dd < hd; ++dd) {
                        fp[bt * d + h * hd + dd] = out_h.atf((h * T + t2) * hd + dd);
                    }
                }
            }
        }
    }
    // proj_out = out_flat @ proj^T
    mt::Tensor proj_out = mt::matmul(out_flat, transpose_2d(pre.proj));
    cache.pre_out_flat = out_flat;
    // Cache state
    cache.pre_x_after_attn = zeros2d(M, d);
    for (int64_t i = 0; i < M * d; ++i)
        cache.pre_x_after_attn.set_flat(i, x.at_flat(i) + proj_out.at_flat(i));

    // Write cache back
    mt::Tensor k_cache = zeros2d(H * kv_len, hd);
    mt::Tensor v_cache = zeros2d(H * kv_len, hd);
    std::memcpy(k_cache.data_.data(), k_concat.data_.data(),
                static_cast<size_t>(H * kv_len * hd) * sizeof(float));
    std::memcpy(v_cache.data_.data(), v_concat.data_.data(),
                static_cast<size_t>(H * kv_len * hd) * sizeof(float));
    mt::Shape csh; csh.rank = 4;
    csh.d[0] = 1; csh.d[1] = H; csh.d[2] = kv_len; csh.d[3] = hd;
    caches[ci].k = k_cache; caches[ci].k.shape = csh; caches[ci].active = true;
    caches[ci].v = v_cache; caches[ci].v.shape = csh; caches[ci].active = true;
    ++ci;

    // Dense MLP (prelude)
    mt::Tensor xn2 = mt::rms_norm(cache.pre_x_after_attn, pre.ln2, 1e-6f);
    cache.pre_xn2 = xn2;
    mt::Tensor h1 = mt::matmul(xn2, transpose_2d(pre.w1));
    mt::Tensor h3 = mt::matmul(xn2, transpose_2d(pre.w3));
    cache.pre_h1 = h1;
    cache.pre_h3 = h3;
    mt::Tensor g = zeros2d(M, dff);
    float* gp = g.ptr<float>();
    for (int64_t i = 0; i < M * dff; ++i)
        gp[i] = silu1(h1.atf(i)) * h3.atf(i);
    cache.pre_g = g;
    mt::Tensor y_mlp = mt::matmul(g, transpose_2d(pre.w2));
    cache.pre_out = zeros2d(M, d);
    for (int64_t i = 0; i < M * d; ++i)
        cache.pre_out.set_flat(i, cache.pre_x_after_attn.at_flat(i) + y_mlp.at_flat(i));

    // Observe: pool.summary = mean over M of pre_out
    if (coder.pool() && cfg.use_pool) {
        mt::Shape os; os.rank = 1; os.d[0] = d;
        mt::Tensor mean = mt::make(os, mt::DType::FP32, 0.0f);
        float* mp = mean.ptr<float>();
        const float* op = cache.pre_out.ptr<float>();
        for (int64_t col = 0; col < d; ++col) {
            double acc = 0.0;
            for (int64_t t = 0; t < M; ++t)
                acc += static_cast<double>(op[t * d + col]);
            mp[col] = static_cast<float>(acc / static_cast<double>(M));
        }
        coder.pool()->observe(mean);
    }

    // Recurrence
    mt::Tensor h = zeros2d(M, d);
    std::vector<float> cum(M, 1.0f);
    std::vector<char> halted(M, 0);
    mt::Tensor halted_logits;
    cache.steps_used.resize(M, 1);
    cache.cum_init = cum;

    const mt::Tensor& head_w = coder.tok_emb();
    const mt::Tensor& halt_w = coder.halt_w();
    const mt::Tensor& halt_b = coder.halt_b();
    const float hb = halt_b.atf(0);
    const float* hw = halt_w.ptr<float>();

    const int min_steps = cfg.min_steps;
    const float thresh = static_cast<float>(cfg.halt_thresh);

    cache.h_before_adapter.resize(n_steps);
    cache.h_after_adapter.resize(n_steps);
    cache.h_end_per_step.resize(n_steps);
    cache.yf_per_step.resize(n_steps);
    cache.logits_per_step.resize(n_steps);
    cache.lam_per_step.resize(n_steps);
    cache.rb_xn1.resize(n_steps, std::vector<mt::Tensor>(n_recur));
    cache.rb_qkv.resize(n_steps, std::vector<mt::Tensor>(n_recur));
    cache.rb_qh.resize(n_steps, std::vector<mt::Tensor>(n_recur));
    cache.rb_kc.resize(n_steps, std::vector<mt::Tensor>(n_recur));
    cache.rb_vc.resize(n_steps, std::vector<mt::Tensor>(n_recur));
    cache.rb_sm.resize(n_steps, std::vector<mt::Tensor>(n_recur));
    cache.rb_out_flat.resize(n_steps, std::vector<mt::Tensor>(n_recur));
    cache.rb_in.resize(n_steps, std::vector<mt::Tensor>(n_recur));
    cache.rb_after_attn.resize(n_steps, std::vector<mt::Tensor>(n_recur));
    cache.rb_h1.resize(n_steps, std::vector<mt::Tensor>(n_recur));
    cache.rb_h3.resize(n_steps, std::vector<mt::Tensor>(n_recur));
    cache.rb_g.resize(n_steps, std::vector<mt::Tensor>(n_recur));
    cache.rb_xn2.resize(n_steps, std::vector<mt::Tensor>(n_recur));
    cache.rb_pool_cache.resize(n_steps, std::vector<PoolBackwardCache>(n_recur));

    for (int n = 0; n < n_steps; ++n) {
        cache.h_before_adapter[static_cast<size_t>(n)] = h;

        // h = adapter(cat([h, x], last dim))
        {
            mt::Tensor hin = zeros2d(M, 2 * d);
            float* hp = hin.ptr<float>();
            const float* hp_src = h.ptr<float>();
            const float* xp_src = cache.pre_out.ptr<float>();
            for (int64_t r = 0; r < M; ++r) {
                for (int64_t c = 0; c < d; ++c) {
                    hp[r * (2 * d) + c] = hp_src[r * d + c];
                    hp[r * (2 * d) + d + c] = xp_src[r * d + c];
                }
            }
            mt::Tensor hnew = mt::matmul(hin, transpose_2d(coder.adapter_w()));
            h = hnew;
        }
        cache.h_after_adapter[static_cast<size_t>(n)] = h;

        // Recurrent blocks
        for (int r = 0; r < n_recur; ++r) {
            cache.rb_in[static_cast<size_t>(n)][static_cast<size_t>(r)] = h;
            const auto& b = coder.blocks()[n_prelude + r];

            // Attention
            mt::Tensor xn = mt::rms_norm(h, b.ln1, 1e-6f);
            cache.rb_xn1[static_cast<size_t>(n)][static_cast<size_t>(r)] = xn;
            mt::Tensor qkv_b = mt::matmul(xn, transpose_2d(b.qkv));
            cache.rb_qkv[static_cast<size_t>(n)][static_cast<size_t>(r)] = qkv_b;

            mt::Tensor qb = zeros2d(M, d), kb = zeros2d(M, d), vb = zeros2d(M, d);
            for (int64_t bt = 0; bt < M; ++bt) {
                for (int64_t hh = 0; hh < H; ++hh) {
                    for (int64_t dd = 0; dd < hd; ++dd) {
                        const int64_t src = bt * d + hh * hd + dd;
                        const int64_t s3 = bt * 3 * d;
                        qb.set_flat(src, qkv_b.at_flat(s3 + hh * hd + dd));
                        kb.set_flat(src, qkv_b.at_flat(s3 + d + hh * hd + dd));
                        vb.set_flat(src, qkv_b.at_flat(s3 + 2 * d + hh * hd + dd));
                    }
                }
            }
            mt::Tensor qhb = zeros2d(H * T, hd);
            mt::Tensor khb = zeros2d(H * T, hd);
            mt::Tensor vhb = zeros2d(H * T, hd);
            for (int64_t t2 = 0; t2 < T; ++t2) {
                for (int64_t hh = 0; hh < H; ++hh) {
                    for (int64_t dd = 0; dd < hd; ++dd) {
                        const int64_t src = t2 * d + hh * hd + dd;
                        const int64_t dst = (hh * T + t2) * hd + dd;
                        qhb.set_flat(dst, qb.at_flat(src));
                        khb.set_flat(dst, kb.at_flat(src));
                        vhb.set_flat(dst, vb.at_flat(src));
                    }
                }
            }
            cache.rb_qh[static_cast<size_t>(n)][static_cast<size_t>(r)] = qhb;
            for (int64_t hhh = 0; hhh < H; ++hhh) {
                for (int64_t t2 = 0; t2 < T; ++t2) {
                    const int64_t row = t2;
                    for (int64_t i = 0; i < half; ++i) {
                        const float c = cos.atf(row * half + i);
                        const float s = sin.atf(row * half + i);
                        const int64_t i0 = (hhh * T + t2) * hd + 2 * i;
                        const int64_t i1 = (hhh * T + t2) * hd + 2 * i + 1;
                        float x1 = qhb.atf(i0); float x2 = qhb.atf(i1);
                        qhb.set_flat(i0, x1 * c - x2 * s);
                        qhb.set_flat(i1, x1 * s + x2 * c);
                        x1 = khb.atf(i0); x2 = khb.atf(i1);
                        khb.set_flat(i0, x1 * c - x2 * s);
                        khb.set_flat(i1, x1 * s + x2 * c);
                    }
                }
            }
            cache.rb_qh[static_cast<size_t>(n)][static_cast<size_t>(r)] = qhb;

            // KV cache concat
            int64_t kv_len_b = T;
            mt::Tensor k_concat_b = khb;
            mt::Tensor v_concat_b = vhb;
            if (caches[ci].active && caches[ci].k.shape.d[2] > 0) {
                const int64_t cached_T = caches[ci].k.shape.d[2];
                kv_len_b = cached_T + T;
                k_concat_b = zeros2d(H * kv_len_b, hd);
                v_concat_b = zeros2d(H * kv_len_b, hd);
                for (int64_t hhh = 0; hhh < H; ++hhh) {
                    for (int64_t cc = 0; cc < cached_T; ++cc) {
                        for (int64_t cd = 0; cd < hd; ++cd) {
                            k_concat_b.set_flat((hhh * kv_len_b + cc) * hd + cd,
                                                caches[ci].k.at_flat((hhh * cached_T + cc) * hd + cd));
                            v_concat_b.set_flat((hhh * kv_len_b + cc) * hd + cd,
                                                caches[ci].v.at_flat((hhh * cached_T + cc) * hd + cd));
                        }
                    }
                    for (int64_t t2 = 0; t2 < T; ++t2) {
                        for (int64_t cd = 0; cd < hd; ++cd) {
                            k_concat_b.set_flat((hhh * kv_len_b + cached_T + t2) * hd + cd,
                                                khb.at_flat((hhh * T + t2) * hd + cd));
                            v_concat_b.set_flat((hhh * kv_len_b + cached_T + t2) * hd + cd,
                                                vhb.at_flat((hhh * T + t2) * hd + cd));
                        }
                    }
                }
            }

            // Scores
            mt::Tensor scores_b = zeros2d(H * T, kv_len_b);
            const int64_t P_b = kv_len_b - T;
            float* sp_b = scores_b.ptr<float>();
            for (int64_t hhh = 0; hhh < H; ++hhh) {
                for (int64_t i = 0; i < T; ++i) {
                    for (int64_t j = 0; j < kv_len_b; ++j) {
                        double dot = 0.0;
                        for (int64_t ddi = 0; ddi < hd; ++ddi) {
                            dot += static_cast<double>(qhb.atf((hhh * T + i) * hd + ddi)) *
                                   static_cast<double>(k_concat_b.atf((hhh * kv_len_b + j) * hd + ddi));
                        }
                        dot *= inv_sqrt;
                        if (j > i + P_b) dot = -1e30f;
                        sp_b[(hhh * T + i) * kv_len_b + j] = static_cast<float>(dot);
                    }
                }
            }
            mt::Tensor sm_b = mt::softmax(scores_b, 1);
            // Keys/values actually attended, plus the weights. Cached positions
            // belong to earlier chunks, which are constant inputs here.
            cache.rb_kc[static_cast<size_t>(n)][static_cast<size_t>(r)] = k_concat_b;
            cache.rb_vc[static_cast<size_t>(n)][static_cast<size_t>(r)] = v_concat_b;
            cache.rb_sm[static_cast<size_t>(n)][static_cast<size_t>(r)] = sm_b;

            // Attention output
            mt::Tensor out_h_b = zeros2d(H * T, hd);
            {
                float* op_b = out_h_b.ptr<float>();
                const float* sp2 = sm_b.ptr<float>();
                const float* vp = v_concat_b.ptr<float>();
                for (int64_t hhh = 0; hhh < H; ++hhh) {
                    for (int64_t t2 = 0; t2 < T; ++t2) {
                        for (int64_t dd = 0; dd < hd; ++dd) {
                            double acc = 0.0;
                            for (int64_t j = 0; j < kv_len_b; ++j) {
                                acc += static_cast<double>(sp2[(hhh * T + t2) * kv_len_b + j]) *
                                       static_cast<double>(vp[(hhh * kv_len_b + j) * hd + dd]);
                            }
                            op_b[(hhh * T + t2) * hd + dd] = static_cast<float>(acc);
                        }
                    }
                }
            }
            // Back to [M, d]
            mt::Tensor out_flat_b = zeros2d(M, d);
            {
                float* fp_b = out_flat_b.ptr<float>();
                for (int64_t t2 = 0; t2 < T; ++t2) {
                    for (int64_t hhh = 0; hhh < H; ++hhh) {
                        for (int64_t dd = 0; dd < hd; ++dd) {
                            fp_b[t2 * d + hhh * hd + dd] = out_h_b.atf((hhh * T + t2) * hd + dd);
                        }
                    }
                }
            }
            cache.rb_out_flat[static_cast<size_t>(n)][static_cast<size_t>(r)] = out_flat_b;

            // Update cache
            mt::Shape csh_b; csh_b.rank = 4;
            csh_b.d[0] = 1; csh_b.d[1] = H; csh_b.d[2] = kv_len_b; csh_b.d[3] = hd;
            mt::Tensor k_cache_b = zeros2d(H * kv_len_b, hd);
            mt::Tensor v_cache_b = zeros2d(H * kv_len_b, hd);
            std::memcpy(k_cache_b.data_.data(), k_concat_b.data_.data(),
                        static_cast<size_t>(H * kv_len_b * hd) * sizeof(float));
            std::memcpy(v_cache_b.data_.data(), v_concat_b.data_.data(),
                        static_cast<size_t>(H * kv_len_b * hd) * sizeof(float));
            caches[ci].k = k_cache_b; caches[ci].k.shape = csh_b; caches[ci].active = true;
            caches[ci].v = v_cache_b; caches[ci].v.shape = csh_b; caches[ci].active = true;
            ++ci;

            // h = h + proj_out_b (attention residual)
            mt::Tensor proj_out_b = mt::matmul(out_flat_b, transpose_2d(b.proj));
            for (int64_t i = 0; i < M * d; ++i)
                h.set_flat(i, h.at_flat(i) + proj_out_b.at_flat(i));
            cache.rb_after_attn[static_cast<size_t>(n)][static_cast<size_t>(r)] = h;

            // Pooled MLP
            mt::Tensor hn2 = mt::rms_norm(h, b.ln2, 1e-6f);
            cache.rb_xn2[static_cast<size_t>(n)][static_cast<size_t>(r)] = hn2;
            mt::Tensor mlp_out;
            if (coder.pool() && cfg.use_pool) {
            mlp_out = pool_mlp_forward(*coder.pool(), hn2, b.router, b.depth_emb,
                                       cfg.pool_top_k,
                                       cfg.pool_capacity_factor, nullptr,
                                       &cache.rb_pool_cache[static_cast<size_t>(n)][static_cast<size_t>(r)]);
            } else {
                // Fallback: dense MLP using w1/w3/w2
                mt::Tensor h1b = mt::matmul(hn2, transpose_2d(b.w1));
                mt::Tensor h3b = mt::matmul(hn2, transpose_2d(b.w3));
                mt::Tensor gb = zeros2d(M, dff);
                float* gp = gb.ptr<float>();
                for (int64_t i = 0; i < M * dff; ++i)
                    gp[i] = silu1(h1b.atf(i)) * h3b.atf(i);
                cache.rb_h1[static_cast<size_t>(n)][static_cast<size_t>(r)] = h1b;
                cache.rb_h3[static_cast<size_t>(n)][static_cast<size_t>(r)] = h3b;
                cache.rb_g[static_cast<size_t>(n)][static_cast<size_t>(r)] = gb;
                mlp_out = mt::matmul(gb, transpose_2d(b.w2));
            }
            const float* mp = mlp_out.ptr<float>();
            float* hp = h.ptr<float>();
            for (int64_t i = 0; i < M * d; ++i)
                hp[i] += mp[i];
        }

        cache.h_end_per_step[static_cast<size_t>(n)] = h;

        // ln_f, head, halt
        mt::Tensor yf = mt::rms_norm(h, coder.ln_f_w(), 1e-6f);
        cache.yf_per_step[static_cast<size_t>(n)] = yf;
        mt::Tensor logits_n = mt::matmul(yf, transpose_2d(head_w));
        cache.logits_per_step[static_cast<size_t>(n)] = logits_n;

        // lam = sigmoid(halt(yf))
        mt::Tensor lam_t = zeros2d(1, M);
        float* lp = lam_t.ptr<float>();
        for (int64_t m = 0; m < M; ++m) {
            const float* yrow = yf.ptr<float>() + m * d;
            double dot = 0.0;
            for (int64_t dd = 0; dd < d; ++dd)
                dot += static_cast<double>(hw[dd]) * static_cast<double>(yrow[dd]);
            dot += hb;
            lp[m] = sigmoid(static_cast<float>(dot));
        }
        cache.lam_per_step[static_cast<size_t>(n)] = lam_t;

        if (n == n_steps - 1) {
            for (int64_t m = 0; m < M; ++m) lp[m] = 1.0f;
        } else if (n < min_steps - 1) {
            for (int64_t m = 0; m < M; ++m) lp[m] = 0.0f;
        }
        cache.lam_per_step[static_cast<size_t>(n)] = lam_t;
    }

    return cache;
}

// ---- Cross-entropy loss ----

float cross_entropy_loss(const BackwardCache& cache,
                         mt::Tensor& d_logits_out) {
    const int M = cache.M;
    const int V = cache.V;
    const int N = cache.n_steps;

    const auto& tgt_ids = cache.target_flat;

    float total_loss = 0.0f;
    d_logits_out = zeros2d(static_cast<int64_t>(N) * M, V);
    float* dp = d_logits_out.ptr<float>();

    // cum tracking (for p_n computation)
    std::vector<float> cum = cache.cum_init;

    for (int n = 0; n < N; ++n) {
        const mt::Tensor& logits = cache.logits_per_step[static_cast<size_t>(n)];
        const float* lam_p = cache.lam_per_step[static_cast<size_t>(n)].ptr<float>();

        for (int64_t t = 0; t < M; ++t) {
            float pn = cum[static_cast<size_t>(t)] * lam_p[t];  // p_n(t)
            // Softmax
            float mx = -std::numeric_limits<float>::infinity();
            for (int64_t v = 0; v < V; ++v) {
                float l = logits.atf(t * V + v);
                if (l > mx) mx = l;
            }
            double sum = 0.0;
            std::vector<float> exps(static_cast<size_t>(V));
            for (int64_t v = 0; v < V; ++v) {
                exps[static_cast<size_t>(v)] = static_cast<float>(std::exp(logits.atf(t * V + v) - mx));
                sum += exps[static_cast<size_t>(v)];
            }
            int32_t tgt = tgt_ids[static_cast<size_t>(t)];
            float prob_tgt = exps[static_cast<size_t>(tgt)] / static_cast<float>(sum);
            float ce = -static_cast<float>(std::log(std::max(prob_tgt, 1e-30f)));
            // The loss reported is the mean over the batch, matching
            // recur.py's L.sum(0).mean() and the /M in the gradient below.
            // Summing here without the 1/M reported an M-times-larger number
            // than the objective actually being differentiated.
            total_loss += pn * ce / static_cast<float>(M);

            // Gradient: dL/dlogit[n,t,v] = pn * (probs[tgt] - onehot) / M
            for (int64_t v = 0; v < V; ++v) {
                double grad = static_cast<double>(exps[static_cast<size_t>(v)]) / sum;
                if (static_cast<int64_t>(v) == tgt) grad -= 1.0;
                grad *= static_cast<double>(pn) / static_cast<double>(M);
                dp[n * M * V + t * V + v] = static_cast<float>(grad);
            }

            cum[static_cast<size_t>(t)] *= (1.0f - lam_p[t]);
        }
    }

    return total_loss;
}

// ---- Backward helpers ----

namespace {

void add_into(mt::Tensor& dst, const mt::Tensor& src) {
    for (int64_t i = 0; i < dst.shape.numel(); ++i) {
        dst.set_flat(i, dst.at_flat(i) + src.at_flat(i));
    }
}

// y_i = x_i * r * w_i with r = 1/sqrt(mean(x^2)+eps) shared by the row.
//   dw_j  = sum_rows  g_j * (x_j * r)
//   du_j  = g_j * w_j                       (u = x*r)
//   dx_i  = r*du_i - r^3 * (x_i/D) * sum_j du_j*x_j
// The second term is the normalisation Jacobian. Dropping it (as an earlier
// version of this file did) leaves every gradient that flows THROUGH a norm
// wrong - ln1, qkv, proj, adapter and tok_emb included - while leaving the
// weight gradients alone, which is exactly the failure signature this test
// caught.
void rms_norm_backward(const mt::Tensor& x, const mt::Tensor& w,
                       const mt::Tensor& d_y, float eps,
                       mt::Tensor& d_x, mt::Tensor& d_w) {
    const int64_t rows = x.shape.d[0];
    const int64_t cols = x.shape.d[1];
    for (int64_t r = 0; r < rows; ++r) {
        double sum_sq = 0.0;
        for (int64_t c = 0; c < cols; ++c) {
            const double v = x.at_flat(r * cols + c);
            sum_sq += v * v;
        }
        const double mean_sq = sum_sq / static_cast<double>(cols);
        const double scale = 1.0 / std::sqrt(mean_sq + static_cast<double>(eps));
        const double scale3 = scale * scale * scale;

        // sum_j du_j * x_j, needed by the Jacobian term.
        double dot = 0.0;
        for (int64_t c = 0; c < cols; ++c) {
            dot += d_y.at_flat(r * cols + c) * w.at_flat(c) * x.at_flat(r * cols + c);
        }

        for (int64_t c = 0; c < cols; ++c) {
            const double xv = x.at_flat(r * cols + c);
            const double dy = d_y.at_flat(r * cols + c);
            const double du = dy * w.at_flat(c);
            d_x.set_flat(r * cols + c,
                         scale * du - scale3 * (xv / static_cast<double>(cols)) * dot);
            d_w.set_flat(c, d_w.at_flat(c) + dy * xv * scale);
        }
    }
}

// RoPE is a per-pair rotation [[c,-s],[s,c]], so its transpose is the
// rotation by -theta: dq0 = g0*c + g1*s ; dq1 = -g0*s + g1*c.
void rope_backward(const mt::Tensor& d_rot, const mt::Tensor& cos,
                   const mt::Tensor& sin, int T, int H, int hd,
                   mt::Tensor& d_raw) {
    const int64_t half = hd / 2;
    for (int64_t h = 0; h < H; ++h) {
        for (int64_t t = 0; t < T; ++t) {
            for (int64_t i = 0; i < half; ++i) {
                const double c = cos.atf(t * half + i);
                const double s = sin.atf(t * half + i);
                const int64_t i0 = (h * T + t) * hd + 2 * i;
                const int64_t i1 = i0 + 1;
                const double g0 = d_rot.at_flat(i0);
                const double g1 = d_rot.at_flat(i1);
                d_raw.set_flat(i0, g0 * c + g1 * s);
                d_raw.set_flat(i1, -g0 * s + g1 * c);
            }
        }
    }
}

// The one attention block shared by the prelude and every recurrent site.
// Forward (per block, B=1):
//   xn1 = rms_norm(x, ln1) ; qkv = xn1 @ qkv_w^T
//   q,k,v = split(qkv) ; qh,kh = rope(head_major(q), rope(head_major(k)))
//   kc,vc = concat(cached_kv, head_major(k), head_major(v))
//   sm = softmax(qh @ kc^T / sqrt(hd))  (causal mask constant => no gradient)
//   out_h = sm @ vc ; out_flat = to_BTHD(out_h) ; proj_out = out_flat @ proj_w^T
//   x_after = x + proj_out
// d_block_out is the gradient on x_after. Fills g_qkv/g_proj and returns
// d_xn1 [M,d] for the caller's RMSNorm backward.
void attn_backward(const mt::Tensor& xn1, const mt::Tensor& qh,
                   const mt::Tensor& kc, const mt::Tensor& vc,
                   const mt::Tensor& sm, const mt::Tensor& cos, const mt::Tensor& sin,
                   int T, int H, int hd, int d, float inv_sqrt,
                   const mt::Tensor& out_flat, const mt::Tensor& d_block_out,
                   const mt::Tensor& proj_w, const mt::Tensor& qkv_w,
                   mt::Tensor& g_qkv, mt::Tensor& g_proj, mt::Tensor& d_xn1) {
    const int64_t kv_len = sm.shape.d[1];
    const int64_t M = static_cast<int64_t>(T);

    // d_out_flat = d_block_out @ proj_w  (proj_out = out_flat @ proj^T)
    mt::Tensor d_out_flat = zeros2d(M, d);
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t c = 0; c < d; ++c) {
            double acc = 0.0;
            for (int64_t row = 0; row < d; ++row) {
                acc += d_block_out.at_flat(m * d + row) * proj_w.at_flat(row * d + c);
            }
            d_out_flat.set_flat(m * d + c, acc);
        }
    }
    // g_proj += d_block_out^T @ out_flat
    for (int64_t row = 0; row < d; ++row) {
        for (int64_t col = 0; col < d; ++col) {
            double acc = 0.0;
            for (int64_t m = 0; m < M; ++m) {
                acc += d_block_out.at_flat(m * d + row) * out_flat.at_flat(m * d + col);
            }
            g_proj.set_flat(row * d + col, g_proj.at_flat(row * d + col) + acc);
        }
    }

    // Head-major [H*T, hd] gradient of the attention output.
    mt::Tensor d_out_h = zeros2d(H * T, hd);
    for (int64_t h = 0; h < H; ++h) {
        for (int64_t t = 0; t < T; ++t) {
            for (int64_t dd = 0; dd < hd; ++dd) {
                d_out_h.set_flat((h * T + t) * hd + dd,
                                 d_out_flat.at_flat(t * d + h * hd + dd));
            }
        }
    }

    // out = sm @ v => d_vc[j] += sm[i,j]*d_out[i] ; d_sm[i,j] = d_out[i].vc[j]
    mt::Tensor d_vc = zeros2d(H * kv_len, hd);
    mt::Tensor d_sm = zeros2d(H * T, kv_len);
    for (int64_t h = 0; h < H; ++h) {
        for (int64_t t = 0; t < T; ++t) {
            for (int64_t j = 0; j < kv_len; ++j) {
                const double weight = sm.at_flat((h * T + t) * kv_len + j);
                double dot = 0.0;
                for (int64_t dd = 0; dd < hd; ++dd) {
                    const double g = d_out_h.at_flat((h * T + t) * hd + dd);
                    d_vc.set_flat((h * kv_len + j) * hd + dd,
                                  d_vc.at_flat((h * kv_len + j) * hd + dd) + weight * g);
                    dot += g * vc.at_flat((h * kv_len + j) * hd + dd);
                }
                d_sm.set_flat((h * T + t) * kv_len + j, dot);
            }
        }
    }

    // Softmax Jacobian. Masked-out positions have sm==0, so their d_sm drops
    // out and the mask needs no explicit gradient term.
    mt::Tensor d_scores = zeros2d(H * T, kv_len);
    for (int64_t h = 0; h < H; ++h) {
        for (int64_t t = 0; t < T; ++t) {
            double row_dot = 0.0;
            for (int64_t j = 0; j < kv_len; ++j) {
                row_dot += sm.at_flat((h * T + t) * kv_len + j) *
                           d_sm.at_flat((h * T + t) * kv_len + j);
            }
            for (int64_t j = 0; j < kv_len; ++j) {
                d_scores.set_flat((h * T + t) * kv_len + j,
                                  sm.at_flat((h * T + t) * kv_len + j) *
                                  (d_sm.at_flat((h * T + t) * kv_len + j) - row_dot));
            }
        }
    }

    // scores = qh . kc / sqrt(hd)
    mt::Tensor d_qh = zeros2d(H * T, hd);
    mt::Tensor d_kc = zeros2d(H * kv_len, hd);
    for (int64_t h = 0; h < H; ++h) {
        for (int64_t t = 0; t < T; ++t) {
            for (int64_t j = 0; j < kv_len; ++j) {
                const double ds = d_scores.at_flat((h * T + t) * kv_len + j) * inv_sqrt;
                for (int64_t dd = 0; dd < hd; ++dd) {
                    d_qh.set_flat((h * T + t) * hd + dd,
                                  d_qh.at_flat((h * T + t) * hd + dd) +
                                      ds * kc.at_flat((h * kv_len + j) * hd + dd));
                    d_kc.set_flat((h * kv_len + j) * hd + dd,
                                  d_kc.at_flat((h * kv_len + j) * hd + dd) +
                                      ds * qh.at_flat((h * T + t) * hd + dd));
                }
            }
        }
    }

    // RoPE applies to q and k only. d_kc also carries the tail rows belonging to
    // earlier chunks, which are constant inputs: only the window's own keys
    // (the last T rows per head) are folded into d_qkv.
    mt::Tensor d_q_raw = zeros2d(H * T, hd);
    rope_backward(d_qh, cos, sin, T, H, hd, d_q_raw);
    mt::Tensor d_k_window = zeros2d(H * T, hd);
    for (int64_t h = 0; h < H; ++h) {
        for (int64_t t = 0; t < T; ++t) {
            for (int64_t dd = 0; dd < hd; ++dd) {
                d_k_window.set_flat((h * T + t) * hd + dd,
                                    d_kc.at_flat((h * kv_len + (kv_len - T) + t) * hd + dd));
            }
        }
    }
    mt::Tensor d_k_raw = zeros2d(H * T, hd);
    rope_backward(d_k_window, cos, sin, T, H, hd, d_k_raw);

    // Head-major -> interleaved [M, 3d]
    mt::Tensor d_qkv = zeros2d(M, 3 * d);
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t h = 0; h < H; ++h) {
            for (int64_t dd = 0; dd < hd; ++dd) {
                d_qkv.set_flat(t * 3 * d + h * hd + dd,
                               d_q_raw.at_flat((h * T + t) * hd + dd));
                d_qkv.set_flat(t * 3 * d + d + h * hd + dd,
                               d_k_raw.at_flat((h * T + t) * hd + dd));
                d_qkv.set_flat(t * 3 * d + 2 * d + h * hd + dd,
                               d_vc.at_flat((h * kv_len + (kv_len - T) + t) * hd + dd));
            }
        }
    }

    // g_qkv += d_qkv^T @ xn1
    for (int64_t row = 0; row < 3 * d; ++row) {
        for (int64_t col = 0; col < d; ++col) {
            double acc = 0.0;
            for (int64_t m = 0; m < M; ++m) {
                acc += d_qkv.at_flat(m * 3 * d + row) * xn1.at_flat(m * d + col);
            }
            g_qkv.set_flat(row * d + col, g_qkv.at_flat(row * d + col) + acc);
        }
    }
    // d_xn1 = d_qkv @ qkv_w
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t c = 0; c < d; ++c) {
            double acc = 0.0;
            for (int64_t row = 0; row < 3 * d; ++row) {
                acc += d_qkv.at_flat(m * 3 * d + row) * qkv_w.at_flat(row * d + c);
            }
            d_xn1.set_flat(m * d + c, acc);
        }
    }
}

}  // namespace

Gradients backward(const Config& cfg, Coder& coder,
                   const BackwardCache& cache,
                   const mt::Tensor& d_logits,
                   std::vector<model::KVCache>& caches,
                   std::vector<std::vector<float>>* d_lam_out) {

    (void)caches;
    const int d = cache.d;
    const int V = cache.V;
    const int M = cache.M;
    const int T = cache.T;
    const int H = cache.H;
    const int hd = cache.hd;
    const int n_steps = cache.n_steps;
    const int n_recur = cache.n_recur;
    const int n_prelude = cfg.n_prelude;
    const int dff = cfg.d_ff;
    const float inv_sqrt = 1.0f / std::sqrt(static_cast<float>(hd));

    Gradients grads;
    grads.grads.reserve(200);

    auto make_zero = [&](const std::string& name, const mt::Tensor& like) -> mt::Tensor& {
        for (size_t i = 0; i < grads.grads.size(); ++i)
            if (grads.grads[i].first == name) return grads.grads[i].second;
        grads.grads.emplace_back(name, mt::make_zeros(like.shape, mt::DType::FP32));
        return grads.grads.back().second;
    };

    auto& g_tok_emb = make_zero("tok_emb.weight", coder.tok_emb());
    auto& g_ln_f_w = make_zero("ln_f.weight", coder.ln_f_w());
    auto& g_halt_w = make_zero("halt.weight", coder.halt_w());
    auto& g_halt_b = make_zero("halt.bias", coder.halt_b());
    auto& g_adapter_w = make_zero("adapter.weight", coder.adapter_w());

    const auto& pre = coder.blocks()[0];
    auto& g_pre_ln1 = make_zero("prelude.0.ln1.weight", pre.ln1);
    auto& g_pre_qkv = make_zero("prelude.0.attn.qkv.weight", pre.qkv);
    auto& g_pre_proj = make_zero("prelude.0.attn.proj.weight", pre.proj);
    auto& g_pre_ln2 = make_zero("prelude.0.ln2.weight", pre.ln2);
    auto& g_pre_w1 = make_zero("prelude.0.mlp.w1.weight", pre.w1);
    auto& g_pre_w3 = make_zero("prelude.0.mlp.w3.weight", pre.w3);
    auto& g_pre_w2 = make_zero("prelude.0.mlp.w2.weight", pre.w2);

    // Per-block recurrent grads. The dense w1/w3/w2 exist only when use_pool is
    // off, so they are created on demand below rather than up front.
    struct BlockGrads {
        mt::Tensor* ln1 = nullptr;
        mt::Tensor* qkv = nullptr;
        mt::Tensor* proj = nullptr;
        mt::Tensor* ln2 = nullptr;
        std::string prefix;
    };
    std::vector<BlockGrads> bg(static_cast<size_t>(n_recur));
    for (int r = 0; r < n_recur; ++r) {
        const auto& b = coder.blocks()[n_prelude + r];
        bg[static_cast<size_t>(r)].prefix = "recur." + std::to_string(r) + ".";
        bg[static_cast<size_t>(r)].ln1 =
            &make_zero(bg[static_cast<size_t>(r)].prefix + "ln1.weight", b.ln1);
        bg[static_cast<size_t>(r)].qkv =
            &make_zero(bg[static_cast<size_t>(r)].prefix + "attn.qkv.weight", b.qkv);
        bg[static_cast<size_t>(r)].proj =
            &make_zero(bg[static_cast<size_t>(r)].prefix + "attn.proj.weight", b.proj);
        bg[static_cast<size_t>(r)].ln2 =
            &make_zero(bg[static_cast<size_t>(r)].prefix + "ln2.weight", b.ln2);
    }

    // Dense SwiGLU shared by the prelude and the non-pooled recur sites.
    // hn = rms_norm(x, ln2); g = silu(hn @ w1^T) * (hn @ w3^T); y = g @ w2^T
    auto dense_mlp_backward = [&](const mt::Tensor& x_in, const mt::Tensor& ln2_w,
                                  const mt::Tensor& h1, const mt::Tensor& h3,
                                  const mt::Tensor& g, const mt::Tensor& w1,
                                  const mt::Tensor& w3, const mt::Tensor& w2,
                                  const mt::Tensor& d_y,
                                  mt::Tensor& g_w1, mt::Tensor& g_w3,
                                  mt::Tensor& g_w2,
                                  mt::Tensor& g_ln2, mt::Tensor& d_xn) {

        // d_g = d_y @ w2   (y = g @ w2^T)
        mt::Tensor d_g = zeros2d(M, dff);
        for (int64_t m = 0; m < M; ++m) {
            for (int64_t f = 0; f < dff; ++f) {
                double acc = 0.0;
                for (int64_t c = 0; c < d; ++c) {
                    acc += d_y.at_flat(m * d + c) * w2.at_flat(c * dff + f);
                }
                d_g.set_flat(m * dff + f, acc);
            }
        }
        // g_w2 += d_y^T @ g
        for (int64_t row = 0; row < d; ++row) {
            for (int64_t col = 0; col < dff; ++col) {
                double acc = 0.0;
                for (int64_t m = 0; m < M; ++m) {
                    acc += d_y.at_flat(m * d + row) * g.at_flat(m * dff + col);
                }
                g_w2.set_flat(row * dff + col, g_w2.at_flat(row * dff + col) + acc);
            }
        }
        // d_h1 = d_g * h3 * silu'(h1) ; d_h3 = d_g * silu(h1)
        mt::Tensor d_h1 = zeros2d(M, dff);
        mt::Tensor d_h3 = zeros2d(M, dff);
        for (int64_t i = 0; i < M * dff; ++i) {
            const float h1v = h1.atf(i);
            const double dg = d_g.at_flat(i);
            d_h1.set_flat(i, dg * h3.at_flat(i) * silu_deriv(h1v));
            d_h3.set_flat(i, dg * silu1(h1v));
        }
        // g_w1 += d_h1^T @ hn ; g_w3 += d_h3^T @ hn
        mt::Tensor hn = mt::rms_norm(x_in, ln2_w, 1e-6f);
        for (int64_t row = 0; row < dff; ++row) {
            for (int64_t col = 0; col < d; ++col) {
                double a1 = 0.0;
                double a3 = 0.0;
                for (int64_t m = 0; m < M; ++m) {
                    a1 += d_h1.at_flat(m * dff + row) * hn.at_flat(m * d + col);
                    a3 += d_h3.at_flat(m * dff + row) * hn.at_flat(m * d + col);
                }
                g_w1.set_flat(row * d + col, g_w1.at_flat(row * d + col) + a1);
                g_w3.set_flat(row * d + col, g_w3.at_flat(row * d + col) + a3);
            }
        }
        // d_hn = d_h1 @ w1 + d_h3 @ w3  (transposed projections)
        for (int64_t m = 0; m < M; ++m) {
            for (int64_t c = 0; c < d; ++c) {
                double acc = 0.0;
                for (int64_t f = 0; f < dff; ++f) {
                    acc += d_h1.at_flat(m * dff + f) * w1.at_flat(f * d + c);
                    acc += d_h3.at_flat(m * dff + f) * w3.at_flat(f * d + c);
                }
                d_xn.set_flat(m * d + c, acc);
            }
        }
        // ln2 weight grad, then the RMSNorm input path
        mt::Tensor tmp = zeros2d(M, d);
        rms_norm_backward(x_in, ln2_w, d_xn, 1e-6f, tmp, g_ln2);
        add_into(d_xn, tmp);
        (void)w3;
    };

    // ---- Halting ----
    // p_n = cum_{n-1} * lam_n ; cum_n = cum_{n-1} * (1 - lam_n)
    // L   = sum_n p_n * CE_n / M
    //   d_lam_n(t)   = cum_{n-1}(t) * (CE_n(t)/M - dc(t))
    //   dc_prev(t)    = CE_n(t)*lam_n(t)/M + dc(t)*(1 - lam_n(t))
    // dc is the gradient flowing from later steps into cum_n; it is zero at the
    // last step. cum_prev is recomputed forward from the cached lam values.
    //
    // The /M lives HERE, in step_ce, rather than in the recursion below,
    // because lam is not a differentiable output of the matmul chain: d_logits
    // cannot express dL/dlam, so this path re-derives the loss's dependence on
    // lam by hand and must therefore apply the batch mean itself. The d_logits
    // path gets the same factor independently at line ~653. The two are separate
    // expressions of one objective and they did drift: with the /M missing here,
    // every halting gradient was Mx too large, which the gradient suite could
    // not see because its halting rows had a tolerance loose enough to absorb a
    // constant factor. Verified by sweeping the batch size: the analytic/numeric
    // ratio was exactly 1/M (1.000000 at M=1, 0.500000 at M=2, 0.333333 at M=3).
    // Production runs M = chunk_size (8), so this was an 8x overestimate of
    // halt.weight, halt.bias, and - via the d_yf += dz*halt_w edge - of every
    // parameter upstream of ln_f.
    auto step_ce = [&](int n, std::vector<float>& ce) {
        const mt::Tensor& logits_n = cache.logits_per_step[static_cast<size_t>(n)];
        const float inv_M = 1.0f / static_cast<float>(M);
        ce.assign(static_cast<size_t>(M), 0.0f);
        for (int64_t t = 0; t < M; ++t) {
            const float* row = logits_n.ptr<float>() + t * V;
            float mx = -std::numeric_limits<float>::infinity();
            for (int64_t v = 0; v < V; ++v) {
                if (row[v] > mx) mx = row[v];
            }
            double sum = 0.0;
            float tgt_e = 0.0f;
            for (int64_t v = 0; v < V; ++v) {
                const float e = std::exp(row[v] - mx);
                sum += e;
                if (v == cache.target_flat[static_cast<size_t>(t)]) tgt_e = e;
            }
            const float prob = static_cast<float>(tgt_e / sum);
            ce[static_cast<size_t>(t)] = -std::log(std::max(prob, 1e-30f)) * inv_M;
        }
    };


    // dH[n] accumulates every path into h at step n.
    std::vector<mt::Tensor> dH(static_cast<size_t>(n_steps), zeros2d(M, d));
    // The adapter reads the prelude output at EVERY step, not only step 0, so
    // the prelude's gradient is a sum over steps. Accumulating it only for
    // n==0 left it roughly a third short (the adjoint measured 1.33x on
    // tok_emb), which then propagates into the adapter and the embedding.
    mt::Tensor d_pre_out = zeros2d(M, d);


    // Precompute per-step CE once; the halting recursion needs all of them.
    std::vector<std::vector<float>> ce_per_step(static_cast<size_t>(n_steps));
    for (int n = 0; n < n_steps; ++n) step_ce(n, ce_per_step[static_cast<size_t>(n)]);

    // d_lam per step. dc is dL/d(cum_n): zero at the last step, then rolled
    // back one step at a time. cum_prev is recomputed forward from the cached
    // lam values. The recursion's SHAPE is right; its SCALE comes from step_ce,
    // which applies the batch mean (see the note there).
    //
    // Do not "verify" this against an M=1 hand-written reference. That is what
    // hid the missing /M: such a reference is vacuously correct in the one
    // regime where the factor is invisible, and it will confidently refute the
    // fix. Sweep the batch size instead - a correct recursion gives the same
    // per-step ratio at every M. test_backward case 2 now does this at M=4.
    std::vector<std::vector<float>> d_lam_per_step(static_cast<size_t>(n_steps));
    {
        std::vector<float> dc(static_cast<size_t>(M), 0.0f);
        for (int n = n_steps - 1; n >= 0; --n) {
            const std::vector<float>& ce = ce_per_step[static_cast<size_t>(n)];
            const float* lam_n = cache.lam_per_step[static_cast<size_t>(n)].ptr<float>();

            std::vector<float> cum_prev(static_cast<size_t>(M), 1.0f);
            for (int nn = 0; nn < n; ++nn) {
                const float* lam_k = cache.lam_per_step[static_cast<size_t>(nn)].ptr<float>();
                for (int64_t t = 0; t < M; ++t) {
                    cum_prev[static_cast<size_t>(t)] *= (1.0f - lam_k[t]);
                }
            }

            d_lam_per_step[static_cast<size_t>(n)].resize(static_cast<size_t>(M));
            for (int64_t t = 0; t < M; ++t) {
                d_lam_per_step[static_cast<size_t>(n)][static_cast<size_t>(t)] =
                    cum_prev[static_cast<size_t>(t)] *
                    (ce[static_cast<size_t>(t)] - dc[static_cast<size_t>(t)]);
            }

            std::vector<float> dc_prev(static_cast<size_t>(M), 0.0f);
            for (int64_t t = 0; t < M; ++t) {
                dc_prev[static_cast<size_t>(t)] =
                    ce[static_cast<size_t>(t)] * lam_n[t] +
                    dc[static_cast<size_t>(t)] * (1.0f - lam_n[t]);
            }
            dc = dc_prev;
        }
    }

    // Hand the analytic d_lam to the caller. The halting gradient cannot be
    // validated by perturbing a parameter (the float32 forward buries it), and
    // a test that re-derives this recursion cannot catch a bug in it - that is
    // how a missing batch mean reached a green suite. Exposing the library's
    // own value lets a test compare against the shipped code.
    if (d_lam_out != nullptr) {
        *d_lam_out = d_lam_per_step;
    }



    for (int n = n_steps - 1; n >= 0; --n) {
        const mt::Tensor& yf = cache.yf_per_step[static_cast<size_t>(n)];
        const float* lam_n = cache.lam_per_step[static_cast<size_t>(n)].ptr<float>();
        const std::vector<float>& d_lam = d_lam_per_step[static_cast<size_t>(n)];

        // d_logits_n
        mt::Tensor d_logits_n = zeros2d(M, V);
        for (int64_t i = 0; i < M * V; ++i) {
            d_logits_n.set_flat(i, d_logits.at_flat(static_cast<int64_t>(n) * M * V + i));
        }

        // Tied head: logits = yf @ tok_emb^T. d_yf = d_logits @ tok_emb and the
        // head contributes d_logits^T @ yf to tok_emb.
        mt::Tensor d_yf = zeros2d(M, d);
        for (int64_t m = 0; m < M; ++m) {
            for (int64_t c = 0; c < d; ++c) {
                double acc = 0.0;
                for (int64_t v = 0; v < V; ++v) {
                    acc += d_logits_n.at_flat(m * V + v) * coder.tok_emb().at_flat(v * d + c);
                }
                d_yf.set_flat(m * d + c, acc);
            }
        }
        for (int64_t v = 0; v < V; ++v) {
            for (int64_t c = 0; c < d; ++c) {
                double acc = 0.0;
                for (int64_t m = 0; m < M; ++m) {
                    acc += d_logits_n.at_flat(m * V + v) * yf.at_flat(m * d + c);
                }
                g_tok_emb.set_flat(v * d + c, g_tok_emb.at_flat(v * d + c) + acc);
            }
        }

        // halt: lam = sigmoid(halt_w . yf + halt_b)
        for (int64_t t = 0; t < M; ++t) {
            const double lam_t = lam_n[t];
            const double dz = d_lam[static_cast<size_t>(t)] * lam_t * (1.0 - lam_t);
            for (int64_t c = 0; c < d; ++c) {
                // lam = sigmoid(halt_w . yf + halt_b) reads the SAME state as
                // the decoder head, so d_lam has two edges: into halt_w, and
                // back into yf. Without the second, every parameter upstream
                // of ln_f is wrong whenever halting is free - invisible while
                // min_steps == max_steps forces lam to 0/1 and d_lam to zero.
                const double via_halt = dz * coder.halt_w().at_flat(c);
                g_halt_w.set_flat(c, g_halt_w.at_flat(c) + dz * yf.at_flat(t * d + c));
                d_yf.set_flat(t * d + c, d_yf.at_flat(t * d + c) + via_halt);
            }
            g_halt_b.set_flat(0, g_halt_b.at_flat(0) + dz);
        }


        // ln_f: yf = rms_norm(h, ln_f_w) where h is the post-block state.
        const mt::Tensor& h_end = cache.h_end_per_step[static_cast<size_t>(n)];
        mt::Tensor d_h_end = zeros2d(M, d);
        rms_norm_backward(h_end, coder.ln_f_w(), d_yf, 1e-6f, d_h_end, g_ln_f_w);
        add_into(dH[static_cast<size_t>(n)], d_h_end);

        // ---- Recurrent blocks, last to first ----
        for (int r = n_recur - 1; r >= 0; --r) {
            const auto& b = coder.blocks()[n_prelude + r];
            BlockGrads& g = bg[static_cast<size_t>(r)];
            const size_t sn = static_cast<size_t>(n);
            const size_t sr = static_cast<size_t>(r);

            // h_out = h_in + proj_out + mlp_out, so the incoming gradient
            // dH[n] reaches h_in directly (residual) and the MLP.
            mt::Tensor d_block_in = zeros2d(M, d);
            add_into(d_block_in, dH[sn]);

            // --- attention ---
            mt::Tensor d_xn1 = zeros2d(M, d);
            attn_backward(cache.rb_xn1[sn][sr], cache.rb_qh[sn][sr],
                          cache.rb_kc[sn][sr], cache.rb_vc[sn][sr],
                          cache.rb_sm[sn][sr], cache.cos, cache.sin,
                          T, H, hd, d, inv_sqrt,
                          cache.rb_out_flat[sn][sr], dH[sn],
                          b.proj, b.qkv, *g.qkv, *g.proj, d_xn1);
            // ln1: d_h_block_in += rms_norm_backward(rb_in, ln1, d_xn1)
            add_into(d_block_in, d_xn1);
            {
                mt::Tensor tmp = zeros2d(M, d);
                rms_norm_backward(cache.rb_in[sn][sr], b.ln1, d_xn1, 1e-6f, tmp, *g.ln1);
                add_into(d_block_in, tmp);
            }

            // --- MLP (pooled or dense) ---
            const mt::Tensor& h_after_attn = cache.rb_after_attn[sn][sr];
            mt::Tensor d_xn2 = zeros2d(M, d);
            if (coder.pool() && cfg.use_pool) {
                const auto& pool = *coder.pool();
                const PoolBackwardCache& pcache = cache.rb_pool_cache[sn][sr];
                if (!pcache.kept.empty()) {
                    mt::Tensor d_router_w, d_depth_emb, d_ew1, d_ew3, d_ew2, d_egate;
                    mt::Tensor d_pool_x = pool_mlp_backward(
                        pool, pcache, dH[sn], b.router, b.depth_emb,
                        cfg.pool_top_k, cfg.pool_capacity_factor,
                        &d_router_w, &d_depth_emb, &d_ew1, &d_ew3, &d_ew2, &d_egate);
                    grads.add(g.prefix + "mlp.router.weight", d_router_w);
                    grads.add(g.prefix + "mlp.depth_emb", d_depth_emb);
                    grads.add("pool.gate", d_egate);

                    const std::vector<int>& pslots = pool.slots();
                    for (size_t s = 0; s < pslots.size(); ++s) {
                        const int pos = pslots[s];
                        if (pos < 0) continue;
                        const int uid = pool.expert_uid(pos);
                        if (uid < 0) continue;
                        const std::string base = "pool.experts." + std::to_string(uid) + ".";
                        auto take_row = [&](const mt::Tensor& src) {
                            const int64_t rn = src.shape.d[1] * src.shape.d[2];
                            mt::Shape sh;
                            sh.rank = 2;
                            sh.d[0] = src.shape.d[1];
                            sh.d[1] = src.shape.d[2];
                            mt::Tensor row = mt::make(sh, mt::DType::FP32, 0.0f);
                            std::memcpy(row.ptr<float>(),
                                        src.ptr<float>() + s * static_cast<size_t>(rn),
                                        static_cast<size_t>(rn) * sizeof(float));
                            return row;
                        };
                        grads.add(base + "w1.weight", take_row(d_ew1));
                        grads.add(base + "w3.weight", take_row(d_ew3));
                        grads.add(base + "w2.weight", take_row(d_ew2));
                    }
                    // ln2 backward on the pool input
                    mt::Tensor tmp = zeros2d(M, d);
                    rms_norm_backward(h_after_attn, b.ln2, d_pool_x, 1e-6f, tmp, *g.ln2);
                    add_into(d_xn2, tmp);
                }
            } else {
                mt::Tensor& g_w1 = make_zero(g.prefix + "mlp.w1.weight", b.w1);
                mt::Tensor& g_w3 = make_zero(g.prefix + "mlp.w3.weight", b.w3);
                mt::Tensor& g_w2 = make_zero(g.prefix + "mlp.w2.weight", b.w2);
                dense_mlp_backward(h_after_attn, b.ln2,
                                   cache.rb_h1[sn][sr], cache.rb_h3[sn][sr],
                                   cache.rb_g[sn][sr], b.w1, b.w3, b.w2,
                                   dH[sn], g_w1, g_w3, g_w2, *g.ln2, d_xn2);
            }
            add_into(d_block_in, d_xn2);

            // The gradient on the block input flows to the previous block (or,
            // for r==0, to the adapter output).
            dH[sn] = d_block_in;
        }

        // ---- adapter: h_after = cat([h_prev, x]) @ adapter_w^T ----
        // d_adapter_w[:, :d]   += d_h_after^T @ h_prev
        // d_adapter_w[:, d:2d]  += d_h_after^T @ x
        // d_h_prev               = d_h_after @ adapter_w[:, :d]
        // d_x                    = d_h_after @ adapter_w[:, d:2d]
        // At n==0, h_prev is exactly zero, so only the x half gets a weight
        // grad, while d_x is still the full matrix product.
        {
            const mt::Tensor& d_h_after = dH[static_cast<size_t>(n)];
            const mt::Tensor& x_input = cache.pre_out;
            const bool has_prev = (n > 0);

            // d_x_prelude = d_h_after @ adapter_w[:, d:2d]. Accrued at EVERY
            // step: the adapter's x half is the prelude's only output edge, and
            // the forward evaluates it once per step.
            for (int64_t m = 0; m < M; ++m) {
                for (int64_t c = 0; c < d; ++c) {
                    double acc = 0.0;
                    for (int64_t row = 0; row < d; ++row) {
                        acc += d_h_after.at_flat(m * d + row) *
                               coder.adapter_w().at_flat(row * 2 * d + d + c);
                    }
                    d_pre_out.set_flat(m * d + c, d_pre_out.at_flat(m * d + c) + acc);
                }
            }

            // d_adapter_w[:, :d] += d_h_after^T @ h_prev is skipped at n==0,
            // where h_prev is exactly zero.
            if (has_prev) {
                const mt::Tensor& h_prev = cache.h_before_adapter[static_cast<size_t>(n - 1)];
                for (int64_t row = 0; row < d; ++row) {
                    for (int64_t col = 0; col < d; ++col) {
                        double a_prev = 0.0;
                        double a_x = 0.0;
                        for (int64_t m = 0; m < M; ++m) {
                            a_prev += d_h_after.at_flat(m * d + row) * h_prev.at_flat(m * d + col);
                            a_x += d_h_after.at_flat(m * d + row) * x_input.at_flat(m * d + col);
                        }
                        g_adapter_w.set_flat(row * 2 * d + col,
                            g_adapter_w.at_flat(row * 2 * d + col) + a_prev);
                        g_adapter_w.set_flat(row * 2 * d + d + col,
                            g_adapter_w.at_flat(row * 2 * d + d + col) + a_x);
                    }
                }
            } else {
                for (int64_t row = 0; row < d; ++row) {
                    for (int64_t col = 0; col < d; ++col) {
                        double a_x = 0.0;
                        for (int64_t m = 0; m < M; ++m) {
                            a_x += d_h_after.at_flat(m * d + row) * x_input.at_flat(m * d + col);
                        }
                        g_adapter_w.set_flat(row * 2 * d + d + col,
                            g_adapter_w.at_flat(row * 2 * d + d + col) + a_x);
                    }
                }
            }

            if (has_prev) {
                mt::Tensor d_h_prev = zeros2d(M, d);
                for (int64_t m = 0; m < M; ++m) {
                    for (int64_t c = 0; c < d; ++c) {
                        double acc = 0.0;
                        for (int64_t row = 0; row < d; ++row) {
                            acc += d_h_after.at_flat(m * d + row) *
                                   coder.adapter_w().at_flat(row * 2 * d + c);
                        }
                        d_h_prev.set_flat(m * d + c, acc);
                    }
                }
                add_into(dH[static_cast<size_t>(n - 1)], d_h_prev);
            }
        }
    }

    // ---- Prelude block ----
    // Runs once, after every step has contributed to d_pre_out.
    {
        const auto& pre = coder.blocks()[0];
        (void)n_prelude;

        // pre_out = x_after_attn + mlp, so the MLP sees the FULL gradient on its
        // output (the residual share plus the path down to x_after_attn). It
        // writes its own input-side gradient to a separate tensor: aliasing
        // would leave the attention reading a tensor the MLP had overwritten.
        mt::Tensor d_pre_x_after_attn = zeros2d(M, d);
        add_into(d_pre_x_after_attn, d_pre_out);
        mt::Tensor d_pre_xn2 = zeros2d(M, d);
        dense_mlp_backward(cache.pre_x_after_attn, pre.ln2,
                           cache.pre_h1, cache.pre_h3, cache.pre_g,
                           pre.w1, pre.w3, pre.w2,
                           d_pre_x_after_attn, g_pre_w1, g_pre_w3, g_pre_w2,
                           g_pre_ln2, d_pre_xn2);
        add_into(d_pre_x_after_attn, d_pre_xn2);

        // attention residual: x_after_attn = x + proj_out
        mt::Tensor d_pre_xn1 = zeros2d(M, d);
        attn_backward(cache.pre_xn1, cache.pre_qh, cache.pre_kc,
                      cache.pre_vc, cache.pre_sm, cache.cos, cache.sin,
                      T, H, hd, d, inv_sqrt,
                      cache.pre_out_flat, d_pre_x_after_attn,
                      pre.proj, pre.qkv, g_pre_qkv, g_pre_proj, d_pre_xn1);
        mt::Tensor d_pre_x = zeros2d(M, d);
        add_into(d_pre_x, d_pre_x_after_attn);  // residual
        mt::Tensor tmp = zeros2d(M, d);
        rms_norm_backward(cache.pre_x, pre.ln1, d_pre_xn1, 1e-6f, tmp, g_pre_ln1);
        add_into(d_pre_x, tmp);

        // tok_emb scatter-add
        for (int64_t m = 0; m < M; ++m) {
            const int32_t id = cache.idx_flat[static_cast<size_t>(m)];
            if (id < 0 || id >= V) continue;
            for (int64_t c = 0; c < d; ++c) {
                g_tok_emb.set_flat(id * d + c,
                    g_tok_emb.at_flat(id * d + c) + d_pre_x.at_flat(m * d + c));
            }
        }
    }

    return grads;
}


// ---- Convenience wrapper ----

float compute_loss_and_grads(const Config& cfg, Coder& coder,
                             const mt::Tensor& idx,
                             const mt::Tensor& targets,
                             int64_t pos_offset,
                             std::vector<model::KVCache>& caches,
                             Gradients& grads_out,
                             BackwardCache* cache_out) {
    BackwardCache cache = forward_with_cache(cfg, coder, idx, targets, caches, pos_offset);
    if (cache.logits_per_step.empty()) {
        return 0.0f;
    }
    mt::Tensor d_logits;
    float loss = cross_entropy_loss(cache, d_logits);
    grads_out = backward(cfg, coder, cache, d_logits, caches, &cache.d_lam_per_step);
    if (cache_out != nullptr) {
        // Copy rather than move: `caches` and the tensors in here are still
        // referenced by the caller's model state, and a move left the
        // d_lam_per_step vectors in an indeterminate state that read back as
        // plausible-but-wrong numbers (2.06e-5 where 0.919 was expected).
        *cache_out = cache;
    }
    return loss;
}

}  // namespace minagi::backward
