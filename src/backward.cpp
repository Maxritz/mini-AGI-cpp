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

    // Prelude block
    int ci = 0;
    const auto& pre = coder.blocks()[0];

    // Attention: rn1 = rms_norm(x, ln1), qkv = rn1 @ qkv^T
    mt::Tensor rn1 = mt::rms_norm(x, pre.ln1, 1e-6f);
    mt::Tensor qkv_t = mt::matmul(rn1, transpose_2d(pre.qkv));
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
    cache.yf_per_step.resize(n_steps);
    cache.logits_per_step.resize(n_steps);
    cache.lam_per_step.resize(n_steps);
    cache.rb_xn1.resize(n_steps, std::vector<mt::Tensor>(n_recur));
    cache.rb_qkv.resize(n_steps, std::vector<mt::Tensor>(n_recur));
    cache.rb_qh.resize(n_steps, std::vector<mt::Tensor>(n_recur));
    cache.rb_kh.resize(n_steps, std::vector<mt::Tensor>(n_recur));
    cache.rb_vh.resize(n_steps, std::vector<mt::Tensor>(n_recur));
    cache.rb_sm.resize(n_steps, std::vector<mt::Tensor>(n_recur));
    cache.rb_out_flat.resize(n_steps, std::vector<mt::Tensor>(n_recur));
    cache.rb_in.resize(n_steps, std::vector<mt::Tensor>(n_recur));
    cache.rb_xn2.resize(n_steps);
    cache.rb_pool_cache.resize(n_steps);

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
            cache.rb_kh[static_cast<size_t>(n)][static_cast<size_t>(r)] = khb;
            cache.rb_vh[static_cast<size_t>(n)][static_cast<size_t>(r)] = vhb;
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
            cache.rb_kh[static_cast<size_t>(n)][static_cast<size_t>(r)] = khb;

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

            // Pooled MLP
            mt::Tensor hn2 = mt::rms_norm(h, b.ln2, 1e-6f);
            cache.rb_xn2[static_cast<size_t>(n)] = hn2;
            mt::Tensor mlp_out;
            if (coder.pool() && cfg.use_pool) {
            mlp_out = pool_mlp_forward(*coder.pool(), hn2, b.router, b.depth_emb,
                                       cfg.pool_top_k,
                                       cfg.pool_capacity_factor, nullptr,
                                       &cache.rb_pool_cache[static_cast<size_t>(n)]);
            } else {
                // Fallback: dense MLP using w1/w3/w2
                mt::Tensor h1b = mt::matmul(hn2, transpose_2d(b.w1));
                mt::Tensor h3b = mt::matmul(hn2, transpose_2d(b.w3));
                mt::Tensor gb = zeros2d(M, dff);
                float* gp = gb.ptr<float>();
                for (int64_t i = 0; i < M * dff; ++i)
                    gp[i] = silu1(h1b.atf(i)) * h3b.atf(i);
                mlp_out = mt::matmul(gb, transpose_2d(b.w2));
            }
            const float* mp = mlp_out.ptr<float>();
            float* hp = h.ptr<float>();
            for (int64_t i = 0; i < M * d; ++i)
                hp[i] += mp[i];
        }

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
            total_loss += pn * ce;

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

// ---- Backward ----

Gradients backward(const Config& cfg, Coder& coder,
                    const BackwardCache& cache,
                    const mt::Tensor& d_logits,
                    std::vector<model::KVCache>& caches) {
    const int d = cache.d;
    const int V = cache.V;
    const int M = cache.M;
    const int H = cache.H;
    const int hd = cache.hd;
    const int half = cache.half;
    const int n_steps = cache.n_steps;
    const int n_recur = cache.n_recur;
    const int n_prelude = cfg.n_prelude;
    const float inv_sqrt = 1.0f / std::sqrt(static_cast<float>(hd));
    const auto& tgt_ids = cache.target_flat;

    Gradients grads;
    grads.grads.reserve(200);

    auto make_zero = [&](const std::string& name, const mt::Tensor& like) -> mt::Tensor& {
        for (size_t i = 0; i < grads.grads.size(); ++i) if (grads.grads[i].first == name) return grads.grads[i].second;
        grads.grads.emplace_back(name, mt::make_zeros(like.shape, mt::DType::FP32));
        return grads.grads.back().second;
    };

    // Grad accumulators
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

    // Recur block grads
    std::vector<std::array<mt::Tensor*, 6>> g_recur;  // per-block: ln1, qkv, proj, ln2, router, depth_emb
    for (int r = 0; r < n_recur; ++r) {
        const auto& b = coder.blocks()[n_prelude + r];
        g_recur.push_back({
            &make_zero("recur." + std::to_string(r) + ".ln1.weight", b.ln1),
            &make_zero("recur." + std::to_string(r) + ".attn.qkv.weight", b.qkv),
            &make_zero("recur." + std::to_string(r) + ".attn.proj.weight", b.proj),
            &make_zero("recur." + std::to_string(r) + ".ln2.weight", b.ln2),
            &make_zero("recur." + std::to_string(r) + ".mlp.router.weight", b.router),
            &make_zero("recur." + std::to_string(r) + ".mlp.depth_emb", b.depth_emb)
        });
    }

    // Backward through recurrence (reverse time)
    std::vector<mt::Tensor> dH(n_steps, zeros2d(M, d));  // dH[n] = gradient on h AFTER adapter, BEFORE blocks

    std::vector<float> cum = cache.cum_init;

    for (int n = n_steps - 1; n >= 0; --n) {
        const mt::Tensor& yf = cache.yf_per_step[static_cast<size_t>(n)];
        const mt::Tensor& logits_n = cache.logits_per_step[static_cast<size_t>(n)];
        const float* lam_p = cache.lam_per_step[static_cast<size_t>(n)].ptr<float>();
        const mt::Tensor& h_after = cache.h_after_adapter[static_cast<size_t>(n)];

        // d_logits_n = d_logits[n*M:(n+1)*M, :]
        mt::Tensor d_logits_n = zeros2d(M, V);
        {
            const float* dp = d_logits.ptr<float>();
            float* lp = d_logits_n.ptr<float>();
            for (int64_t i = 0; i < M * V; ++i)
                lp[i] = dp[n * M * V + i];
        }

        // Head backward: logits = yf @ head_w^T => d_yf = d_logits @ head_w
        // Head backward: logits = yf @ head_w^T => d_yf = d_logits @ head_w
        mt::Tensor d_yf = zeros2d(M, d);
        {
            const float* lp = d_logits_n.ptr<float>();
            const float* wp = coder.tok_emb().ptr<float>();
            float* yp = d_yf.ptr<float>();
            for (int64_t m = 0; m < M; ++m) {
                for (int64_t c = 0; c < d; ++c) {
                    double acc = 0.0;
                    for (int64_t v = 0; v < V; ++v) {
                        acc += static_cast<double>(lp[m * V + v]) * static_cast<double>(wp[v * d + c]);
                    }
                    yp[m * d + c] = static_cast<float>(acc);
                }
            }
        }

        // tok_emb grad: d_head = d_logits^T @ yf  (head is tied to tok_emb)
        {
            const float* lp = d_logits_n.ptr<float>();
             const float* yp = yf.ptr<float>();
             for (int64_t v = 0; v < V; ++v) {
                for (int64_t c = 0; c < d; ++c) {
                    double acc = 0.0;
                    for (int64_t m = 0; m < M; ++m)
                        acc += static_cast<double>(lp[m * V + v]) * static_cast<double>(yp[m * d + c]);
                    g_tok_emb.set_flat(v * d + c, g_tok_emb.at_flat(v * d + c) + acc);
                }
            }
        }

        // --- Halting backward ---
        // lam_n(t) = sigmoid(halt_w @ yf_n(t) + halt_b)
        // p_n(t) = cum_{n-1}(t) * lam_n(t)
        // cum_n(t) = cum_{n-1}(t) * (1 - lam_n(t))
        // L = sum_n sum_t p_n(t) * CE_n(t)
        //
        // Going backward through time:
        //   dc = dL/d_cum_n  (gradient flowing from future steps into cum_n)
        //   At last step: dc = 0
        //   dL/d_lam_n(t) = cum_{n-1}(t) * (CE_n(t) - dc(t))
        //   dc_prev(t) = CE_n(t) * lam_n(t) + dc(t) * (1 - lam_n(t))

        // Compute CE_n(t) for this step
        std::vector<float> ce_n(static_cast<size_t>(M), 0.0f);
        for (int64_t t = 0; t < M; ++t) {
            const float* row_p = logits_n.ptr<float>() + t * V;
            float mx = -std::numeric_limits<float>::infinity();
            for (int64_t v = 0; v < V; ++v) {
                if (row_p[v] > mx) mx = row_p[v];
            }
            double sum = 0.0;
            float prob_tgt = 0.0f;
            for (int64_t v = 0; v < V; ++v) {
                float e = static_cast<float>(std::exp(row_p[v] - mx));
                sum += e;
                if (static_cast<int64_t>(v) == tgt_ids[static_cast<size_t>(t)])
                    prob_tgt = e;
            }
            prob_tgt = static_cast<float>(prob_tgt / sum);
            ce_n[static_cast<size_t>(t)] = -static_cast<float>(std::log(std::max(prob_tgt, 1e-30f)));
        }

        // cum_prev(t) = cum before this step (recomputed forward from init)
        std::vector<float> cum_prev(M);
        if (n == 0) {
            cum_prev = cache.cum_init;
        } else {
            std::vector<float> c = cache.cum_init;
            for (int nn = 0; nn < n; ++nn) {
                const float* lam_n = cache.lam_per_step[static_cast<size_t>(nn)].ptr<float>();
                for (int64_t t = 0; t < M; ++t) {
                    c[static_cast<size_t>(t)] *= (1.0f - lam_n[t]);
                }
            }
            cum_prev = c;
        }

        // dc: gradient on cum_n from future steps (n+1, ..., N-1)
        std::vector<float> dc(M, 0.0f);
        if (n < n_steps - 1) {
            std::vector<float> dc_fwd(M, 0.0f);
            for (int k = n + 1; k < n_steps; ++k) {
                const float* lam_k = cache.lam_per_step[static_cast<size_t>(k)].ptr<float>();
                const mt::Tensor& logits_k = cache.logits_per_step[static_cast<size_t>(k)];
                for (int64_t t = 0; t < M; ++t) {
                    const float* row_p = logits_k.ptr<float>() + t * V;
                    float mx = -std::numeric_limits<float>::infinity();
                    for (int64_t v = 0; v < V; ++v) if (row_p[v] > mx) mx = row_p[v];
                    double sum = 0.0;
                    float prob_tgt = 0.0f;
                    for (int64_t v = 0; v < V; ++v) {
                        float e = static_cast<float>(std::exp(row_p[v] - mx));
                        sum += e;
                        if (static_cast<int64_t>(v) == cache.target_flat[static_cast<size_t>(t)])
                            prob_tgt = e;
                    }
                    prob_tgt = static_cast<float>(prob_tgt / sum);
                    float ce_k = -static_cast<float>(std::log(std::max(prob_tgt, 1e-30f)));
                    dc_fwd[static_cast<size_t>(t)] = dc_fwd[static_cast<size_t>(t)] * (1.0f - lam_k[t])
                                                   + ce_k * lam_k[t];
                }
            }
            dc = dc_fwd;
        }

        // d_lam_n(t) = cum_prev(t) * (CE_n(t) - dc(t))
        std::vector<float> d_lam(static_cast<size_t>(M));
        for (int64_t t = 0; t < M; ++t) {
            d_lam[static_cast<size_t>(t)] = cum_prev[static_cast<size_t>(t)] *
                                            (ce_n[static_cast<size_t>(t)] - dc[static_cast<size_t>(t)]);
        }

        // d_halt_w/b: lam = sigmoid(halt_w @ yf + halt_b)
        const float* lam_ptr = lam_p;
        const float* yfp = yf.ptr<float>();
        for (int64_t t = 0; t < M; ++t) {
            float lam_t = lam_ptr[t];
            float dz = d_lam[static_cast<size_t>(t)] * lam_t * (1.0f - lam_t);
            for (int64_t c = 0; c < d; ++c) {
                g_halt_w.set_flat(c, g_halt_w.at_flat(c) + dz * yfp[t * d + c]);
            }
            g_halt_b.set_flat(0, g_halt_b.at_flat(0) + dz);
        }


        // ln_f backward: yf = rms_norm(h, ln_f_w)
        // Simplified: dyf/dx = ln_f_w (ignoring normalization coupling)
        // d_h = d_yf * ln_f_w
        mt::Tensor d_h = zeros2d(M, d);
        const float* yp = d_yf.ptr<float>();
        const float* fp = coder.ln_f_w().ptr<float>();
        float* hp = d_h.ptr<float>();
            for (int64_t m = 0; m < M; ++m) {
                for (int64_t c = 0; c < d; ++c) {
                hp[m * d + c] = static_cast<float>(yp[m * d + c]) * fp[c];
            }
        }

        // ln_f_w backward
        for (int64_t c = 0; c < d; ++c) {
            double acc = 0.0;
            for (int64_t m = 0; m < M; ++m) {
                acc += static_cast<double>(yp[m * d + c]) * static_cast<double>(h_after.at_flat(m * d + c));
            }
            g_ln_f_w.set_flat(c, g_ln_f_w.at_flat(c) + acc);
        }

        // h_after is the output of adapter cat([h_prev, x]) @ adapter_w^T
        // d_h_after flows back to: d_h_prev (recurrent) + d_x (prelude)
        // Also d_h_after = d_h (from ln_f) + dH_next (from next step's adapter)
        for (int64_t i = 0; i < M * d; ++i) {
            dH[static_cast<size_t>(n)].set_flat(i, dH[static_cast<size_t>(n)].at_flat(i) + d_h.at_flat(i));
        }

        // Backprop through adapter (cat([h_prev, x]) @ adapter_w^T)
        if (n > 0) {
            const mt::Tensor& h_prev = cache.h_before_adapter[static_cast<size_t>(n - 1)];
            const mt::Tensor& x_input = cache.pre_out;  // [M, d]

            // d_adapter_w += d_h_after^T @ cat  -> [d, 2d]
            const float* dh = dH[static_cast<size_t>(n)].ptr<float>();
            const float* hp = h_prev.ptr<float>();
            const float* xp = x_input.ptr<float>();
            for (int64_t row = 0; row < d; ++row) {
                for (int64_t col = 0; col < d; ++col) {
                    double acc = 0.0;
                    for (int64_t m = 0; m < M; ++m) {
                        acc += static_cast<double>(dh[m * d + row]) * static_cast<double>(hp[m * d + col]);
                    }
                    g_adapter_w.set_flat(row * 2 * d + col,
                        g_adapter_w.at_flat(row * 2 * d + col) + acc);
                }
            }
            for (int64_t row = 0; row < d; ++row) {
                for (int64_t col = 0; col < d; ++col) {
                    double acc = 0.0;
                    for (int64_t m = 0; m < M; ++m) {
                        acc += static_cast<double>(dh[m * d + row]) * static_cast<double>(xp[m * d + col]);
                    }
                    g_adapter_w.set_flat(row * 2 * d + d + col,
                        g_adapter_w.at_flat(row * 2 * d + d + col) + acc);
                }
            }

            // d_h_prev = d_h_after @ adapter_w[:, :d]  (first half of adapter_w)
            mt::Tensor d_h_prev = zeros2d(M, d);
            for (int64_t m = 0; m < M; ++m) {
                for (int64_t c = 0; c < d; ++c) {
                    double acc = 0.0;
                    for (int64_t row = 0; row < d; ++row) {
                        acc += static_cast<double>(dh[m * d + row]) *
                               static_cast<double>(coder.adapter_w().at_flat(row * 2 * d + c));
                    }
                    d_h_prev.set_flat(m * d + c, acc);
                }
            }
            // Add to dH[n-1]
            for (int64_t i = 0; i < M * d; ++i) {
                dH[static_cast<size_t>(n - 1)].set_flat(i,
                    dH[static_cast<size_t>(n - 1)].at_flat(i) + d_h_prev.at_flat(i));
            }
        }

        // Backprop through recurrent blocks (reverse)
        for (int r = n_recur - 1; r >= 0; --r) {
            int ci_r = n_prelude + r;
            const auto& b = coder.blocks()[ci_r];
            mt::Tensor& d_h_block = dH[static_cast<size_t>(n)];  // gradient on h after adapter, before blocks

            // h = h + proj_out_b (residual) + mlp_out
            // d_xn2_input = d_h_block (through residual)
            // d_mlp_out = d_h_block (through residual)

            // Backprop MLP
            // For pooled MLP: mlp_out = pool_mlp_forward(pool, hn2, router, depth_emb, ...)
            // For dense MLP: mlp_out = silu(hn2 @ w1^T) * h3 @ w2^T

            if (coder.pool() && cfg.use_pool) {
                // Pool MLP backward
                const auto& pool = *coder.pool();
                const PoolBackwardCache& pcache = cache.rb_pool_cache[static_cast<size_t>(n)];
                if (!pcache.kept.empty()) {
                    mt::Tensor d_router_w, d_depth_emb;
                    mt::Tensor d_pool_x = pool_mlp_backward(
                        pool, pcache, d_h_block, b.router, b.depth_emb,
                        cfg.pool_top_k, cfg.pool_capacity_factor,
                        &d_router_w, &d_depth_emb);

                    // Add router weight gradient
                    grads.add("recur." + std::to_string(r) + ".mlp.router.weight", d_router_w);
                    // Add depth_emb gradient
                    grads.add("recur." + std::to_string(r) + ".mlp.depth_emb", d_depth_emb);

                    // d_h_block flows to hn2 through residual (d_mlp_out = d_h_block)
                    // d_xn2 = d_h_block (gradient propagates through residual)
                    // d_h = d_xn2 (the input to the block)

                    // Expert weight gradients: pool weights trained via separate path
                    // (expert files on disk, updated through optimizer in training loop)
                    // Gate gradient is approximate: push towards balanced usage
                    // For now, skip gate gradient and per-expert weight gradients
                    // (they would require saving per-expert activations, which the
                    // cache doesn't do)
                }
            }

            // ln2 backward
            mt::Tensor hn2 = cache.rb_xn2[static_cast<size_t>(n)];
            mt::Tensor d_xn2 = d_h_block;
        }

        // At step 0, dH[0] contributes to the prelude output gradient
        if (n == 0) {
            // d_x_input = d_h (from step 0's adapter, the x portion)
            // The adapter at step 0: cat([h_{-1}=0, x]) @ adapter_w^T
            // d_x = d_h_after @ adapter_w[:, d:]
            const float* dh0 = dH[static_cast<size_t>(0)].ptr<float>();
            const float* wp = coder.adapter_w().ptr<float>();
            mt::Tensor d_x_input = zeros2d(M, d);
            float* xp = d_x_input.ptr<float>();
            for (int64_t m = 0; m < M; ++m) {
                for (int64_t c = 0; c < d; ++c) {
                    double acc = 0.0;
                    for (int64_t row = 0; row < d; ++row) {
                        acc += static_cast<double>(dh0[m * d + row]) *
                               static_cast<double>(wp[row * 2 * d + d + c]);
                    }
                    xp[m * d + c] = static_cast<float>(acc);
                }
            }

            // Now backprop d_x_input through the prelude block
            // pre_out = x_after_attn + mlp(g)
            // x_after_attn = x + proj_out (x = tok_emb(idx))
            // d_pre_out = d_x_input (through residual)
            // d_x_after_attn = d_x_input, d_mlp = d_x_input

            // Backprop MLP
            // g = silu(h1) * h3, g = xn2 @ w1^T, etc.
            mt::Tensor xn2_pre = cache.pre_xn2;
            mt::Tensor d_g = zeros2d(M, cfg.d_ff);
            // d_g = d_mlp @ w2  -> [M, dff]
            {
                const float* dp = d_x_input.ptr<float>();
                float* gp = d_g.ptr<float>();
                for (int64_t m = 0; m < M; ++m) {
                    for (int64_t f = 0; f < cfg.d_ff; ++f) {
                        double acc = 0.0;
                        for (int64_t c = 0; c < d; ++c) {
                            acc += static_cast<double>(dp[m * d + c]) *
                                   static_cast<double>(pre.w2.at_flat(c * cfg.d_ff + f));
                        }
                        gp[m * cfg.d_ff + f] = static_cast<float>(acc);
                    }
                }
            }

            // d_w2 = d_mlp^T @ g
            {
                const float* dp = d_x_input.ptr<float>();
                const float* gp = cache.pre_g.ptr<float>();
                for (int64_t row = 0; row < cfg.d_ff; ++row) {
                    for (int64_t col = 0; col < d; ++col) {
                        double acc = 0.0;
                        for (int64_t m = 0; m < M; ++m) {
                            acc += static_cast<double>(gp[m * cfg.d_ff + row]) *
                                   static_cast<double>(dp[m * d + col]);
                        }
                        g_pre_w2.set_flat(row * d + col, g_pre_w2.at_flat(row * d + col) + acc);
                    }
                }
            }

            // d_h1 = d_g * h3 * silu'(h1), d_h3 = d_g * silu(h1)
            const float* g1p = cache.pre_h1.ptr<float>();
            const float* h3p = cache.pre_h3.ptr<float>();
            const float* dgp = d_g.ptr<float>();
            mt::Tensor d_h1 = d_g, d_h3 = d_g;
            float* d1p = d_h1.ptr<float>();
            float* d3p = d_h3.ptr<float>();
            for (int64_t i = 0; i < M * cfg.d_ff; ++i) {
                float s = silu1(g1p[i]);
                d1p[i] = dgp[i] * h3p[i] * silu_deriv(g1p[i]);
                d3p[i] = dgp[i] * s;
            }

            // d_w1 = d_h1^T @ xn2
            {
                const float* d1p = d_h1.ptr<float>();
                const float* xp2 = xn2_pre.ptr<float>();
                for (int64_t row = 0; row < cfg.d_ff; ++row) {
                    for (int64_t col = 0; col < d; ++col) {
                        double acc = 0.0;
                        for (int64_t m = 0; m < M; ++m) {
                            acc += static_cast<double>(d1p[m * cfg.d_ff + row]) *
                                   static_cast<double>(xp2[m * d + col]);
                        }
                        g_pre_w1.set_flat(row * d + col, g_pre_w1.at_flat(row * d + col) + acc);
                    }
                }
            }

            // d_w3 = d_h3^T @ xn2
            {
                const float* d3p = d_h3.ptr<float>();
                const float* xp2 = xn2_pre.ptr<float>();
                for (int64_t row = 0; row < cfg.d_ff; ++row) {
                    for (int64_t col = 0; col < d; ++col) {
                        double acc = 0.0;
                        for (int64_t m = 0; m < M; ++m) {
                            acc += static_cast<double>(d3p[m * cfg.d_ff + row]) *
                                   static_cast<double>(xp2[m * d + col]);
                        }
                        g_pre_w3.set_flat(row * d + col, g_pre_w3.at_flat(row * d + col) + acc);
                    }
                }
            }

            // d_xn2 = d_h1 @ w1 + d_h3 @ w3  (through transpose)
            mt::Tensor d_xn2 = zeros2d(M, d);
            {
                const float* d1p = d_h1.ptr<float>();
                const float* d3p = d_h3.ptr<float>();
                float* xp = d_xn2.ptr<float>();
                for (int64_t m = 0; m < M; ++m) {
                    for (int64_t c = 0; c < d; ++c) {
                        double acc = 0.0;
                        for (int64_t f = 0; f < cfg.d_ff; ++f) {
                            acc += static_cast<double>(d1p[m * cfg.d_ff + f]) * static_cast<double>(pre.w1.at_flat(f * d + c));
                            acc += static_cast<double>(d3p[m * cfg.d_ff + f]) * static_cast<double>(pre.w3.at_flat(f * d + c));
                        }
                        xp[m * d + c] = static_cast<float>(acc);
                    }
                }
            }

            // d_ln2_w
            {
                const float* dxp = d_xn2.ptr<float>();
                const float* xinp = cache.pre_x_after_attn.ptr<float>();
                for (int64_t c = 0; c < d; ++c) {
                    double acc = 0.0;
                    for (int64_t m = 0; m < M; ++m)
                        acc += static_cast<double>(dxp[m * d + c]) * static_cast<double>(xinp[m * d + c]);
                    g_pre_ln2.set_flat(c, g_pre_ln2.at_flat(c) + acc);
                }
            }

            // d_x_after_attn = d_x_input (residual) + d_xn2 * ln2_w (approx)
            mt::Tensor d_x_after_attn = d_x_input;

            // Backprop attention (simplified)
            // Skip full attention backward; route d_x_after_attn to x_input via residual
            // d_x_input = d_x_after_attn (through residual x_after_attn = x + proj_out)

            // tok_emb scatter-add: d_tok_emb[idx] += d_x_input
            for (int64_t m = 0; m < M; ++m) {
                int32_t id = cache.idx_flat[static_cast<size_t>(m)];
                if (id >= 0 && id < V) {
                    for (int64_t c = 0; c < d; ++c) {
                        g_tok_emb.set_flat(id * d + c,
                            g_tok_emb.at_flat(id * d + c) + d_x_after_attn.at_flat(m * d + c));
                    }
                }
            }
        }
    }

    // Also handle step 0's dH for adapter gradient on x
    // (the x gradient from adapter at step 0 is handled above via d_x_input)

    return grads;
}

// ---- Convenience wrapper ----

float compute_loss_and_grads(const Config& cfg, Coder& coder,
                             const mt::Tensor& idx,
                             const mt::Tensor& targets,
                             int64_t pos_offset,
                             std::vector<model::KVCache>& caches,
                             Gradients& grads_out) {
    BackwardCache cache = forward_with_cache(cfg, coder, idx, targets, caches, pos_offset);
    if (cache.logits_per_step.empty()) {
        return 0.0f;
    }
    mt::Tensor d_logits;
    float loss = cross_entropy_loss(cache, d_logits);
    grads_out = backward(cfg, coder, cache, d_logits, caches);
    return loss;
}

}  // namespace minagi::backward
