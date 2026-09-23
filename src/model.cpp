// src/model.cpp
#include "model.hpp"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <stdexcept>

namespace model {

static inline int64_t I64(int v) { return static_cast<int64_t>(v); }

static mt::Tensor make_zeros_fp32(const mt::Shape& shp) {
    mt::Tensor t = mt::make_zeros(shp, mt::DType::FP32);
    return t;
}

static mt::Shape shape2(int64_t r0, int64_t r1) {
    mt::Shape s;
    s.rank = 2;
    s.d[0] = r0;
    s.d[1] = r1;
    return s;
}

static mt::Shape shape4(int64_t d0, int64_t d1, int64_t d2, int64_t d3) {
    mt::Shape s;
    s.rank = 4;
    s.d[0] = d0;
    s.d[1] = d1;
    s.d[2] = d2;
    s.d[3] = d3;
    return s;
}

static mt::Tensor transpose_2d(const mt::Tensor& a) {
    int64_t M = a.shape.d[0];
    int64_t N = a.shape.d[1];
    mt::Tensor out = make_zeros_fp32(shape2(N, M));
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            out.set_flat(j * M + i, a.at_flat(i * N + j));
        }
    }
    return out;
}

static mt::Tensor slice_cols(const mt::Tensor& a, int64_t start, int64_t count) {
    int64_t M = a.shape.d[0];
    int64_t N = a.shape.d[1];
    mt::Tensor out = make_zeros_fp32(shape2(M, count));
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < count; ++j) {
            out.set_flat(i * count + j, a.at_flat(i * N + (start + j)));
        }
    }
    return out;
}

static mt::Tensor embedding_gather(const mt::Tensor& emb, const mt::Tensor& idx) {
    int64_t B = idx.shape.d[0];
    int64_t T = idx.shape.d[1];
    int64_t V = emb.shape.d[0];
    int64_t C = emb.shape.d[1];
    mt::Tensor out = make_zeros_fp32(shape2(B * T, C));
    for (int64_t bt = 0; bt < B * T; ++bt) {
        int32_t id = static_cast<int32_t>(idx.at_flat(bt));
        if (id < 0 || id >= V) {
            for (int64_t j = 0; j < C; ++j) out.set_flat(bt * C + j, 0.0);
        } else {
            for (int64_t j = 0; j < C; ++j) out.set_flat(bt * C + j, emb.at_flat(id * C + j));
        }
    }
    return out;
}

static mt::Tensor reshape_BTHD(const mt::Tensor& x, int64_t B, int64_t T, int64_t H, int64_t hd) {
    mt::Tensor out = make_zeros_fp32(shape4(B, H, T, hd));
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t t = 0; t < T; ++t) {
            for (int64_t h = 0; h < H; ++h) {
                for (int64_t d = 0; d < hd; ++d) {
                    int64_t src = (b * T + t) * (H * hd) + h * hd + d;
                    int64_t dst = ((b * H + h) * T + t) * hd + d;
                    out.set_flat(dst, x.at_flat(src));
                }
            }
        }
    }
    return out;
}

static mt::Tensor transpose_BHTD_to_BTHD(const mt::Tensor& x, int64_t B, int64_t H, int64_t T, int64_t hd) {
    mt::Tensor out = make_zeros_fp32(shape4(B, T, H, hd));
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t h = 0; h < H; ++h) {
            for (int64_t t = 0; t < T; ++t) {
                for (int64_t d = 0; d < hd; ++d) {
                    int64_t src = ((b * H + h) * T + t) * hd + d;
                    int64_t dst = ((b * T + t) * H + h) * hd + d;
                    out.set_flat(dst, x.at_flat(src));
                }
            }
        }
    }
    return out;
}

static mt::Tensor apply_rope(const mt::Tensor& x, int64_t B, int64_t H, int64_t T, int64_t hd,
                             const mt::Tensor& cos, const mt::Tensor& sin, int64_t pos_offset) {
    int64_t half = hd / 2;
    mt::Tensor out = make_zeros_fp32(shape4(B, H, T, hd));
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t h = 0; h < H; ++h) {
            for (int64_t t = 0; t < T; ++t) {
                int64_t row = pos_offset + t;
                for (int64_t d = 0; d < half; ++d) {
                    float c = static_cast<float>(cos.at_flat(row * half + d));
                    float s = static_cast<float>(sin.at_flat(row * half + d));
                    float x1 = static_cast<float>(x.at_flat(((b * H + h) * T + t) * hd + 2 * d));
                    float x2 = static_cast<float>(x.at_flat(((b * H + h) * T + t) * hd + 2 * d + 1));
                    out.set_flat(((b * H + h) * T + t) * hd + 2 * d, x1 * c - x2 * s);
                    out.set_flat(((b * H + h) * T + t) * hd + 2 * d + 1, x1 * s + x2 * c);
                }
            }
        }
    }
    return out;
}

static mt::Tensor attn(const mt::Tensor& x, const mt::Tensor& qkv_w, const mt::Tensor& proj_w,
                       const mt::Tensor& ln1_w, const mt::Tensor& cos, const mt::Tensor& sin,
                       int64_t B, int64_t T, int64_t H, int64_t hd, int64_t C,
                       KVCache& cache, int64_t pos_offset) {
    mt::Tensor xn = mt::rms_norm(x, ln1_w, 1e-6f);
    mt::Tensor qkv = mt::matmul(xn, transpose_2d(qkv_w));
    mt::Tensor q = slice_cols(qkv, 0, C);
    mt::Tensor k = slice_cols(qkv, C, C);
    mt::Tensor v = slice_cols(qkv, 2 * C, C);

    mt::Tensor qh = reshape_BTHD(q, B, T, H, hd);
    mt::Tensor kh = reshape_BTHD(k, B, T, H, hd);
    mt::Tensor vh = reshape_BTHD(v, B, T, H, hd);

    qh = apply_rope(qh, B, H, T, hd, cos, sin, pos_offset);
    kh = apply_rope(kh, B, H, T, hd, cos, sin, pos_offset);

    if (cache.active && cache.k.shape.d[2] > 0) {
        int64_t cached_T = cache.k.shape.d[2];
        mt::Tensor kh_flat = make_zeros_fp32(shape2(B * H * T, hd));
        for (int64_t i = 0; i < B * H * T; ++i)
            for (int64_t d = 0; d < hd; ++d)
                kh_flat.set_flat(i * hd + d, kh.at_flat(i * hd + d));
        mt::Tensor vh_flat = make_zeros_fp32(shape2(B * H * T, hd));
        for (int64_t i = 0; i < B * H * T; ++i)
            for (int64_t d = 0; d < hd; ++d)
                vh_flat.set_flat(i * hd + d, vh.at_flat(i * hd + d));

        mt::Tensor kh_cached_flat = make_zeros_fp32(shape2(B * H * cached_T, hd));
        for (int64_t i = 0; i < B * H * cached_T; ++i)
            for (int64_t d = 0; d < hd; ++d)
                kh_cached_flat.set_flat(i * hd + d, cache.k.at_flat(i * hd + d));
        mt::Tensor vh_cached_flat = make_zeros_fp32(shape2(B * H * cached_T, hd));
        for (int64_t i = 0; i < B * H * cached_T; ++i)
            for (int64_t d = 0; d < hd; ++d)
                vh_cached_flat.set_flat(i * hd + d, cache.v.at_flat(i * hd + d));

        int64_t kv_len = cached_T + T;
        mt::Tensor k_concat = make_zeros_fp32(shape2(B * H * kv_len, hd));
        mt::Tensor v_concat = make_zeros_fp32(shape2(B * H * kv_len, hd));
        for (int64_t bh = 0; bh < B * H; ++bh) {
            for (int64_t j = 0; j < cached_T; ++j)
                for (int64_t d = 0; d < hd; ++d) {
                    k_concat.set_flat((bh * kv_len + j) * hd + d, kh_cached_flat.at_flat((bh * cached_T + j) * hd + d));
                    v_concat.set_flat((bh * kv_len + j) * hd + d, vh_cached_flat.at_flat((bh * cached_T + j) * hd + d));
                }
            for (int64_t jn = 0; jn < T; ++jn)
                for (int64_t d = 0; d < hd; ++d) {
                    k_concat.set_flat((bh * kv_len + cached_T + jn) * hd + d, kh_flat.at_flat((bh * T + jn) * hd + d));
                    v_concat.set_flat((bh * kv_len + cached_T + jn) * hd + d, vh_flat.at_flat((bh * T + jn) * hd + d));
                }
        }

        mt::Tensor scores = make_zeros_fp32(shape2(B * H * T, kv_len));
        float inv_sqrt = 1.0f / std::sqrt(static_cast<float>(hd));
        for (int64_t bh = 0; bh < B * H; ++bh) {
            for (int64_t i = 0; i < T; ++i) {
                for (int64_t j = 0; j < kv_len; ++j) {
                    float dot = 0.0f;
                    for (int64_t d = 0; d < hd; ++d) {
                        dot += static_cast<float>(qh.at_flat((bh * T + i) * hd + d)) *
                               static_cast<float>(k_concat.at_flat((bh * kv_len + j) * hd + d));
                    }
                    dot *= inv_sqrt;
                    int64_t P = kv_len - T;
                    if (j > i + P) dot = -1e30f;
                    scores.set_flat(bh * kv_len * T + i * kv_len + j, dot);
                }
            }
        }

        mt::Tensor sm = mt::softmax(scores, 1);
        mt::Tensor out_h = make_zeros_fp32(shape4(B, H, T, hd));
        for (int64_t bh = 0; bh < B * H; ++bh) {
            for (int64_t t = 0; t < T; ++t) {
                for (int64_t d = 0; d < hd; ++d) {
                    float acc = 0.0f;
                    for (int64_t j = 0; j < kv_len; ++j) {
                        float a = static_cast<float>(sm.at_flat((bh * T + t) * kv_len + j));
                        acc += a * static_cast<float>(v_concat.at_flat((bh * kv_len + j) * hd + d));
                    }
                    out_h.set_flat((bh * T + t) * hd + d, acc);
                }
            }
        }
        mt::Tensor out_bthd = transpose_BHTD_to_BTHD(out_h, B, H, T, hd);
        mt::Tensor out_flat = make_zeros_fp32(shape2(B * T, C));
        for (int64_t bt = 0; bt < B * T; ++bt) {
            for (int64_t h = 0; h < H; ++h) {
                for (int64_t d = 0; d < hd; ++d) {
                    out_flat.set_flat(bt * C + h * hd + d, out_bthd.at_flat((bt * H + h) * hd + d));
                }
            }
        }

        cache.k = make_zeros_fp32(shape4(B, H, kv_len, hd));
        cache.v = make_zeros_fp32(shape4(B, H, kv_len, hd));
        for (int64_t i = 0; i < B * H * kv_len; ++i)
            for (int64_t d = 0; d < hd; ++d) {
                cache.k.set_flat(i * hd + d, k_concat.at_flat(i * hd + d));
                cache.v.set_flat(i * hd + d, v_concat.at_flat(i * hd + d));
            }
        cache.active = true;

        mt::Tensor proj_out = mt::matmul(out_flat, transpose_2d(proj_w));
        return proj_out;
    } else {
        int64_t kv_len = T;
        mt::Tensor scores = make_zeros_fp32(shape2(B * H * T, kv_len));
        float inv_sqrt = 1.0f / std::sqrt(static_cast<float>(hd));
        for (int64_t bh = 0; bh < B * H; ++bh) {
            for (int64_t i = 0; i < T; ++i) {
                for (int64_t j = 0; j < kv_len; ++j) {
                    float dot = 0.0f;
                    for (int64_t d = 0; d < hd; ++d) {
                        dot += static_cast<float>(qh.at_flat((bh * T + i) * hd + d)) *
                               static_cast<float>(kh.at_flat((bh * T + j) * hd + d));
                    }
                    dot *= inv_sqrt;
                    if (j > i) dot = -1e30f;
                    scores.set_flat(bh * kv_len * T + i * kv_len + j, dot);
                }
            }
        }

        mt::Tensor sm = mt::softmax(scores, 1);
        mt::Tensor out_h = make_zeros_fp32(shape4(B, H, T, hd));
        for (int64_t bh = 0; bh < B * H; ++bh) {
            for (int64_t t = 0; t < T; ++t) {
                for (int64_t d = 0; d < hd; ++d) {
                    float acc = 0.0f;
                    for (int64_t j = 0; j < kv_len; ++j) {
                        float a = static_cast<float>(sm.at_flat((bh * T + t) * kv_len + j));
                        acc += a * static_cast<float>(vh.at_flat((bh * T + j) * hd + d));
                    }
                    out_h.set_flat((bh * T + t) * hd + d, acc);
                }
            }
        }
        mt::Tensor out_bthd = transpose_BHTD_to_BTHD(out_h, B, H, T, hd);
        mt::Tensor out_flat = make_zeros_fp32(shape2(B * T, C));
        for (int64_t bt = 0; bt < B * T; ++bt)
            for (int64_t h = 0; h < H; ++h)
                for (int64_t d = 0; d < hd; ++d)
                    out_flat.set_flat(bt * C + h * hd + d, out_bthd.at_flat((bt * H + h) * hd + d));

        cache.k = make_zeros_fp32(shape4(B, H, T, hd));
        cache.v = make_zeros_fp32(shape4(B, H, T, hd));
        for (int64_t i = 0; i < B * H * T; ++i)
            for (int64_t d = 0; d < hd; ++d) {
                cache.k.set_flat(i * hd + d, kh.at_flat(i * hd + d));
                cache.v.set_flat(i * hd + d, vh.at_flat(i * hd + d));
            }
        cache.active = true;

        mt::Tensor proj_out = mt::matmul(out_flat, transpose_2d(proj_w));
        return proj_out;
    }
}

static mt::Tensor silu_elem(const mt::Tensor& a) {
    mt::Tensor out = mt::make_like(a);
    const int64_t n = a.shape.numel();
    for (int64_t i = 0; i < n; ++i) {
        const float v = static_cast<float>(a.atf(i));
        out.set_flat(i, static_cast<double>(v * (1.0f / (1.0f + std::exp(-v)))));
    }
    return out;
}

static mt::Tensor mlp(const mt::Tensor& x, const mt::Tensor& w1, const mt::Tensor& w3,
                      const mt::Tensor& w2, const mt::Tensor& ln2_w) {
    mt::Tensor xn = mt::rms_norm(x, ln2_w, 1e-6f);
    mt::Tensor h1 = mt::matmul(xn, transpose_2d(w1));
    mt::Tensor h3 = mt::matmul(xn, transpose_2d(w3));
    mt::Tensor g = silu_elem(h1);
    mt::Tensor prod = make_zeros_fp32(shape2(g.shape.d[0], g.shape.d[1]));
    for (int64_t i = 0; i < g.shape.d[0]; ++i)
        for (int64_t j = 0; j < g.shape.d[1]; ++j)
            prod.set_flat(i * g.shape.d[1] + j, g.at_flat(i * g.shape.d[1] + j) * h3.at_flat(i * g.shape.d[1] + j));
    mt::Tensor out = mt::matmul(prod, transpose_2d(w2));
    return out;
}

bool load_state(const Config& c, const std::vector<std::pair<std::string, mt::Tensor>>& sd,
                State& out) {
    out.ln1.clear(); out.qkv.clear(); out.proj.clear(); out.ln2.clear();
    out.w1.clear(); out.w3.clear(); out.w2.clear();
    out.tok_emb = mt::Tensor();
    out.head = mt::Tensor();
    out.ln_f = mt::Tensor();

    auto find = [&](const std::string& key) -> const mt::Tensor* {
        for (auto& kv : sd) if (kv.first == key) return &kv.second;
        return nullptr;
    };

    auto get = [&](const std::string& key, int64_t d0, int64_t d1) -> mt::Tensor {
        const mt::Tensor* t = find(key);
        if (!t) return mt::Tensor();
        if (t->dtype != mt::DType::FP32) return mt::to_fp32(*t);
        if (t->shape.rank != 2 || t->shape.d[0] != d0 || t->shape.d[1] != d1) return mt::Tensor();
        return *t;
    };

    auto get1d = [&](const std::string& key, int64_t d0) -> mt::Tensor {
        const mt::Tensor* t = find(key);
        if (!t) return mt::Tensor();
        if (t->dtype != mt::DType::FP32) return mt::to_fp32(*t);
        if (t->shape.rank != 1 || t->shape.d[0] != d0) return mt::Tensor();
        return *t;
    };

    out.tok_emb = get("tok_emb.weight", c.vocab_size, c.d_model);
    if (out.tok_emb.shape.rank == 0) return false;

    for (int i = 0; i < c.n_layer; ++i) {
        std::string b = "blocks." + std::to_string(i) + ".";
        mt::Tensor l1 = get1d(b + "ln1.weight", c.d_model);
        mt::Tensor qk = get(b + "attn.qkv.weight", 3 * c.d_model, c.d_model);
        mt::Tensor pj = get(b + "attn.proj.weight", c.d_model, c.d_model);
        mt::Tensor l2 = get1d(b + "ln2.weight", c.d_model);
        mt::Tensor w1t = get(b + "mlp.w1.weight", c.d_ff, c.d_model);
        mt::Tensor w3t = get(b + "mlp.w3.weight", c.d_ff, c.d_model);
        mt::Tensor w2t = get(b + "mlp.w2.weight", c.d_model, c.d_ff);
        if (l1.shape.rank == 0 || qk.shape.rank == 0 || pj.shape.rank == 0 ||
            l2.shape.rank == 0 || w1t.shape.rank == 0 || w3t.shape.rank == 0 || w2t.shape.rank == 0)
            return false;
        out.ln1.push_back(l1);
        out.qkv.push_back(qk);
        out.proj.push_back(pj);
        out.ln2.push_back(l2);
        out.w1.push_back(w1t);
        out.w3.push_back(w3t);
        out.w2.push_back(w2t);
    }

    out.ln_f = get1d("ln_f.weight", c.d_model);
    if (out.ln_f.shape.rank == 0) return false;

    if (!c.tie_embeddings) {
        out.head = get("head.weight", c.vocab_size, c.d_model);
        if (out.head.shape.rank == 0) return false;
    }
    return true;
}

void build_rope(int block, int head_dim, double theta, mt::Tensor& cos, mt::Tensor& sin) {
    int half = head_dim / 2;
    cos = make_zeros_fp32(shape2(block, half));
    sin = make_zeros_fp32(shape2(block, half));
    for (int i = 0; i < half; ++i) {
        double inv = std::pow(theta, -(2.0 * i / static_cast<double>(head_dim)));
        for (int t = 0; t < block; ++t) {
            double freq = t * inv;
            cos.set_flat(t * half + i, std::cos(freq));
            sin.set_flat(t * half + i, std::sin(freq));
        }
    }
}

mt::Tensor forward(const State& s, const Config& c, const mt::Tensor& idx,
                   std::vector<KVCache>& caches, int64_t pos_offset) {
    int64_t B = idx.shape.d[0];
    int64_t T = idx.shape.d[1];
    int64_t C = c.d_model;
    int64_t H = c.n_head;
    int64_t hd = C / H;
    int64_t V = c.vocab_size;

    if (pos_offset + T > c.block) {
        mt::Shape empty;
        empty.rank = 0;
        return mt::make(empty, mt::DType::FP32, 0.0f);
    }

    mt::Tensor x = embedding_gather(s.tok_emb, idx);
    mt::Tensor cos, sin;
    build_rope(static_cast<int>(c.block), static_cast<int>(hd), c.rope_theta, cos, sin);

    for (int i = 0; i < c.n_layer; ++i) {
        KVCache& cache = caches[i];
        mt::Tensor a = attn(x, s.qkv[i], s.proj[i], s.ln1[i], cos, sin, B, T, H, hd, C, cache, pos_offset);
        for (int64_t j = 0; j < B * T * C; ++j)
            x.set_flat(j, x.at_flat(j) + a.at_flat(j));
        mt::Tensor m = mlp(x, s.w1[i], s.w3[i], s.w2[i], s.ln2[i]);
        for (int64_t j = 0; j < B * T * C; ++j)
            x.set_flat(j, x.at_flat(j) + m.at_flat(j));
    }

    mt::Tensor y = mt::rms_norm(x, s.ln_f, 1e-6f);
    mt::Tensor head_w = c.tie_embeddings ? s.tok_emb : s.head;
    mt::Tensor logits_2d = mt::matmul(y, transpose_2d(head_w));  // [B*T, V]
    mt::Shape out_shape;
    out_shape.rank = 3;
    out_shape.d[0] = B;
    out_shape.d[1] = T;
    out_shape.d[2] = V;
    mt::Tensor logits = mt::make(out_shape, mt::DType::FP32, 0.0f);
    for (int64_t i = 0; i < B * T * V; ++i)
        logits.set_flat(i, logits_2d.at_flat(i));
    return logits;
}

}  // namespace model
