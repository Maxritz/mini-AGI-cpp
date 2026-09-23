// tests/test_model.cpp
#include "model.hpp"
#include "tensor.hpp"
#include "golden_model.inc"
#include <cstdio>
#include <cmath>
#include <vector>
#include <string>
#include <utility>
#include <cstdint>

static int failures = 0;

static void check(const char* name, bool cond) {
    if (cond) {
        std::printf("ok %s\n", name);
    } else {
        std::printf("FAIL %s\n", name);
        ++failures;
    }
}

static mt::Tensor make_tensor_from_float(const float* data, int64_t d0, int64_t d1) {
    mt::Shape s;
    s.rank = 2;
    s.d[0] = d0;
    s.d[1] = d1;
    mt::Tensor t = mt::make(s, mt::DType::FP32, 0.0f);
    for (int64_t i = 0; i < d0 * d1; ++i) t.set_flat(i, data[i]);
    return t;
}

static mt::Tensor make_tensor_from_float_1d(const float* data, int64_t d0) {
    mt::Shape s;
    s.rank = 1;
    s.d[0] = d0;
    mt::Tensor t = mt::make(s, mt::DType::FP32, 0.0f);
    for (int64_t i = 0; i < d0; ++i) t.set_flat(i, data[i]);
    return t;
}

static mt::Tensor make_idx_tensor(const int32_t* data, int64_t d0, int64_t d1) {
    mt::Shape s;
    s.rank = 2;
    s.d[0] = d0;
    s.d[1] = d1;
    mt::Tensor t = mt::make(s, mt::DType::FP32, 0.0f);
    for (int64_t i = 0; i < d0 * d1; ++i) t.set_flat(i, static_cast<float>(data[i]));
    return t;
}

int main() {
    model::Config c;
    c.vocab_size = 33;
    c.n_layer = 2;
    c.n_head = 4;
    c.d_model = 16;
    c.block = 16;
    c.d_ff = 32;
    c.rope_theta = 10000.0;
    c.tie_embeddings = true;

    check("config", c.vocab_size == 33 && c.n_layer == 2 && c.n_head == 4 &&
                     c.d_model == 16 && c.block == 16 && c.d_ff == 32 &&
                     c.rope_theta == 10000.0 && c.tie_embeddings == true);

    std::vector<std::pair<std::string, mt::Tensor>> sd;
    sd.emplace_back("tok_emb.weight", make_tensor_from_float(golden::TOK_EMB, 33, 16));
    for (int i = 0; i < 2; ++i) {
        std::string b = "blocks." + std::to_string(i) + ".";
        sd.emplace_back(b + "ln1.weight", make_tensor_from_float_1d(
            i == 0 ? golden::L0_LN1 : golden::L1_LN1, 16));
        sd.emplace_back(b + "attn.qkv.weight", make_tensor_from_float(
            i == 0 ? golden::L0_QKV : golden::L1_QKV, 48, 16));
        sd.emplace_back(b + "attn.proj.weight", make_tensor_from_float(
            i == 0 ? golden::L0_PROJ : golden::L1_PROJ, 16, 16));
        sd.emplace_back(b + "ln2.weight", make_tensor_from_float_1d(
            i == 0 ? golden::L0_LN2 : golden::L1_LN2, 16));
        sd.emplace_back(b + "mlp.w1.weight", make_tensor_from_float(
            i == 0 ? golden::L0_W1 : golden::L1_W1, 32, 16));
        sd.emplace_back(b + "mlp.w3.weight", make_tensor_from_float(
            i == 0 ? golden::L0_W3 : golden::L1_W3, 32, 16));
        sd.emplace_back(b + "mlp.w2.weight", make_tensor_from_float(
            i == 0 ? golden::L0_W2 : golden::L1_W2, 16, 32));
    }
    sd.emplace_back("ln_f.weight", make_tensor_from_float_1d(golden::LN_F, 16));

    model::State s;
    bool ok = model::load_state(c, sd, s);
    check("load_state", ok);
    check("load_state_tok_emb", s.tok_emb.shape.rank == 2 && s.tok_emb.shape.d[0] == 33 && s.tok_emb.shape.d[1] == 16);
    check("load_state_ln1_count", s.ln1.size() == 2);
    check("load_state_qkv_count", s.qkv.size() == 2);
    check("load_state_proj_count", s.proj.size() == 2);
    check("load_state_ln2_count", s.ln2.size() == 2);
    check("load_state_w1_count", s.w1.size() == 2);
    check("load_state_w3_count", s.w3.size() == 2);
    check("load_state_w2_count", s.w2.size() == 2);
    check("load_state_ln_f", s.ln_f.shape.rank == 1 && s.ln_f.shape.d[0] == 16);
    check("load_state_head_ignored", s.head.shape.rank == 0);

    mt::Tensor cos, sin;
    model::build_rope(16, 4, 10000.0, cos, sin);
    bool rope_ok = true;
    for (int t = 0; t < 16 && rope_ok; ++t)
        for (int i = 0; i < 2; ++i) {
            if (std::fabs(cos.at_flat(t * 2 + i) - golden::COS[t][i]) > 1e-6) rope_ok = false;
            if (std::fabs(sin.at_flat(t * 2 + i) - golden::SIN[t][i]) > 1e-6) rope_ok = false;
        }
    check("build_rope", rope_ok);

    std::vector<model::KVCache> caches(2);
    int32_t tokens0[4] = {5, 3, 17, 29};
    mt::Tensor idx0 = make_idx_tensor(tokens0, 1, 4);
    mt::Tensor logits1 = model::forward(s, c, idx0, caches, 0);
    bool l1_ok = logits1.shape.rank == 3 && logits1.shape.d[0] == 1 &&
                 logits1.shape.d[1] == 4 && logits1.shape.d[2] == 33;
    double max_diff1 = 0.0;
    if (l1_ok) {
        for (int t = 0; t < 4; ++t)
            for (int v = 0; v < 33; ++v) {
                double d = std::fabs(logits1.at_flat(t * 33 + v) - golden::G_LOGITS1[t][v]);
                if (d > max_diff1) max_diff1 = d;
            }
    }
    check("forward_logits1_shape", l1_ok);
    check("forward_logits1_values", max_diff1 <= 1e-4);
    check("forward_cache_active", caches[0].active && caches[1].active);
    bool cache_shape_ok = true;
    for (int i = 0; i < 2; ++i) {
        if (caches[i].k.shape.rank != 4 || caches[i].k.shape.d[0] != 1 ||
            caches[i].k.shape.d[1] != 4 || caches[i].k.shape.d[2] != 4 ||
            caches[i].k.shape.d[3] != 4) cache_shape_ok = false;
        if (caches[i].v.shape.rank != 4 || caches[i].v.shape.d[0] != 1 ||
            caches[i].v.shape.d[1] != 4 || caches[i].v.shape.d[2] != 4 ||
            caches[i].v.shape.d[3] != 4) cache_shape_ok = false;
    }
    check("forward_cache_shape", cache_shape_ok);

    int32_t tokens1[3] = {8, 12, 1};
    mt::Tensor idx1 = make_idx_tensor(tokens1, 1, 3);
    mt::Tensor logits2 = model::forward(s, c, idx1, caches, 4);
    bool l2_ok = logits2.shape.rank == 3 && logits2.shape.d[0] == 1 &&
                 logits2.shape.d[1] == 3 && logits2.shape.d[2] == 33;
    double max_diff2 = 0.0;
    if (l2_ok) {
        for (int t = 0; t < 3; ++t)
            for (int v = 0; v < 33; ++v) {
                double d = std::fabs(logits2.at_flat(t * 33 + v) - golden::G_LOGITS2[t][v]);
                if (d > max_diff2) max_diff2 = d;
            }
    }
    check("forward_logits2_shape", l2_ok);
    check("forward_logits2_values", max_diff2 <= 1e-4);
    bool cache7_ok = true;
    for (int i = 0; i < 2; ++i) {
        if (caches[i].k.shape.d[2] != 7) cache7_ok = false;
        if (caches[i].v.shape.d[2] != 7) cache7_ok = false;
    }
    check("forward_cache_Ttot7", cache7_ok);

    int32_t tokens_oob[2] = {1, 2};
    mt::Tensor idx_oob = make_idx_tensor(tokens_oob, 1, 2);
    mt::Tensor logits_oob = model::forward(s, c, idx_oob, caches, 31);
    check("forward_oob_empty", logits_oob.shape.rank == 0);

    if (failures == 0) {
        std::printf("ALL_TESTS_PASSED\n");
        return 0;
    } else {
        std::printf("SOME_TESTS_FAILED\n");
        return 1;
    }
}
