// src/model.hpp
#pragma once
#include "tensor.hpp"
#include <vector>
#include <string>
#include <utility>
#include <cstdint>

namespace model {

struct Config {
    int vocab_size = 8192;
    int n_layer = 8;
    int n_head = 8;
    int d_model = 512;
    int block = 512;
    int d_ff = 1408;
    double rope_theta = 10000.0;
    bool tie_embeddings = true;
};

struct State {
    mt::Tensor tok_emb;
    mt::Tensor head;
    std::vector<mt::Tensor> ln1, qkv, proj, ln2, w1, w3, w2;
    mt::Tensor ln_f;
};

struct KVCache {
    bool active = false;
    mt::Tensor k;
    mt::Tensor v;
};

bool load_state(const Config& c, const std::vector<std::pair<std::string, mt::Tensor>>& sd,
                State& out);

void build_rope(int block, int head_dim, double theta, mt::Tensor& cos, mt::Tensor& sin);

mt::Tensor forward(const State& s, const Config& c, const mt::Tensor& idx,
                   std::vector<KVCache>& caches, int64_t pos_offset);

}  // namespace model
