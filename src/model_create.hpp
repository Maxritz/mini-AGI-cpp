// src/model_create.hpp
// Model creation: build a fresh tensor map from config, matching the exact
// key naming produced by minagi/recur.py RecurCoder.__init__ +
// minagi/pool.py SharedPool.__init__. Weights are initialized with
// PyTorch-equivalent RNG (normal 0.02, zeros, eye, constant) and returned
// as a state dict ready for store::save().
#pragma once
#include "tensor.hpp"
#include "init.hpp"
#include "recur.hpp"
#include "mininpz.hpp"
#include <map>
#include <string>
#include <vector>

namespace minagi {

// Build the full state-dict tensor map for a fresh model:
//   - tok_emb (tied to head if tie_embeddings)
//   - prelude.N.{ln1,attn.qkv,attn.proj,ln2,mlp.w1,mlp.w3,mlp.w2}
//   - recur.N.{ln1,attn.qkv,attn.proj,ln2,mlp.router.weight,mlp.depth_emb}
//   - coda.N.{ln1,attn.qkv,attn.proj,ln2,mlp.router.weight,mlp.depth_emb}
//   - adapter.weight [I|I]
//   - ln_f.weight = 1.0
//   - head.weight (if not tied)
//   - halt.weight *0.01, halt.bias = -2.0
//   - pool.gate = 1.0
//   - pool.segment_router.weight ~ normal(0, 0.02)
//   - pool.experts.{N}.w1/w3/w2 per expert
//
// Returns std::map of name -> Tensor. Convert to mininpz::Array via
// tensors_to_state for store::save().
std::map<std::string, mt::Tensor> build_model(const Config& cfg, PcgRng& rng);

// Convert tensor map to Array state map for store::save.
std::map<std::string, mininpz::Array> tensors_to_state(
    const std::map<std::string, mt::Tensor>& model_tensors);

}  // namespace minagi
