// src/recur.hpp
// Latent recurrence (pondering + greedy decode on the mt tensor core): the
// section-6 RecurCoder forward, the section-7/8 run loop, and the weights
// directory loader. Implements the inference path of minagi/recur.py; the
// pooled MLP and the expert pool live in paged.hpp. Note this file defines
// `minagi::Config` (section 8/9) -- the on-disk config used by the golden
// run. Do NOT co-include src/config.hpp (a different minagi::Config for the
// tokenizer) in any translation unit that also uses this one.
#pragma once
#include "tensor.hpp"
#include "paged.hpp"
#include "model.hpp"   // model::KVCache, model::build_rope
#include "mininpz.hpp" // mini::JsonValue
#include "store.hpp"   // store::Manifest, store::load, store::expert_dir
#include <cstdint>
#include <string>
#include <vector>

namespace minagi {

struct Config {
  // model
  int vocab_size = 265;
  int n_layer = 8;        // vestigial (python keeps it); block count is n_prelude+n_recur+n_coda
  int n_head = 3;
  int d_model = 24;
  int block = 1024;
  int d_ff = 48;
  double rope_theta = 10000.0;
  bool tie_embeddings = true;
  // recur / pool
  bool use_pool = true;
  int pool_experts = 6;
  int pool_d_ff = 32;
  int pool_depth = 1;
  int pool_top_k = 2;
  double pool_capacity_factor = 1.5;
  int pool_max = 6;
  double pool_aux = 0.01;
  int n_prelude = 1;
  int n_recur = 2;
  int n_coda = 0;
  int max_steps = 5;
  int min_steps = 1;
  double train_steps_mean = 0.0;
  int bptt_window = 4;
  double ponder_beta = 0.01;
  double halt_prior = 0.4;
  double halt_thresh = 0.9;
  // run-time (pool_resident from cfg; read_only/pool_ram from the manifest
  // top-level -- set by minagi::load, not by from_manifest)
  int pool_resident = 3;
  int pool_ram = 4;
  bool read_only = true;
  double explore = 0.15;
  double margin = 0.10;
  int dwell = 4;

  bool from_manifest(const mini::JsonValue& cfg_obj); // flat manifest "cfg" keys; missing -> defaults stay
};

struct StepOut {
  mt::Tensor logits;   // [1,T,V] fp32; rank-0 EMPTY when the window is exceeded
  mt::Tensor halt_row; // [1,T] fp32 (integral values, 1-based)
  mt::Tensor halt_p;    // [1,T] fp32 = (1-cum) at the halt row
};

class Coder {
public:
  Coder() = default;
  explicit Coder(const Config& cfg);
  bool loaded() const;
  // Maps the flat python-keyed sd onto the coder: tok_emb, prelude/recur dense
  // tensors, adapter, ln_f, head (TIED -- if both head.weight and tok_emb.weight
  // are present they must match bit-for-bit; keep one), halt weight/bias; the
  // sites' router.weight + depth_emb; and pool.gate / pool.segment_router.weight
  // into the pool. Returns false on a missing or wrongly-shaped required key.
  bool load_weights(std::vector<std::pair<std::string, mt::Tensor>>& sd,
                    paged::PagedPool& pool);
  int n_slots() const;                                   // n_prelude + max_steps*(n_recur+n_codo)
  std::vector<model::KVCache> empty_caches() const;      // n_slots fresh caches
  StepOut forward(const mt::Tensor& idx, std::vector<model::KVCache>& caches,
                  int64_t pos_offset);                   // section 6, B=1; observes the summary inside
  int begin_segment();                                   // paged: swap_to(choose()) -> loads
  void end_segment(const mt::Tensor& x);                 // paged: pool.observe(mean); kept for parity (the driver never calls it)
  const Config& cfg() const;
  paged::PagedPool* pool();
  const paged::PagedPool* pool() const;
  void set_pool(paged::PagedPool* p);            // used by minagi::load
 private:
  Config cfg_;
  bool loaded_ = false;
  paged::PagedPool* pool_ = nullptr; // external; set by load_weights

  // weights
  mt::Tensor tok_emb_;                 // [V, d_model]
  mt::Tensor adapter_w_;               // [d_model, 2*d_model] = [I|I]
  mt::Tensor ln_f_w_;                  // [d_model]
  mt::Tensor halt_w_;                  // [1, d_model]
  mt::Tensor halt_b_;                  // [1]
  struct Block {
    mt::Tensor ln1;                    // [d]
    mt::Tensor qkv;                    // [3d, d]
    mt::Tensor proj;                   // [d, d]
    mt::Tensor ln2;                    // [d]
    // exactly one of the two MLP kinds is used per block kind:
    mt::Tensor w1, w3, w2;             // dense SwiGLU (prelude.0.*)
    mt::Tensor router;                 // [n_experts, d] (recur.*.mlp.router.weight)
    mt::Tensor depth_emb;              // [d]       (recur.*.mlp.depth_emb)
  };
  // blocks_[0] = prelude, blocks_[1..n_prelude+n_recur+n_coda-1] = recurrent sites.
  std::vector<Block> blocks_;
  // cached rope tables (built once, reused across forwards).
  mt::Tensor rope_cos_;
  mt::Tensor rope_sin_;

  // cached per-position helpers
  int n_slots_ = 0;

  // (attn_residual is a free function used by forward; see recur.cpp)
};

// Loads a weights directory through store::load (manifest + bundles + experts),
// builds the Config from the manifest cfg (pool_resident from cfg "pool_resident";
// read_only from the top-level manifest key "read_only", defaulting true;
// pool_ram from top-level "pool_ram", defaulting 4), builds a FRESH PagedPool
// (lifecycle zeroed; segments=0, ever/since/last_seen/use=0, uid arange(n),
// next_uid=n) and the Coder, applies the read_only flag + pool.gate from the sd.
// The manifest "telemetry" is the END state and is NOT applied here (the test
// re-loads for a fresh lifecycle). Returns false on error.
bool load(const std::string& wdir, Config& cfg_out,
          paged::PagedPool& pool_out, Coder& coder_out);

}  // namespace minagi
