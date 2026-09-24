// src/store.hpp
// Weights-directory store: save/load a model's full tensor map as a directory
// of .npz bundles (core.npz, routers.npz, optim.npz, experts/e%05d.npz) plus a
// manifest.json. Port of minagi/store.py (non-paged path).
#pragma once
#include "mininpz.hpp"
#include "tensor.hpp"
#include <string>
#include <vector>
#include <map>
#include <cstdint>
#include <cstddef>

namespace minagi::paged { class PagedPool; }

namespace store {

struct Manifest {
    int step = 0;
    double val = -1.0;
    std::map<std::string, std::string> cfg;  // original string map (kept for parity)
    mini::JsonValue cfg_obj;                  // parsed cfg object (typed numbers/bools)
    mini::JsonValue telemetry;                // section-3 telemetry block (possibly null)
    bool paged = false;                       // manifest "paged"
    bool read_only = false;                  // top-level manifest "read_only"
    int pool_ram = 4;                        // top-level manifest "pool_ram"
    std::vector<bool> pool_ever;             // cfg.pool_ever (lifecycle snapshot)
    std::vector<double> pool_since;          // cfg.pool_since
    int n_experts = 0;
    int d_model = 0;
    int d_ff = 0;
    std::vector<std::string> core_tensors;
    std::vector<std::string> router_tensors;
    struct ExpertInfo {
        int id = 0;
        std::string file;
        int remotes = 0;
        int params = 0;
        long long bytes = 0;
        bool moments = false;
        double gate = 1.0;
    };
    std::vector<ExpertInfo> experts;
    long long total_bytes = 0;
    int removed_expert_files = 0;
};

struct SaveResult {
    int n_experts = 0;
    long long total_bytes = 0;
};

// Classification helpers
bool is_expert(const std::string& key);
bool is_router(const std::string& key);
bool is_moment(const std::string& name);
bool is_optim(const std::string& name);
int expert_index(const std::string& key);
std::string expert_leaf(const std::string& key);
std::string expert_dir(const std::string& dir);  // dir/experts (helpers above use it)

// bf16 packing (fp32 -> int16 bits, RNE)
std::vector<int16_t> pack_bf16(const std::vector<float>& vals);
std::vector<float> unpack_bf16(const std::vector<int16_t>& bits);

// Save / Load
bool save(const std::string& dir,
          const std::map<std::string, mininpz::Array>& state,
          int step,
          double val,
          const std::map<std::string, std::string>& cfg,
          const std::map<int, double>& gates,
          SaveResult& out);

bool load(const std::string& dir,
          Manifest& manifest,
          std::map<std::string, mininpz::Array>& state);

// Paged-path save (section 1 _save_paged). pool.flush() first, writes core +
// router bundles, manifest with pool_resident and pool_max=max(pool_max,entries)
// and cfg merged with pool_ever/pool_since; per-expert entries mirror the
// manifest's expert list (moments==true). Atomic writes. Returns false on error.
bool save_paged(const std::vector<std::pair<std::string, mt::Tensor>>& sd,
                const minagi::paged::PagedPool& pool,
                const mini::JsonValue& cfg_obj,
                const std::string& dir, int step, double val,
                SaveResult* out);

double best_val(const std::string& dir);
std::string summarise(const std::string& dir);

}  // namespace store