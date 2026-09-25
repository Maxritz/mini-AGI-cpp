// src/paged.hpp
// Paged expert pool for the inference path (minagi/paged.py): a Tiers LRU
// disk/RAM cache plus a PagedPool whose slot tensors front only the resident
// experts. Sections 2 and 3 of the task; Tier-2 pieces (keys/demand/telemetry
// round-trip) are included for the self-consistency test.
#pragma once
#include "pool.hpp"
#include "mininpz.hpp"
#include <cstdint>
#include <list>
#include <memory>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace minagi { class PcgRng; }  // forward decl (defined in init.hpp)
namespace minagi::paged {

struct TierCounters {
  long long reads = 0;
  long long hits = 0;
  long long evictions = 0;
  long long writebacks = 0;
};

struct PagedParams {
  int resident = 3;
  int ram_capacity = 4;
  double explore = 0.15;
  double margin = 0.10;
  int dwell = 4;
  bool read_only = true;
};

// fp32 <-> mininpz::Array conversion shared by the file round-trips (Tiers,
// store::save_paged, recurs). Not training logic; just layout.
mt::Tensor array_to_tensor(const mininpz::Array& arr);
mininpz::Array tensor_to_array(const mt::Tensor& t);

// Section 2: the disk/RAM LRU cache holding expert e%05d.npz files.
class Tiers {
public:
  // experts_dir holds e%05d.npz. read_only means nothing is ever marked dirty
  // and nothing is ever written.
  Tiers(const std::string& experts_dir, int d_model, int d_ff, int ram_capacity,
        bool read_only);

  // Returns false when the expert file is missing or lacks w1/w3/w2 (the file
  // is then NOT cached and no counter moves beyond reads++).
  bool fetch(int uid, mt::Tensor& w1, mt::Tensor& w3, mt::Tensor& w2);
  // dirty=true but read_only -> the dirty flag is silently dropped.
  void put(int uid, const mt::Tensor& w1, const mt::Tensor& w3,
           const mt::Tensor& w2, bool dirty);
  void flush();  // write all dirty in insertion order (no-op when read_only)
  const TierCounters& counters() const;
  void reset_counters();  // Tier-2: zero reads/hits/evictions/writebacks

private:
  struct Entry {
    mt::Tensor w1, w3, w2;
  };
  std::string path_;
  int d_model_;
  int d_ff_;
  int ram_capacity_;
  bool read_only_;
  std::unordered_map<int, Entry> ram_;
  std::list<int> order_;            // MRU at the back
  std::unordered_set<int> dirty_;
  std::vector<int> dirty_order_;    // insertion order for flush()
  TierCounters counters_;

  std::string file_path(int uid) const;
  bool from_disk(int uid, Entry& out);
  bool to_disk(int uid, const Entry& e);
  void trim();
  void move_mru(int uid);
};

// Section 3: the pooled surface PooledMLP routes into.
class PagedPool : public minagi::Pool {
public:
  // Fresh lifecycle state (segments=0, ever/since/last_seen/use/admits/
  // gate_seen/born=0, uid arange(n), next_uid=n, summary zeros). gate_init is
  // the initial gate tensor (ones(n) from the caller; overwritten by the
  // directory via set_gate when loading weights).
  PagedPool(int n_experts, int d_model, int pool_d_ff, int resident,
            int ram_capacity, double explore, double margin, int dwell,
            bool read_only, mt::Tensor gate_init);

  // Wires the Tiers backing swap_to (the experts directory). The listed ctor
  // carries no directory, so recur::load supplies it here. Tiers keeps the
  // pool's d_model/d_ff/ram_capacity/read_only.
   void set_experts_dir(const std::string& experts_dir);  // stores dir + builds Tiers
   void reset_counters();                                  // Tier-2: zero TIERS counters

  // Tier-2: restore use/admits/born/since/last_seen/gate_seen/ever/uid/
  // next_uid/segments from a manifest "telemetry" block (section 3). Accepts
  // either the block itself or a wrapper object carrying a "telemetry" key;
  // a null JsonValue is skipped.
  void load_telemetry(const mini::JsonValue& telemetry);
  bool have_telemetry() const;  // a non-null load_telemetry already ran

  // The section-3 surface:
  int n_experts() const override;
  int n_routable() const override;
  int router_rows() const override;
  const std::vector<int>& slots() const override;
  const mt::Tensor& gate() const override;
  const mt::Tensor& w1() const override;
  const mt::Tensor& w3() const override;
  const mt::Tensor& w2() const override;
  void note_use(const std::vector<long long>& slot_hits) const override;

  std::vector<int> resident_rows() const;   // [max(s,0) for s in slots]
  mt::Tensor routable_gate() const;         // gate[resident_rows()]

  // Segment lifecycle:
  std::vector<int> choose();                // section-3 choose()
  int swap_to(const std::vector<int>& ids); // returns loads; exact order
  void observe(const mt::Tensor& segment_mean);  // summary = segment_mean (copy)
  const mt::Tensor& summary() const;
   int segments() const;
   const TierCounters& counters() const;
  template <class... Args>
  void attach_sites(Args&&...) {}   // no-op: no growth in the C++ reference
  void flush();                     // park residents via Tiers then flush Tiers

  // Tier-2 (self-consistency only):
  int build_keys(bool verbose);                                   // power iteration over every expert's w1
  int choose_by_demand(const mt::Tensor& gates_abs, const mt::Tensor& ever_bool,
                       const mt::Tensor& since, const mt::Tensor& last_seen,
                       const std::vector<double>& want,
                       std::vector<int>& out) const;

  // Wiring required by Coder::load_weights and store::save_paged (the listed
  // surface is read-only, so the directory's values are set here).
  void set_gate(const mt::Tensor& g);
   void set_segment_router(const mt::Tensor& w);
   const mt::Tensor& segment_router() const { return segment_router_; }
  mini::JsonValue telemetry() const;  // section-3 telemetry block
  mini::JsonValue telemetry_json() const;  // alias used by store::save_paged
  void lookup_gate(int eid, double& g) const;  // gate[eid] or 1.0 if unknown
  void emit_manifest_lifecycle(std::map<std::string, mini::JsonValue>& obj) const;
  static bool is_moment_key(const std::string& key);
  static bool is_router_key(const std::string& key);
   int d_model() const { return d_model_; }
   int d_ff() const { return d_ff_; }

   // --- Training lifecycle (growth/prune) ---
   // Grow by k new experts born at birth_gate. Returns new total n_experts.
   int grow(int k, int step, double birth_gate,
            float weight_std, minagi::PcgRng& rng, const std::string& experts_dir);
   // Prune stale experts. Returns number removed.
   int prune(int step, int survival_steps, int protect);

private:
  void refresh_keys();                  // segments%8==0 (never fires in the golden)
  mt::Tensor key_of(const mt::Tensor& w1) const;  // dominant input direction
  double segment_score(int e) const;    // choose() logit for expert e
  void row_of(const mt::Tensor& t, int slot, int64_t row_numel,
              mt::Tensor& out) const;       // copy [d_ff,d]/[d,d_ff]/[d] row
  void set_row(mt::Tensor& t, int slot, const mt::Tensor& row);
  std::string expert_file_path(int uid) const;  // experts/e%05d.npz in experts_dir_

  mutable int n_experts_;
  int d_model_;
  int d_ff_;
  int resident_;
  double explore_;
  double margin_;
  int dwell_;
  bool read_only_;
  int ram_capacity_;

  mutable std::vector<int> slots_;              // slot -> expert uid (-1 empty)
  mt::Tensor gate_;                     // [n_experts]
  mt::Tensor w1_, w3_, w2_;             // [resident, d_ff, d] / [resident, d, d_ff]
  mutable mt::Tensor use_, age_, born_, gate_seen_, last_seen_, since_, admits_;
  mutable std::vector<char> ever_;
  mutable std::vector<long long> uid_;
  mutable long long next_uid_;
  mutable int segments_;
  mutable long long swaps_;
  mt::Tensor summary_;                  // [d_model]
  mt::Tensor segment_router_;           // [max(n_experts,1), d_model] weight
   std::unique_ptr<Tiers> tiers_;
   TierCounters zero_counters_;
   bool have_telemetry_;
   bool experts_dir_set_;
   std::string experts_dir_;

  // Tier-2 keys (demand): dominant input direction per expert.
  mt::Tensor keys_;
  bool keys_built_;
};

}  // namespace minagi::paged