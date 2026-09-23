#pragma once
#include <cstdint>
#include <string>
#include <string_view>
namespace minagi {
struct Config {
  // model
  int64_t d_model=512, n_head=8, d_ff=1408, n_prelude=2, n_recur=1, n_coda=0;
  int64_t max_steps=24, min_steps=1, bptt_window=24;
  double  halt_prior=0.072, halt_thresh=0.9, ponder_beta=0.01, train_steps_mean=12.8;
  int64_t context_start=2048, context_end=4096, context_step=1;
  int64_t context_grow_every_chars=100000;
  double  context_gain_min=0.015;
  int64_t context_every_chars=65536;
  // pool
  int64_t experts=64, width=2048, depth=1, top_k=8, resident=32, ram_cache=96;
  double  capacity_factor=1.5;
  int64_t segment_chars=2048, reselect_chars=64;
  double  margin=0.10;
  int64_t dwell_chars=2048;
  double  explore=0.15;
  // training
  int64_t save_every=5;
  std::string precision="bf16";
  int64_t chunk=2048;
  double  lr=3.0e-4, trunk_lr_mult=0.1, weight_decay=0.1, clip=1.0;
  // growth
  int64_t every_chars=2000000, k=1;
  double  max_gap=0.4, dying_frac_max=0.35;
  int64_t max_in_flight=50;
  double  keep_ratio_min=0.35, recent_mult=4.0, max_disk_gb=10.0, mem_frac=0.95;
  double  birth_gate=0.001;
  // prune
  int64_t survival_chars=100000000;
  double  dying_at=0.65;
  // data
  std::string train_dir="data/train", held_out_dir="data/val", weights_dir="weights";
  int64_t shuffle_seed=0, passage=32768;
  // decoding
  double  adapt_strength=2.5, adapt_decay=0.88, rep_penalty=1.0;
  // plots
  bool    plots_enabled=true;
  int64_t last=100;
};
bool config_load(const std::string& path, Config& out);
double config_get_double(const Config&, const std::string& dotted, double fallback);
int64_t config_get_int(const Config&, const std::string& dotted, int64_t fallback);
std::string config_get_str(const Config&, const std::string& dotted, const std::string& fallback);
bool config_get_bool(const Config&, const std::string& dotted, bool fallback);
bool config_apply_cli(Config&, const std::string& dotted, const std::string& value_text);
}
