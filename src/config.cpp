#include "config.hpp"
#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <string>
namespace minagi {
namespace {
std::string strip_underscores(std::string s) {
  s.erase(std::remove(s.begin(), s.end(), '_'), s.end());
  return s;
}
std::string trim(std::string s) {
  auto notspace = [](int c){ return !std::isspace(c); };
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), notspace));
  s.erase(std::find_if(s.rbegin(), s.rend(), notspace).base(), s.end());
  return s;
}
std::string strip_comment(std::string s) {
  size_t i = 0;
  while (i < s.size() && s[i] != '#') ++i;
  return s.substr(0, i);
}
bool parse_int64(std::string_view text, int64_t& out) {
  std::string t = strip_underscores(std::string(text));
  t = trim(t);
  if (t.empty()) return false;
  try {
    size_t pos = 0;
    long long v = std::stoll(t, &pos);
    if (pos != t.size()) return false;
    out = v;
    return true;
  } catch (...) { return false; }
}
bool parse_double(std::string_view text, double& out) {
  std::string t = strip_underscores(std::string(text));
  t = trim(t);
  if (t.empty()) return false;
  try {
    size_t pos = 0;
    double v = std::stod(t, &pos);
    if (pos != t.size()) return false;
    out = v;
    return true;
  } catch (...) { return false; }
}
bool parse_bool(std::string_view text, bool& out) {
  std::string t = trim(std::string(text));
  if (t == "true") { out = true; return true; }
  if (t == "false") { out = false; return true; }
  return false;
}
}
bool config_load(const std::string& path, Config& out) {
  std::ifstream f(path);
  if (!f) return false;
  std::string line;
  std::string section;
  while (std::getline(f, line)) {
    std::string no_comment = strip_comment(line);
    std::string trimmed = trim(no_comment);
    if (trimmed.empty()) continue;
    if (trimmed.back() == ':') {
      section = trimmed.substr(0, trimmed.size() - 1);
      continue;
    }
    size_t colon = trimmed.find(':');
    if (colon == std::string::npos) continue;
    std::string key = trim(trimmed.substr(0, colon));
    std::string val = trim(trimmed.substr(colon + 1));
    std::string full = section.empty() ? key : (section + "." + key);
    if (full == "model.d_model") { int64_t v; if (parse_int64(val, v)) out.d_model = v; }
    else if (full == "model.n_head") { int64_t v; if (parse_int64(val, v)) out.n_head = v; }
    else if (full == "model.d_ff") { int64_t v; if (parse_int64(val, v)) out.d_ff = v; }
    else if (full == "model.n_prelude") { int64_t v; if (parse_int64(val, v)) out.n_prelude = v; }
    else if (full == "model.n_recur") { int64_t v; if (parse_int64(val, v)) out.n_recur = v; }
    else if (full == "model.n_coda") { int64_t v; if (parse_int64(val, v)) out.n_coda = v; }
    else if (full == "model.max_steps") { int64_t v; if (parse_int64(val, v)) out.max_steps = v; }
    else if (full == "model.min_steps") { int64_t v; if (parse_int64(val, v)) out.min_steps = v; }
    else if (full == "model.halt_prior") { double v; if (parse_double(val, v)) out.halt_prior = v; }
    else if (full == "model.halt_thresh") { double v; if (parse_double(val, v)) out.halt_thresh = v; }
    else if (full == "model.ponder_beta") { double v; if (parse_double(val, v)) out.ponder_beta = v; }
    else if (full == "model.bptt_window") { int64_t v; if (parse_int64(val, v)) out.bptt_window = v; }
    else if (full == "model.train_steps_mean") { double v; if (parse_double(val, v)) out.train_steps_mean = v; }
    else if (full == "model.context_start") { int64_t v; if (parse_int64(val, v)) out.context_start = v; }
    else if (full == "model.context_end") { int64_t v; if (parse_int64(val, v)) out.context_end = v; }
    else if (full == "model.context_step") { int64_t v; if (parse_int64(val, v)) out.context_step = v; }
    else if (full == "model.context_grow_every_chars") { int64_t v; if (parse_int64(val, v)) out.context_grow_every_chars = v; }
    else if (full == "model.context_gain_min") { double v; if (parse_double(val, v)) out.context_gain_min = v; }
    else if (full == "model.context_every_chars") { int64_t v; if (parse_int64(val, v)) out.context_every_chars = v; }
    else if (full == "pool.experts") { int64_t v; if (parse_int64(val, v)) out.experts = v; }
    else if (full == "pool.width") { int64_t v; if (parse_int64(val, v)) out.width = v; }
    else if (full == "pool.depth") { int64_t v; if (parse_int64(val, v)) out.depth = v; }
    else if (full == "pool.top_k") { int64_t v; if (parse_int64(val, v)) out.top_k = v; }
    else if (full == "pool.resident") { int64_t v; if (parse_int64(val, v)) out.resident = v; }
    else if (full == "pool.ram_cache") { int64_t v; if (parse_int64(val, v)) out.ram_cache = v; }
    else if (full == "pool.capacity_factor") { double v; if (parse_double(val, v)) out.capacity_factor = v; }
    else if (full == "pool.segment_chars") { int64_t v; if (parse_int64(val, v)) out.segment_chars = v; }
    else if (full == "pool.reselect_chars") { int64_t v; if (parse_int64(val, v)) out.reselect_chars = v; }
    else if (full == "pool.margin") { double v; if (parse_double(val, v)) out.margin = v; }
    else if (full == "pool.dwell_chars") { int64_t v; if (parse_int64(val, v)) out.dwell_chars = v; }
    else if (full == "pool.explore") { double v; if (parse_double(val, v)) out.explore = v; }
    else if (full == "training.save_every") { int64_t v; if (parse_int64(val, v)) out.save_every = v; }
    else if (full == "training.precision") { out.precision = val; }
    else if (full == "training.chunk") { int64_t v; if (parse_int64(val, v)) out.chunk = v; }
    else if (full == "training.lr") { double v; if (parse_double(val, v)) out.lr = v; }
    else if (full == "training.trunk_lr_mult") { double v; if (parse_double(val, v)) out.trunk_lr_mult = v; }
    else if (full == "training.weight_decay") { double v; if (parse_double(val, v)) out.weight_decay = v; }
    else if (full == "training.clip") { double v; if (parse_double(val, v)) out.clip = v; }
    else if (full == "growth.every_chars") { int64_t v; if (parse_int64(val, v)) out.every_chars = v; }
    else if (full == "growth.k") { int64_t v; if (parse_int64(val, v)) out.k = v; }
    else if (full == "growth.max_gap") { double v; if (parse_double(val, v)) out.max_gap = v; }
    else if (full == "growth.dying_frac_max") { double v; if (parse_double(val, v)) out.dying_frac_max = v; }
    else if (full == "growth.max_in_flight") { int64_t v; if (parse_int64(val, v)) out.max_in_flight = v; }
    else if (full == "growth.keep_ratio_min") { double v; if (parse_double(val, v)) out.keep_ratio_min = v; }
    else if (full == "growth.recent_mult") { double v; if (parse_double(val, v)) out.recent_mult = v; }
    else if (full == "growth.max_disk_gb") { double v; if (parse_double(val, v)) out.max_disk_gb = v; }
    else if (full == "growth.mem_frac") { double v; if (parse_double(val, v)) out.mem_frac = v; }
    else if (full == "growth.birth_gate") { double v; if (parse_double(val, v)) out.birth_gate = v; }
    else if (full == "prune.survival_chars") { int64_t v; if (parse_int64(val, v)) out.survival_chars = v; }
    else if (full == "prune.dying_at") { double v; if (parse_double(val, v)) out.dying_at = v; }
    else if (full == "data.train") { out.train_dir = val; }
    else if (full == "data.held_out") { out.held_out_dir = val; }
    else if (full == "data.weights") { out.weights_dir = val; }
    else if (full == "data.shuffle_seed") { int64_t v; if (parse_int64(val, v)) out.shuffle_seed = v; }
    else if (full == "data.passage") { int64_t v; if (parse_int64(val, v)) out.passage = v; }
    else if (full == "decoding.adapt_strength") { double v; if (parse_double(val, v)) out.adapt_strength = v; }
    else if (full == "decoding.adapt_decay") { double v; if (parse_double(val, v)) out.adapt_decay = v; }
    else if (full == "decoding.rep_penalty") { double v; if (parse_double(val, v)) out.rep_penalty = v; }
    else if (full == "plots.enabled") { bool v; if (parse_bool(val, v)) out.plots_enabled = v; }
    else if (full == "plots.last") { int64_t v; if (parse_int64(val, v)) out.last = v; }
  }
  return true;
}
double config_get_double(const Config& c, const std::string& dotted, double fallback) {
  if (dotted == "model.d_model") return (double)c.d_model;
  if (dotted == "model.n_head") return (double)c.n_head;
  if (dotted == "model.d_ff") return (double)c.d_ff;
  if (dotted == "model.n_prelude") return (double)c.n_prelude;
  if (dotted == "model.n_recur") return (double)c.n_recur;
  if (dotted == "model.n_coda") return (double)c.n_coda;
  if (dotted == "model.max_steps") return (double)c.max_steps;
  if (dotted == "model.min_steps") return (double)c.min_steps;
  if (dotted == "model.halt_prior") return c.halt_prior;
  if (dotted == "model.halt_thresh") return c.halt_thresh;
  if (dotted == "model.ponder_beta") return c.ponder_beta;
  if (dotted == "model.bptt_window") return (double)c.bptt_window;
  if (dotted == "model.train_steps_mean") return c.train_steps_mean;
  if (dotted == "model.context_start") return (double)c.context_start;
  if (dotted == "model.context_end") return (double)c.context_end;
  if (dotted == "model.context_step") return (double)c.context_step;
  if (dotted == "model.context_grow_every_chars") return (double)c.context_grow_every_chars;
  if (dotted == "model.context_gain_min") return c.context_gain_min;
  if (dotted == "model.context_every_chars") return (double)c.context_every_chars;
  if (dotted == "pool.experts") return (double)c.experts;
  if (dotted == "pool.width") return (double)c.width;
  if (dotted == "pool.depth") return (double)c.depth;
  if (dotted == "pool.top_k") return (double)c.top_k;
  if (dotted == "pool.resident") return (double)c.resident;
  if (dotted == "pool.ram_cache") return (double)c.ram_cache;
  if (dotted == "pool.capacity_factor") return c.capacity_factor;
  if (dotted == "pool.segment_chars") return (double)c.segment_chars;
  if (dotted == "pool.reselect_chars") return (double)c.reselect_chars;
  if (dotted == "pool.margin") return c.margin;
  if (dotted == "pool.dwell_chars") return (double)c.dwell_chars;
  if (dotted == "pool.explore") return c.explore;
  if (dotted == "training.save_every") return (double)c.save_every;
  if (dotted == "training.chunk") return (double)c.chunk;
  if (dotted == "training.lr") return c.lr;
  if (dotted == "training.trunk_lr_mult") return c.trunk_lr_mult;
  if (dotted == "training.weight_decay") return c.weight_decay;
  if (dotted == "training.clip") return c.clip;
  if (dotted == "growth.every_chars") return (double)c.every_chars;
  if (dotted == "growth.k") return (double)c.k;
  if (dotted == "growth.max_gap") return c.max_gap;
  if (dotted == "growth.dying_frac_max") return c.dying_frac_max;
  if (dotted == "growth.max_in_flight") return (double)c.max_in_flight;
  if (dotted == "growth.keep_ratio_min") return c.keep_ratio_min;
  if (dotted == "growth.recent_mult") return c.recent_mult;
  if (dotted == "growth.max_disk_gb") return c.max_disk_gb;
  if (dotted == "growth.mem_frac") return c.mem_frac;
  if (dotted == "growth.birth_gate") return c.birth_gate;
  if (dotted == "prune.survival_chars") return (double)c.survival_chars;
  if (dotted == "prune.dying_at") return c.dying_at;
  if (dotted == "data.shuffle_seed") return (double)c.shuffle_seed;
  if (dotted == "data.passage") return (double)c.passage;
  if (dotted == "decoding.adapt_strength") return c.adapt_strength;
  if (dotted == "decoding.adapt_decay") return c.adapt_decay;
  if (dotted == "decoding.rep_penalty") return c.rep_penalty;
  if (dotted == "plots.last") return (double)c.last;
  return fallback;
}
int64_t config_get_int(const Config& c, const std::string& dotted, int64_t fallback) {
  if (dotted == "model.d_model") return c.d_model;
  if (dotted == "model.n_head") return c.n_head;
  if (dotted == "model.d_ff") return c.d_ff;
  if (dotted == "model.n_prelude") return c.n_prelude;
  if (dotted == "model.n_recur") return c.n_recur;
  if (dotted == "model.n_coda") return c.n_coda;
  if (dotted == "model.max_steps") return c.max_steps;
  if (dotted == "model.min_steps") return c.min_steps;
  if (dotted == "model.bptt_window") return c.bptt_window;
  if (dotted == "model.context_start") return c.context_start;
  if (dotted == "model.context_end") return c.context_end;
  if (dotted == "model.context_step") return c.context_step;
  if (dotted == "model.context_grow_every_chars") return c.context_grow_every_chars;
  if (dotted == "model.context_every_chars") return c.context_every_chars;
  if (dotted == "pool.experts") return c.experts;
  if (dotted == "pool.width") return c.width;
  if (dotted == "pool.depth") return c.depth;
  if (dotted == "pool.top_k") return c.top_k;
  if (dotted == "pool.resident") return c.resident;
  if (dotted == "pool.ram_cache") return c.ram_cache;
  if (dotted == "pool.segment_chars") return c.segment_chars;
  if (dotted == "pool.reselect_chars") return c.reselect_chars;
  if (dotted == "pool.dwell_chars") return c.dwell_chars;
  if (dotted == "training.save_every") return c.save_every;
  if (dotted == "training.chunk") return c.chunk;
  if (dotted == "growth.every_chars") return c.every_chars;
  if (dotted == "growth.k") return c.k;
  if (dotted == "growth.max_in_flight") return c.max_in_flight;
  if (dotted == "prune.survival_chars") return c.survival_chars;
  if (dotted == "data.shuffle_seed") return c.shuffle_seed;
  if (dotted == "data.passage") return c.passage;
  if (dotted == "plots.last") return c.last;
  return fallback;
}
std::string config_get_str(const Config& c, const std::string& dotted, const std::string& fallback) {
  if (dotted == "training.precision") return c.precision;
  if (dotted == "data.train") return c.train_dir;
  if (dotted == "data.held_out") return c.held_out_dir;
  if (dotted == "data.weights") return c.weights_dir;
  return fallback;
}
bool config_get_bool(const Config& c, const std::string& dotted, bool fallback) {
  if (dotted == "plots.enabled") return c.plots_enabled;
  return fallback;
}
bool config_apply_cli(Config& c, const std::string& dotted, const std::string& value_text) {
  if (dotted == "model.d_model") { int64_t v; if (!parse_int64(value_text, v)) return false; c.d_model = v; return true; }
  if (dotted == "model.n_head") { int64_t v; if (!parse_int64(value_text, v)) return false; c.n_head = v; return true; }
  if (dotted == "model.d_ff") { int64_t v; if (!parse_int64(value_text, v)) return false; c.d_ff = v; return true; }
  if (dotted == "model.n_prelude") { int64_t v; if (!parse_int64(value_text, v)) return false; c.n_prelude = v; return true; }
  if (dotted == "model.n_recur") { int64_t v; if (!parse_int64(value_text, v)) return false; c.n_recur = v; return true; }
  if (dotted == "model.n_coda") { int64_t v; if (!parse_int64(value_text, v)) return false; c.n_coda = v; return true; }
  if (dotted == "model.max_steps") { int64_t v; if (!parse_int64(value_text, v)) return false; c.max_steps = v; return true; }
  if (dotted == "model.min_steps") { int64_t v; if (!parse_int64(value_text, v)) return false; c.min_steps = v; return true; }
  if (dotted == "model.halt_prior") { double v; if (!parse_double(value_text, v)) return false; c.halt_prior = v; return true; }
  if (dotted == "model.halt_thresh") { double v; if (!parse_double(value_text, v)) return false; c.halt_thresh = v; return true; }
  if (dotted == "model.ponder_beta") { double v; if (!parse_double(value_text, v)) return false; c.ponder_beta = v; return true; }
  if (dotted == "model.bptt_window") { int64_t v; if (!parse_int64(value_text, v)) return false; c.bptt_window = v; return true; }
  if (dotted == "model.train_steps_mean") { double v; if (!parse_double(value_text, v)) return false; c.train_steps_mean = v; return true; }
  if (dotted == "model.context_start") { int64_t v; if (!parse_int64(value_text, v)) return false; c.context_start = v; return true; }
  if (dotted == "model.context_end") { int64_t v; if (!parse_int64(value_text, v)) return false; c.context_end = v; return true; }
  if (dotted == "model.context_step") { int64_t v; if (!parse_int64(value_text, v)) return false; c.context_step = v; return true; }
  if (dotted == "model.context_grow_every_chars") { int64_t v; if (!parse_int64(value_text, v)) return false; c.context_grow_every_chars = v; return true; }
  if (dotted == "model.context_gain_min") { double v; if (!parse_double(value_text, v)) return false; c.context_gain_min = v; return true; }
  if (dotted == "model.context_every_chars") { int64_t v; if (!parse_int64(value_text, v)) return false; c.context_every_chars = v; return true; }
  if (dotted == "pool.experts") { int64_t v; if (!parse_int64(value_text, v)) return false; c.experts = v; return true; }
  if (dotted == "pool.width") { int64_t v; if (!parse_int64(value_text, v)) return false; c.width = v; return true; }
  if (dotted == "pool.depth") { int64_t v; if (!parse_int64(value_text, v)) return false; c.depth = v; return true; }
  if (dotted == "pool.top_k") { int64_t v; if (!parse_int64(value_text, v)) return false; c.top_k = v; return true; }
  if (dotted == "pool.resident") { int64_t v; if (!parse_int64(value_text, v)) return false; c.resident = v; return true; }
  if (dotted == "pool.ram_cache") { int64_t v; if (!parse_int64(value_text, v)) return false; c.ram_cache = v; return true; }
  if (dotted == "pool.capacity_factor") { double v; if (!parse_double(value_text, v)) return false; c.capacity_factor = v; return true; }
  if (dotted == "pool.segment_chars") { int64_t v; if (!parse_int64(value_text, v)) return false; c.segment_chars = v; return true; }
  if (dotted == "pool.reselect_chars") { int64_t v; if (!parse_int64(value_text, v)) return false; c.reselect_chars = v; return true; }
  if (dotted == "pool.margin") { double v; if (!parse_double(value_text, v)) return false; c.margin = v; return true; }
  if (dotted == "pool.dwell_chars") { int64_t v; if (!parse_int64(value_text, v)) return false; c.dwell_chars = v; return true; }
  if (dotted == "pool.explore") { double v; if (!parse_double(value_text, v)) return false; c.explore = v; return true; }
  if (dotted == "training.save_every") { int64_t v; if (!parse_int64(value_text, v)) return false; c.save_every = v; return true; }
  if (dotted == "training.precision") { c.precision = value_text; return true; }
  if (dotted == "training.chunk") { int64_t v; if (!parse_int64(value_text, v)) return false; c.chunk = v; return true; }
  if (dotted == "training.lr") { double v; if (!parse_double(value_text, v)) return false; c.lr = v; return true; }
  if (dotted == "training.trunk_lr_mult") { double v; if (!parse_double(value_text, v)) return false; c.trunk_lr_mult = v; return true; }
  if (dotted == "training.weight_decay") { double v; if (!parse_double(value_text, v)) return false; c.weight_decay = v; return true; }
  if (dotted == "training.clip") { double v; if (!parse_double(value_text, v)) return false; c.clip = v; return true; }
  if (dotted == "growth.every_chars") { int64_t v; if (!parse_int64(value_text, v)) return false; c.every_chars = v; return true; }
  if (dotted == "growth.k") { int64_t v; if (!parse_int64(value_text, v)) return false; c.k = v; return true; }
  if (dotted == "growth.max_gap") { double v; if (!parse_double(value_text, v)) return false; c.max_gap = v; return true; }
  if (dotted == "growth.dying_frac_max") { double v; if (!parse_double(value_text, v)) return false; c.dying_frac_max = v; return true; }
  if (dotted == "growth.max_in_flight") { int64_t v; if (!parse_int64(value_text, v)) return false; c.max_in_flight = v; return true; }
  if (dotted == "growth.keep_ratio_min") { double v; if (!parse_double(value_text, v)) return false; c.keep_ratio_min = v; return true; }
  if (dotted == "growth.recent_mult") { double v; if (!parse_double(value_text, v)) return false; c.recent_mult = v; return true; }
  if (dotted == "growth.max_disk_gb") { double v; if (!parse_double(value_text, v)) return false; c.max_disk_gb = v; return true; }
  if (dotted == "growth.mem_frac") { double v; if (!parse_double(value_text, v)) return false; c.mem_frac = v; return true; }
  if (dotted == "growth.birth_gate") { double v; if (!parse_double(value_text, v)) return false; c.birth_gate = v; return true; }
  if (dotted == "prune.survival_chars") { int64_t v; if (!parse_int64(value_text, v)) return false; c.survival_chars = v; return true; }
  if (dotted == "prune.dying_at") { double v; if (!parse_double(value_text, v)) return false; c.dying_at = v; return true; }
  if (dotted == "data.train") { c.train_dir = value_text; return true; }
  if (dotted == "data.held_out") { c.held_out_dir = value_text; return true; }
  if (dotted == "data.weights") { c.weights_dir = value_text; return true; }
  if (dotted == "data.shuffle_seed") { int64_t v; if (!parse_int64(value_text, v)) return false; c.shuffle_seed = v; return true; }
  if (dotted == "data.passage") { int64_t v; if (!parse_int64(value_text, v)) return false; c.passage = v; return true; }
  if (dotted == "decoding.adapt_strength") { double v; if (!parse_double(value_text, v)) return false; c.adapt_strength = v; return true; }
  if (dotted == "decoding.adapt_decay") { double v; if (!parse_double(value_text, v)) return false; c.adapt_decay = v; return true; }
  if (dotted == "decoding.rep_penalty") { double v; if (!parse_double(value_text, v)) return false; c.rep_penalty = v; return true; }
  if (dotted == "plots.enabled") { bool v; if (!parse_bool(value_text, v)) return false; c.plots_enabled = v; return true; }
  if (dotted == "plots.last") { int64_t v; if (!parse_int64(value_text, v)) return false; c.last = v; return true; }
  return false;
}
}
