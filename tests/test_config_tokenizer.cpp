#include "config.hpp"
#include "tokenizer.hpp"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
static int failures = 0;
#define CHECK(name, cond) do { if (cond) { std::cout << "ok " << name << "\n"; } else { std::cout << "FAIL " << name << "\n"; ++failures; } } while(0)
static bool approx(double a, double b, double eps=1e-9) { return std::fabs(a - b) <= eps; }
int main() {
  const char* yaml =
    "model:\n"
    "  d_model: 512\n"
    "  n_head: 8\n"
    "  d_ff: 1408\n"
    "  n_prelude: 2\n"
    "  n_recur: 1\n"
    "  n_coda: 0\n"
    "  max_steps: 24\n"
    "  min_steps: 1\n"
    "  halt_prior: 0.072\n"
    "  halt_thresh: 0.9\n"
    "  ponder_beta: 0.01\n"
    "  bptt_window: 24\n"
    "  train_steps_mean: 12.8\n"
    "  context_start: 2048\n"
    "  context_end: 4096\n"
    "  context_step: 1\n"
    "  context_grow_every_chars: 100_000\n"
    "  context_gain_min: 0.015\n"
    "  context_every_chars: 65536\n"
    "pool:\n"
    "  experts: 64\n"
    "  width: 2048\n"
    "  depth: 1\n"
    "  top_k: 8\n"
    "  resident: 32\n"
    "  ram_cache: 96\n"
    "  capacity_factor: 1.5\n"
    "  segment_chars: 2048\n"
    "  reselect_chars: 64\n"
    "  margin: 0.10\n"
    "  dwell_chars: 2048\n"
    "  explore: 0.15\n"
    "training:\n"
    "  save_every: 5\n"
    "  precision: bf16\n"
    "  chunk: 2048\n"
    "  lr: 3.0e-4\n"
    "  trunk_lr_mult: 0.1\n"
    "  weight_decay: 0.1\n"
    "  clip: 1.0\n"
    "growth:\n"
    "  every_chars: 2_000_000\n"
    "  k: 1\n"
    "  max_gap: 0.4\n"
    "  dying_frac_max: 0.35\n"
    "  max_in_flight: 50\n"
    "  keep_ratio_min: 0.35\n"
    "  recent_mult: 4.0\n"
    "  max_disk_gb: 10.0\n"
    "  mem_frac: 0.95\n"
    "  birth_gate: 0.001\n"
    "prune:\n"
    "  survival_chars: 100_000_000\n"
    "  dying_at: 0.65\n"
    "data:\n"
    "  train: data/train\n"
    "  held_out: data/val\n"
    "  weights: weights\n"
    "  shuffle_seed: 0\n"
    "  passage: 32768\n"
    "decoding:\n"
    "  adapt_strength: 2.5\n"
    "  adapt_decay: 0.88\n"
    "  rep_penalty: 1.0\n"
    "plots:\n"
    "  enabled: true\n"
    "  last: 100\n";
  std::string tmp = "test_config_tmp.yaml";
  {
    std::ofstream f(tmp);
    f << yaml;
  }
  minagi::Config cfg;
  CHECK("config_load", minagi::config_load(tmp, cfg));
  CHECK("d_model", cfg.d_model == 512);
  CHECK("n_head", cfg.n_head == 8);
  CHECK("d_ff", cfg.d_ff == 1408);
  CHECK("n_prelude", cfg.n_prelude == 2);
  CHECK("n_recur", cfg.n_recur == 1);
  CHECK("n_coda", cfg.n_coda == 0);
  CHECK("max_steps", cfg.max_steps == 24);
  CHECK("min_steps", cfg.min_steps == 1);
  CHECK("halt_prior", approx(cfg.halt_prior, 0.072));
  CHECK("halt_thresh", approx(cfg.halt_thresh, 0.9));
  CHECK("ponder_beta", approx(cfg.ponder_beta, 0.01));
  CHECK("bptt_window", cfg.bptt_window == 24);
  CHECK("train_steps_mean", approx(cfg.train_steps_mean, 12.8));
  CHECK("context_start", cfg.context_start == 2048);
  CHECK("context_end", cfg.context_end == 4096);
  CHECK("context_grow_every_chars", cfg.context_grow_every_chars == 100000);
  CHECK("resident", cfg.resident == 32);
  CHECK("ram_cache", cfg.ram_cache == 96);
  CHECK("capacity_factor", approx(cfg.capacity_factor, 1.5));
  CHECK("experts", cfg.experts == 64);
  CHECK("width", cfg.width == 2048);
  CHECK("depth", cfg.depth == 1);
  CHECK("top_k", cfg.top_k == 8);
  CHECK("lr", approx(cfg.lr, 3.0e-4));
  CHECK("weight_decay", approx(cfg.weight_decay, 0.1));
  CHECK("survival_chars", cfg.survival_chars == 100000000);
  CHECK("train_dir", cfg.train_dir == "data/train");
  CHECK("held_out_dir", cfg.held_out_dir == "data/val");
  CHECK("weights_dir", cfg.weights_dir == "weights");
  CHECK("adapt_decay", approx(cfg.adapt_decay, 0.88));
  CHECK("rep_penalty", approx(cfg.rep_penalty, 1.0));
  CHECK("plots_enabled", cfg.plots_enabled == true);
  CHECK("get_double_pool_d_ff", minagi::config_get_double(cfg, "pool.d_ff", 1408) == 1408.0);
  CHECK("get_double_missing", minagi::config_get_double(cfg, "pool.nope", 1408) == 1408.0);
  CHECK("get_str_data_train", minagi::config_get_str(cfg, "data.train", "") == "data/train");
  CHECK("apply_cli_lr", (minagi::config_apply_cli(cfg, "training.lr", "1e-3"), approx(cfg.lr, 1e-3)));
  minagi::ByteTokenizer tok;
  {
    auto ids = tok.encode("hello");
    std::vector<int32_t> exp = {104, 101, 108, 108, 111};
    CHECK("encode_hello", ids == exp);
  }
  {
    auto ids = tok.encode("a<g>b");
    std::vector<int32_t> exp = {97, 262, 98};
    CHECK("encode_a_g_b", ids == exp);
  }
  {
    // "<thinking>x<response>" -> specials at ids 256/257 with raw byte in between
    auto ids = tok.encode(std::string("\x3cthink\x3e", 7) + "x" + std::string("\x3c\x2fthink\x3e", 8));
    std::vector<int32_t> exp = {256, 120, 257};
    CHECK("encode_think_x_response", ids == exp);
  }
  {
    auto ids = tok.encode("<user>hi</user>");
    std::vector<int32_t> exp = {258, 104, 105, 259};
    CHECK("encode_user_hi_closeuser", ids == exp);
  }
  {
    auto dec = tok.decode({104, 101, 108, 108, 111});
    CHECK("decode_hello", dec == "hello");
  }
  {
    auto dec = tok.decode({258, 104, 105, 259});
    CHECK("decode_user_hi_closeuser", dec == "<user>hi</user>");
  }
  {
    auto dec = tok.decode({97, 262, 98});
    CHECK("decode_a_g_b", dec == "a<g>b");
  }
  {
    std::string s = "h\xc3\xa9llo";
    auto ids = tok.encode(s);
    std::vector<int32_t> exp = {104, 0xC3, 0xA9, 108, 108, 111};
    CHECK("encode_utf8_bytes", ids == exp);
    auto dec = tok.decode(ids);
    CHECK("decode_utf8_roundtrip", dec == s);
  }
  {
    std::string invalid = std::string("a") + std::string("\xc0\x80", 2) + std::string("b");
    auto dec = tok.decode({97, 0xC0, 0x80, 98});
    std::string repl = "\xef\xbf\xbd";
    CHECK("decode_invalid_replacement", dec == "a" + repl + repl + "b");
  }
  {
    auto dec = tok.decode({0x80});
    std::string repl = "\xef\xbf\xbd";
    CHECK("decode_lone_continuation", dec == repl);
  }
  {
    std::string s = "h\xc3\xa9llo w\xc3\xb6rld <g> done";
    auto dec = tok.decode(tok.encode(s));
    CHECK("roundtrip_utf8_special", dec == s);
  }
  {
    std::string s = "\xf0\x9f\x98\x80";  // U+1F600 emoji (4-byte UTF-8)
    auto ids = tok.encode(s);
    std::vector<int32_t> exp = {0xF0, 0x9F, 0x98, 0x80};
    CHECK("encode_4byte_utf8", ids == exp);
    CHECK("roundtrip_emoji", tok.decode(ids) == s);
  }
  {
    CHECK("roundtrip_empty", tok.decode(tok.encode("")) == "");
  }
  {
    std::string s;
    for (int i = 0; i < 300; ++i) s.push_back(static_cast<char>('a' + (i % 26)));
    CHECK("roundtrip_ascii_300", tok.decode(tok.encode(s)) == s);
  }
  {
    CHECK("encode_eot", tok.encode("<|endoftext|>") == std::vector<int32_t>({264}));
  }
  CHECK("token_to_id_think", tok.token_to_id("\x3cthink\x3e") == 256);
  CHECK("token_to_id_response", tok.token_to_id("\x3c\x2fthink\x3e") == 257);
  CHECK("token_to_id_eot", tok.token_to_id("<|endoftext|>") == 264);
  CHECK("token_to_id_x", tok.token_to_id("x") == 120);
  CHECK("token_to_id_xy", tok.token_to_id("xy") == -1);
  std::remove(tmp.c_str());
  if (failures == 0) { std::cout << "ALL_TESTS_PASSED\n"; return 0; }
  std::cout << "SOME_TESTS_FAILED\n";
  return 1;
}
