#include "decode.hpp"

#include <cmath>
#include <limits>
#include <vector>

namespace minagi {

int pick_next(const mt::Tensor& logits, const mt::Tensor& prev_ids,
              const DecodeWeights& w) {
  if (logits.dtype != mt::DType::FP32 || logits.shape.rank != 2 ||
      logits.shape.d[0] != 1 || logits.shape.d[1] <= 0) {
    return -1;
  }

  const int64_t V = logits.shape.d[1];
  const float* src = logits.ptr<float>();
  std::vector<float> l(src, src + V);

  const bool has_history =
      prev_ids.dtype == mt::DType::FP32 && prev_ids.shape.rank == 2 &&
      prev_ids.shape.d[0] == 1 && prev_ids.shape.d[1] > 0;
  const int64_t L = has_history ? prev_ids.shape.d[1] : 0;
  const float* prev = has_history ? prev_ids.ptr<float>() : nullptr;

  auto id_at = [prev](int64_t i) -> int64_t {
    const float v = prev[i];
    if (!(v >= 0.0f) || v > 8.0e6f) return -1;
    return static_cast<int64_t>(v);
  };

  // Adaptation trace (minagi/decode.py pick_next: lines 76-87). The last
  // adapt_window ids each leave a decaying mark on the logits of that id.
  if (w.strength != 0.0 && has_history) {
    const int64_t n =
        std::min<int64_t>(L, w.window > 0 ? static_cast<int64_t>(w.window) : 0);
    if (n > 0) {
      const float strength = static_cast<float>(w.strength);
      for (int64_t j = 0; j < n; ++j) {
        const int64_t tok = id_at(j + (L - n));
        if (tok < 0 || tok >= V) continue;
        const float wt =
            static_cast<float>(std::pow(w.decay, static_cast<double>(n - 1 - j)));
        l[tok] -= strength * wt;
      }
    }
  }

  // Repetition penalty (pick_next: lines 88-93): divide positive logits of
  // any already-emitted id, multiply negative ones.
  if (w.rep_penalty != 0.0 && w.rep_penalty != 1.0 && has_history) {
    const float rp = static_cast<float>(w.rep_penalty);
    std::vector<char> seen(V, 0);
    for (int64_t i = 0; i < L; ++i) {
      const int64_t tok = id_at(i);
      if (tok < 0 || tok >= V || seen[tok]) continue;
      seen[tok] = 1;
      l[tok] = l[tok] > 0.0f ? l[tok] / rp : l[tok] * rp;
    }
  }

  // Hard n-gram ban (pick_next: lines 94-103).
  if (w.no_repeat_ngram > 0 && has_history) {
    const int64_t n = static_cast<int64_t>(w.no_repeat_ngram);
    if (L >= n) {
      std::vector<int64_t> seq(L);
      bool all_ok = true;
      for (int64_t i = 0; i < L; ++i) {
        seq[i] = id_at(i);
        if (seq[i] < 0) all_ok = false;
      }
      if (all_ok) {
        std::vector<char> banned(V, 0);
        const int64_t P = n - 1;
        bool any = false;
        for (int64_t i = 0; i + n <= L; ++i) {
          bool same_prefix = true;
          for (int64_t k = 0; k < P; ++k) {
            if (seq[i + k] != seq[L - P + k]) {
              same_prefix = false;
              break;
            }
          }
          if (same_prefix && seq[i + P] >= 0 && seq[i + P] < V) {
            banned[seq[i + P]] = 1;
            any = true;
          }
        }
        if (any) {
          const float neg_inf = -std::numeric_limits<float>::infinity();
          for (int64_t v = 0; v < V; ++v) {
            if (banned[v]) l[v] = neg_inf;
          }
        }
      }
    }
  }

  // Deterministic greedy: temperature <= 0 in the python path is argmax, and
  // torch.argmax breaks ties by the lowest index.
  float best = -std::numeric_limits<float>::infinity();
  int best_id = 0;
  for (int64_t i = 0; i < V; ++i) {
    if (l[i] > best) {
      best = l[i];
      best_id = static_cast<int>(i);
    }
  }
  return best_id;
}

}  // namespace minagi