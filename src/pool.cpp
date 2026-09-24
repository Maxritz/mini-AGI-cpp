// src/pool.cpp
// The mojito router/dispatch of section 4 (minagi/pool.py PooledMLP.forward,
// inference, fp32): per-token soft top-k over the resident slots, per-expert
// capacity-bounded buffered SwiGLU eval, and scatter-add back into the token
// rows. No autograd, no aux load-balancing loss, no pressure/want_k EMAs
// (training-only; see task TIER-2/EXCLUDED note).
#include "pool.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>
#include <limits>
#include <utility>
#include <vector>

namespace minagi {
namespace {

inline float silu1(float x) {
  // silu(x) = x * sigmoid(x); stable forms split on sign to avoid overflow.
  if (x == 0.0f) return 0.0f;
  if (x > 0.0f) {
    const float e = std::exp(-x);
    return x / (1.0f + e);
  }
  const float e = std::exp(x);
  return x * e / (1.0f + e);
}

mt::Tensor zeros2d(int64_t r, int64_t c) {
  mt::Shape s;
  s.rank = 2;
  s.d[0] = r;
  s.d[1] = c;
  return mt::make(s, mt::DType::FP32, 0.0f);
}

mt::Tensor empty_tensor() {
  mt::Shape s;
  s.rank = 0;
  return mt::make(s, mt::DType::FP32, 0.0f);
}

struct Assignment {
  int64_t slot = 0;     // resident slot index chosen by routing
  int64_t token = 0;    // token row (0..N-1)
  float weight = 0.0f;  // routed weight after gate
  int64_t in_slot = 0;  // position within the slot's buffer (pre-drop)
  bool kept = true;
};

}  // namespace

mt::Tensor pool_mlp_forward(const Pool& pool, const mt::Tensor& x,
                            const mt::Tensor& router_w, const mt::Tensor& depth_emb,
                            int top_k, double capacity_factor, RouteStats* stats) {
  const mt::Tensor bad = empty_tensor();
  if (x.dtype != mt::DType::FP32 || x.shape.rank != 2 ||
      x.shape.d[0] <= 0 || x.shape.d[1] <= 0) {
    return bad;
  }
  if (router_w.dtype != mt::DType::FP32 || router_w.shape.rank != 2) {
    return bad;
  }
  if (depth_emb.dtype != mt::DType::FP32 || depth_emb.shape.rank != 1) {
    return bad;
  }

  const int64_t N = x.shape.d[0];
  const int64_t D = x.shape.d[1];
  const int n = pool.n_routable();
  if (n <= 0) {
    return zeros2d(N, D);
  }
  if (depth_emb.shape.d[0] != D || router_w.shape.d[1] != D) {
    return bad;
  }

  const std::vector<int>& slots = pool.slots();
  const mt::Tensor& gate = pool.gate();
  const mt::Tensor& W1 = pool.w1();
  const mt::Tensor& W3 = pool.w3();
  const mt::Tensor& W2 = pool.w2();
  const int64_t n_experts = router_w.shape.d[0];
  const bool shape_ok =
      gate.dtype == mt::DType::FP32 && gate.shape.rank == 1 &&
      W1.dtype == mt::DType::FP32 && W1.shape.rank == 3 &&
      W3.dtype == mt::DType::FP32 && W3.shape.rank == 3 &&
      W2.dtype == mt::DType::FP32 && W2.shape.rank == 3 &&
      (int64_t)slots.size() == n && W1.shape.d[0] == n && W3.shape.d[0] == n &&
      W2.shape.d[0] == n && W1.shape.d[2] == D && W3.shape.d[2] == D &&
      W2.shape.d[1] == D;
  if (!shape_ok) {
    return bad;
  }
  const int64_t dff = W1.shape.d[1];
  if (W2.shape.d[2] != dff) {
    return bad;
  }

  // resident_rows(): [max(s,0) for s in slots]
  std::vector<int> rows(static_cast<size_t>(n), 0);
  for (int j = 0; j < n; ++j) {
    const int s = slots[j];
    rows[j] = s > 0 ? s : 0;
    if ((int64_t)rows[j] >= n_experts || (int64_t)rows[j] >= gate.shape.d[0]) {
      return bad;
    }
  }

  const int k = top_k > 0 ? std::min(top_k, n) : 0;
  if (k <= 0) {
    return zeros2d(N, D);
  }

  const float* xp = x.ptr<float>();
  const float* dp = depth_emb.ptr<float>();
  const float* rw = router_w.ptr<float>();

  // logits [N, n] = (x + depth_emb) @ w^T ; w = router_w[rows].
  // Window not scaled by 1/sqrt(d) and no bias (deliberate; section 4 + QUALITY
  // BAR note (a)).
  std::vector<float> logits(static_cast<size_t>(N) * static_cast<size_t>(n));
  for (int64_t r = 0; r < N; ++r) {
    for (int c = 0; c < n; ++c) {
      double acc = 0.0;
      const float* wrow = rw + static_cast<size_t>(rows[c]) * static_cast<size_t>(D);
      for (int64_t dd = 0; dd < D; ++dd) {
        const double xv = static_cast<double>(xp[static_cast<size_t>(r) * static_cast<size_t>(D) + dd] + dp[dd]);
        acc += xv * static_cast<double>(wrow[dd]);
      }
      logits[static_cast<size_t>(r) * static_cast<size_t>(n) + c] = static_cast<float>(acc);
    }
  }

  // probs = softmax(logits, -1), fp32 exps + double sum (mt::softmax semantics).
  std::vector<float> probs(static_cast<size_t>(N) * static_cast<size_t>(n));
  std::vector<float> exps(static_cast<size_t>(n));
  for (int64_t r = 0; r < N; ++r) {
    const float* row = logits.data() + static_cast<size_t>(r) * static_cast<size_t>(n);
    float mx = -std::numeric_limits<float>::infinity();
    for (int c = 0; c < n; ++c) {
      if (row[c] > mx) mx = row[c];
    }
    double sum = 0.0;
    for (int c = 0; c < n; ++c) {
      const float e = static_cast<float>(std::exp(static_cast<double>(row[c]) - static_cast<double>(mx)));
      exps[c] = e;
      sum += e;
    }
    for (int c = 0; c < n; ++c) {
      probs[static_cast<size_t>(r) * static_cast<size_t>(n) + c] =
          static_cast<float>(exps[c] / sum);
    }
  }

  // top-k: descending value, ascending index on ties (mt::topk tie-break).
  std::vector<float> wv(static_cast<size_t>(N) * static_cast<size_t>(k));
  std::vector<int64_t> idx(static_cast<size_t>(N) * static_cast<size_t>(k));
  for (int64_t r = 0; r < N; ++r) {
    std::vector<std::pair<float, int>> cand(static_cast<size_t>(n));
    for (int c = 0; c < n; ++c) {
      cand[static_cast<size_t>(c)] = {probs[static_cast<size_t>(r) * static_cast<size_t>(n) + c], c};
    }
    std::partial_sort(cand.begin(), cand.begin() + k, cand.end(),
                      [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
                        if (a.first != b.first) return a.first > b.first;
                        return a.second < b.second;
                      });
    for (int i = 0; i < k; ++i) {
      wv[static_cast<size_t>(r) * static_cast<size_t>(k) + i] = cand[static_cast<size_t>(i)].first;
      idx[static_cast<size_t>(r) * static_cast<size_t>(k) + i] = cand[static_cast<size_t>(i)].second;
    }
  }

  // wv = wv / wv.sum(-1); wv = wv * routable_gate()[idx]
  const float* gp = gate.ptr<float>();
  for (int64_t r = 0; r < N; ++r) {
    double sum = 0.0;
    for (int i = 0; i < k; ++i) {
      sum += wv[static_cast<size_t>(r) * static_cast<size_t>(k) + i];
    }
    for (int i = 0; i < k; ++i) {
      float w = wv[static_cast<size_t>(r) * static_cast<size_t>(k) + i];
      if (sum != 0.0) {
        w = static_cast<float>(w / sum);
      }
      const int64_t sl = idx[static_cast<size_t>(r) * static_cast<size_t>(k) + i];
      wv[static_cast<size_t>(r) * static_cast<size_t>(k) + i] = w * gp[rows[sl]];
    }
  }

  {
    static bool pd=false;
    float gsum=0; for(int i=0;i<gate.shape.d[0];++i) gsum+=gate.ptr<float>()[i];
    if(!pd && gsum > 0.5){
      float mx=*std::max_element(logits.begin(),logits.end());
      double sm=0; for(int i=0;i<n;++i){double e=std::exp(static_cast<double>(logits[i])-mx); sm+=e;}
      std::cout << "DBG pool n=" << n << " k=" << k << " gsum=" << gsum;
      for(int c=0;c<n;++c){ std::cout << " [c=" << c << " exp=" << rows[c] << " gate=" << gp[rows[c]] << " log=" << logits[c] << " p=" << (std::exp((static_cast<double>(logits[c])-mx))/sm) << "]"; }
      std::cout << " | idx:"; for(int i=0;i<k;++i) std::cout << " " << idx[i];
      std::cout << " | wv:"; for(int i=0;i<k;++i) std::cout << " " << wv[i];
      std::cout << std::endl; pd=true;
    }
  }

  // Routing bookkeeping: per-slot pick counts (ALL picks, before any drop).
  std::vector<long long> slot_hits(static_cast<size_t>(n), 0);
  for (size_t j = 0; j < static_cast<size_t>(N) * static_cast<size_t>(k); ++j) {
    slot_hits[static_cast<size_t>(idx[j])]++;
  }
  pool.note_use(slot_hits);

  // Capacity-bounded batched dispatch (Switch-style), section 4 verbatim.
  const int64_t total = N * static_cast<int64_t>(k);
  std::vector<Assignment> asn(static_cast<size_t>(total));
  for (int64_t j = 0; j < total; ++j) {
    Assignment& a = asn[static_cast<size_t>(j)];
    a.slot = idx[static_cast<size_t>(j)];
    a.token = j / static_cast<int64_t>(k);
    a.weight = wv[static_cast<size_t>(j)];
  }
  std::stable_sort(asn.begin(), asn.end(),
                   [](const Assignment& a, const Assignment& b) {
                     return a.slot < b.slot;
                   });

  std::vector<int64_t> counts(static_cast<size_t>(n), 0);
  int64_t cap = 0;
  for (const Assignment& a : asn) {
    counts[static_cast<size_t>(a.slot)]++;
    cap = std::max(cap, counts[static_cast<size_t>(a.slot)]);
  }

  // Per-slot position of each assignment, in sorted order (python's
  // arange - starts[e_sorted]); used by the capacity bound.
  {
    std::vector<int64_t> run(static_cast<size_t>(n), 0);
    for (Assignment& a : asn) {
      a.in_slot = run[static_cast<size_t>(a.slot)]++;
    }
  }

  int64_t dropped = 0;
  if (capacity_factor != 0.0) {
    const int64_t limit = std::max<int64_t>(1, static_cast<int64_t>(
        std::ceil(capacity_factor * static_cast<double>(total) / static_cast<double>(n))));
    if (cap > limit) {
      for (Assignment& a : asn) {
        a.kept = a.in_slot < limit;
        if (!a.kept) {
          ++dropped;
        }
      }
    } else {
      for (Assignment& a : asn) {
        a.kept = true;
      }
    }
  }
  if (stats) {
    stats->routed += total;
    stats->dropped += dropped;
  }

  // Buffered expert eval per slot: xs -> silu(xs@W1^T)*(xs@W3^T) -> @W2^T.
  const float* w1p = W1.ptr<float>();
  const float* w3p = W3.ptr<float>();
  const float* w2p = W2.ptr<float>();

  std::vector<std::vector<float>> y_slots(
      static_cast<size_t>(n));
  std::vector<std::vector<std::pair<int64_t, float>>> items(
      static_cast<size_t>(n));

  for (const Assignment& a : asn) {
    if (a.kept) {
      items[static_cast<size_t>(a.slot)].push_back({a.token, a.weight});
    }
  }

  std::vector<float> h(static_cast<size_t>(cap) * static_cast<size_t>(dff));
  std::vector<float> xrows(static_cast<size_t>(cap) * static_cast<size_t>(D));
  for (int s = 0; s < n; ++s) {
    const size_t cs = items[static_cast<size_t>(s)].size();
    if (cs == 0) {
      continue;
    }
    for (size_t i = 0; i < cs; ++i) {
      const int64_t t = items[static_cast<size_t>(s)][i].first;
      std::memcpy(xrows.data() + i * static_cast<size_t>(D), xp + static_cast<size_t>(t) * static_cast<size_t>(D),
                  static_cast<size_t>(D) * sizeof(float));
    }
    for (size_t i = 0; i < cs; ++i) {
      const float* xrow = xrows.data() + i * static_cast<size_t>(D);
      for (int64_t f = 0; f < dff; ++f) {
        double a1 = 0.0;
        double a3 = 0.0;
        const float* r1 = w1p + (static_cast<size_t>(s) * static_cast<size_t>(dff) + static_cast<size_t>(f)) * static_cast<size_t>(D);
        const float* r3 = w3p + (static_cast<size_t>(s) * static_cast<size_t>(dff) + static_cast<size_t>(f)) * static_cast<size_t>(D);
        for (int64_t dd = 0; dd < D; ++dd) {
          const double xv = static_cast<double>(xrow[dd]);
          a1 += xv * static_cast<double>(r1[dd]);
          a3 += xv * static_cast<double>(r3[dd]);
        }
        h[i * static_cast<size_t>(dff) + static_cast<size_t>(f)] =
            silu1(static_cast<float>(a1)) * static_cast<float>(a3);
      }
    }
    std::vector<float>& ys = y_slots[static_cast<size_t>(s)];
    ys.assign(cs * static_cast<size_t>(D), 0.0f);
    for (size_t i = 0; i < cs; ++i) {
      const float* hrow = h.data() + i * static_cast<size_t>(dff);
      float* yrow = ys.data() + i * static_cast<size_t>(D);
      for (int64_t dd = 0; dd < D; ++dd) {
        double acc = 0.0;
        const float* w2r = w2p + (static_cast<size_t>(s) * static_cast<size_t>(D) + static_cast<size_t>(dd)) * static_cast<size_t>(dff);
        for (int64_t f = 0; f < dff; ++f) {
          acc += static_cast<double>(hrow[f]) * static_cast<double>(w2r[f]);
        }
        yrow[dd] = static_cast<float>(acc);
      }
    }
  }

  // out[t] += y * w in kept (slot-major) assignment order -- index_add parity.
  mt::Tensor result = zeros2d(N, D);
  float* rp = result.ptr<float>();
  std::vector<double> acc_rows(static_cast<size_t>(N) * static_cast<size_t>(D), 0.0);
  for (int s = 0; s < n; ++s) {
    const size_t cs = items[static_cast<size_t>(s)].size();
    const float* ys = y_slots[static_cast<size_t>(s)].data();
    for (size_t i = 0; i < cs; ++i) {
      const int64_t t = items[static_cast<size_t>(s)][i].first;
      const float w = items[static_cast<size_t>(s)][i].second;
      double* outrow = acc_rows.data() + static_cast<size_t>(t) * static_cast<size_t>(D);
      for (int64_t dd = 0; dd < D; ++dd) {
        outrow[dd] += static_cast<double>(w) * static_cast<double>(ys[i * static_cast<size_t>(D) + static_cast<size_t>(dd)]);
      }
    }
  }
  for (int64_t j = 0; j < N * D; ++j) {
    rp[j] = static_cast<float>(acc_rows[static_cast<size_t>(j)]);
  }
  return result;
}

}  // namespace minagi