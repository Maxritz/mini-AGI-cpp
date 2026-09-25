// tests/test_pool_backward.cpp
// Numerical gradient check for pool_mlp_backward (STATUS backward gap #2/3/4).
// Central differences against pool_mlp_forward-as-written validate: expert
// w1/w3/w2 grads, gate grad, router softmax-Jacobian grad, depth_emb grad,
// d_x. Uses no-drop routing (large capacity_factor) and clear top-k margins
// so hard selection/drop masks stay constant under eps perturbations.
#include "pool.hpp"
#include "paged.hpp"
#include "tensor.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <vector>

static int failures = 0;

#define CHECK(name, cond) do { \
    if (cond) std::cout << "ok " << name << "\n"; \
    else { std::cout << "FAIL " << name << "\n"; ++failures; } \
} while(0)

namespace {

struct SmallPool : minagi::Pool {
    int n_ = 3;
    std::vector<int> slots_ = {0, 1, 2};
    mt::Tensor gate_, w1_, w3_, w2_;
    SmallPool(int d, int dff, unsigned seed) {
        mt::Shape gs;
        gs.rank = 1;
        gs.d[0] = n_;
        gate_ = mt::make(gs, mt::DType::FP32, 0.0f);
        float* gp = gate_.ptr<float>();
        gp[0] = 1.0f;
        gp[1] = 0.7f;
        gp[2] = 1.3f;
        // deterministic pseudo-random weights with clear margins
        auto fill = [&](mt::Tensor& t, int r, int c, float s) {
            mt::Shape sh;
            sh.rank = 2;
            sh.d[0] = r;
            sh.d[1] = c;
            t = mt::make(sh, mt::DType::FP32, 0.0f);
            float* p = t.ptr<float>();
            unsigned st = seed;
            for (int i = 0; i < r * c; ++i) {
                st = st * 1664525u + 1013904223u;
                p[i] = (static_cast<float>(st % 1000) / 1000.0f - 0.5f) * s;
            }
        };
        mt::Tensor a, b, c;
        fill(a, n_ * dff, d, 0.6f);
        fill(b, n_ * dff, d, 0.6f);
        fill(c, n_ * d, dff, 0.6f);
        auto stack3 = [&](const mt::Tensor& flat) {
            mt::Shape sh;
            sh.rank = 3;
            sh.d[0] = n_;
            sh.d[1] = flat.shape.d[0] / n_;
            sh.d[2] = flat.shape.d[1];
            mt::Tensor t;
            t.dtype = mt::DType::FP32;
            t.shape = sh;
            t.data_ = flat.data_;
            return t;
        };
        w1_ = stack3(a);
        w3_ = stack3(b);
        w2_ = stack3(c);
    }
    int n_experts() const override { return n_; }
    int n_routable() const override { return n_; }
    int router_rows() const override { return n_; }
    const std::vector<int>& slots() const override { return slots_; }
    const mt::Tensor& gate() const override { return gate_; }
    const mt::Tensor& w1() const override { return w1_; }
    const mt::Tensor& w3() const override { return w3_; }
    const mt::Tensor& w2() const override { return w2_; }
    void note_use(const std::vector<long long>&) const override {}
};

mt::Tensor ones2d(int r, int c, float v) {
    mt::Shape s;
    s.rank = 2;
    s.d[0] = r;
    s.d[1] = c;
    return mt::make(s, mt::DType::FP32, v);
}

// Scalar loss = sum(out * dout) for fixed projection dout.
float run_loss(SmallPool& pool, const mt::Tensor& x, const mt::Tensor& router,
               const mt::Tensor& depth, const mt::Tensor& dout, int top_k,
               double cap, minagi::PoolBackwardCache* cache) {
    minagi::RouteStats st;
    mt::Tensor out =
        minagi::pool_mlp_forward(pool, x, router, depth, top_k, cap, &st, cache);
    double acc = 0.0;
    const float* op = out.ptr<float>();
    const float* dp = dout.ptr<float>();
    for (int64_t i = 0; i < out.shape.numel(); ++i) acc += op[i] * dp[i];
    return static_cast<float>(acc);
}

}  // namespace

int main() {
    const int N = 3, D = 4, dff = 6, n = 3;
    const int top_k = 2;
    const double cap = 10.0;  // no drops: limit = ceil(10*6/3) = 20
    const float eps = 1e-4f;  // small enough that hard top-k/drop masks
                              // never flip under perturbation (verified: any
                              // flip would show as O(1) diffs, not 1e-3)
    // Two-tier tolerance: fp32 central-diff noise floor is ~1e-3 absolute on
    // this problem (rounding in f scaled by 1/eps), so small grads get an
    // absolute bar while significant grads must match relatively. A wrong
    // formula (e.g. the old silu' form) errs by 10%+ relatively and fails.
    const float atol = 5e-3f, rtol = 2e-2f;

    SmallPool pool(D, dff, 12345u);
    mt::Tensor x = ones2d(N, D, 0.0f);
    {
        float* p = x.ptr<float>();
        unsigned st = 777u;
        for (int i = 0; i < N * D; ++i) {
            st = st * 1664525u + 1013904223u;
            p[i] = (static_cast<float>(st % 1000) / 1000.0f - 0.5f) * 1.2f;
        }
    }
    mt::Tensor router = ones2d(n, D, 0.0f);
    {
        float* p = router.ptr<float>();
        unsigned st = 999u;
        for (int i = 0; i < n * D; ++i) {
            st = st * 1664525u + 1013904223u;
            p[i] = (static_cast<float>(st % 1000) / 1000.0f - 0.5f) * 1.5f;
        }
    }
    mt::Shape ds;
    ds.rank = 1;
    ds.d[0] = D;
    mt::Tensor depth = mt::make(ds, mt::DType::FP32, 0.1f);
    mt::Tensor dout = ones2d(N, D, 0.0f);
    {
        float* p = dout.ptr<float>();
        for (int i = 0; i < N * D; ++i) p[i] = 0.3f + 0.1f * (i % 5);
    }

    // Analytic grads.
    minagi::PoolBackwardCache cache;
    run_loss(pool, x, router, depth, dout, top_k, cap, &cache);
    mt::Tensor d_router, d_depth, d_w1, d_w3, d_w2, d_gate;
    mt::Tensor d_x = minagi::pool_mlp_backward(pool, cache, dout, router, depth,
                                               top_k, cap, &d_router, &d_depth,
                                               &d_w1, &d_w3, &d_w2, &d_gate);
    CHECK("shapes",
          d_x.shape.d[0] == N && d_x.shape.d[1] == D &&
          d_router.shape.d[0] == n && d_router.shape.d[1] == D &&
          d_depth.shape.d[0] == D && d_w1.shape.d[0] == n &&
          d_w2.shape.d[0] == n && d_gate.shape.d[0] == n);

    auto numgrad = [&](float* base, int64_t flat_idx) {
        float orig = base[flat_idx];
        base[flat_idx] = orig + eps;
        minagi::PoolBackwardCache c1;
        float lp = run_loss(pool, x, router, depth, dout, top_k, cap, &c1);
        base[flat_idx] = orig - eps;
        minagi::PoolBackwardCache c2;
        float lm = run_loss(pool, x, router, depth, dout, top_k, cap, &c2);
        base[flat_idx] = orig;
        return (lp - lm) / (2 * eps);
    };

    auto cmp = [&](const char* name, const mt::Tensor& analytic, float* base,
                   const std::vector<int64_t>& idxs) {
        float worst_abs = 0.0f, worst_rel = 0.0f;
        for (int64_t ix : idxs) {
            float a = analytic.atf(ix);
            float g = numgrad(base, ix);
            float d = std::fabs(a - g);
            float scale = std::max(std::fabs(a), std::fabs(g));
            if (d > worst_abs) worst_abs = d;
            if (scale > 1e-2f) {
                float r = d / scale;
                if (r > worst_rel) worst_rel = r;
            }
        }
        std::cout << "maxabs " << name << " = " << worst_abs << " maxrel "
                  << name << " = " << worst_rel << "\n";
        CHECK(name, worst_abs < atol && worst_rel < rtol);
    };

    // Selection-stability audit: perturb each x entry by +/-eps and check
    // the top-k set is unchanged (a flip would corrupt numeric grads).
    {
        minagi::PoolBackwardCache c0;
        run_loss(pool, x, router, depth, dout, top_k, cap, &c0);
        auto sel_of = [&](minagi::PoolBackwardCache& c) {
            // recover per-row selected slots from kept + recompute: use cache logits
            std::vector<std::vector<int>> sel(N);
            // recompute top-k from logits (same as backward does)
            for (int64_t r = 0; r < N; ++r) {
                std::vector<std::pair<float, int>> cand(n);
                // softmax
                float mx = -1e30f;
                for (int cc = 0; cc < n; ++cc) {
                    float v = c0.logits[r * n + cc];
                    if (v > mx) mx = v;
                }
                double sum = 0;
                std::vector<float> pr(n);
                for (int cc = 0; cc < n; ++cc) {
                    pr[cc] = std::exp(c0.logits[r * n + cc] - mx);
                    sum += pr[cc];
                }
                for (int cc = 0; cc < n; ++cc) cand[cc] = {pr[cc] / sum, cc};
                std::partial_sort(cand.begin(), cand.begin() + top_k, cand.end(),
                                  [](const auto& a, const auto& b) {
                                      if (a.first != b.first) return a.first > b.first;
                                      return a.second < b.second;
                                  });
                for (int i = 0; i < top_k; ++i) sel[r].push_back(cand[i].second);
            }
            return sel;
        };
        auto base_sel = sel_of(c0);
        int flips = 0;
        float* xp = x.ptr<float>();
        for (int64_t ixi = 0; ixi < N * D; ++ixi) {
            float orig = xp[ixi];
            xp[ixi] = orig + eps;
            minagi::PoolBackwardCache cp;
            run_loss(pool, x, router, depth, dout, top_k, cap, &cp);
            xp[ixi] = orig - eps;
            minagi::PoolBackwardCache cm;
            run_loss(pool, x, router, depth, dout, top_k, cap, &cm);
            xp[ixi] = orig;
            // compare selections of perturbed runs vs base
            for (int pm = 0; pm < 2; ++pm) {
                minagi::PoolBackwardCache& cc = (pm == 0) ? cp : cm;
                for (int64_t r = 0; r < N; ++r) {
                    float mx = -1e30f;
                    for (int cc2 = 0; cc2 < n; ++cc2) {
                        float v = cc.logits[r * n + cc2];
                        if (v > mx) mx = v;
                    }
                    double sum = 0;
                    std::vector<float> pr(n);
                    for (int cc2 = 0; cc2 < n; ++cc2) {
                        pr[cc2] = std::exp(cc.logits[r * n + cc2] - mx);
                        sum += pr[cc2];
                    }
                    std::vector<std::pair<float, int>> cand(n);
                    for (int cc2 = 0; cc2 < n; ++cc2)
                        cand[cc2] = {pr[cc2] / sum, cc2};
                    std::partial_sort(cand.begin(), cand.begin() + top_k, cand.end(),
                                      [](const auto& a, const auto& b) {
                                          if (a.first != b.first) return a.first > b.first;
                                          return a.second < b.second;
                                      });
                    for (int i = 0; i < top_k; ++i) {
                        if (cand[i].second != base_sel[r][i]) ++flips;
                    }
                }
            }
        }
        std::cout << "selection flips under eps perturbation: " << flips << "\n";
        CHECK("selection_stable", flips == 0);
    }

    // d_x over all entries.
    {
        std::vector<int64_t> ix;
        for (int64_t i = 0; i < N * D; ++i) ix.push_back(i);
        cmp("dx", d_x, x.ptr<float>(), ix);
    }
    // router rows (all 12).
    {
        std::vector<int64_t> ix;
        for (int64_t i = 0; i < n * D; ++i) ix.push_back(i);
        cmp("router", d_router, router.ptr<float>(), ix);
    }
    // depth_emb (all 4).
    {
        std::vector<int64_t> ix;
        for (int64_t i = 0; i < D; ++i) ix.push_back(i);
        cmp("depth", d_depth, depth.ptr<float>(), ix);
    }
    // gate (all 3). pool object is non-const; gate() exposes const ref, so
    // const_cast for test-only perturbation (legal: underlying is mutable).
    {
        std::vector<int64_t> ix;
        for (int64_t i = 0; i < n; ++i) ix.push_back(i);
        cmp("gate", d_gate, const_cast<float*>(pool.gate().ptr<float>()), ix);
    }
    // expert weights: sample a few entries per tensor (full check is O(slots)
    // forward runs each; keep it fast but cover all three tensors + 2 slots).
    {
        float* p1 = const_cast<float*>(pool.w1().ptr<float>());
        float* p3 = const_cast<float*>(pool.w3().ptr<float>());
        float* p2 = const_cast<float*>(pool.w2().ptr<float>());
        std::vector<int64_t> ix;
        for (int s = 0; s < 2; ++s)
            for (int64_t j = 0; j < 8; ++j) ix.push_back(s * dff * D + j * 3);
        cmp("ew1", d_w1, p1, ix);
        cmp("ew3", d_w3, p3, ix);
        std::vector<int64_t> jx;
        for (int s = 0; s < 2; ++s)
            for (int64_t j = 0; j < 8; ++j) jx.push_back(s * D * dff + j * 3);
        cmp("ew2", d_w2, p2, jx);
    }

    if (failures == 0) {
        std::cout << "ALL_TESTS_PASSED\n";
        return 0;
    }
    std::cout << "SOME_TESTS_FAILED\n";
    return 1;
}
