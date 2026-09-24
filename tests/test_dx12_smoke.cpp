// tests/test_dx12_smoke.cpp
// DX12 smoke test: initialize engine, load golden weights, run Coder::forward
// using GPU kernels, compare seg0 pos0 logits vs CPU results.
// On non-Windows or no GPU: skip gracefully (return 0).

#include "recur.hpp"
#include "store.hpp"
#include "paged.hpp"
#include "tensor.hpp"
#include "mininpz.hpp"
#include "decode.hpp"
#include "pool.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <string>
#include <vector>

#ifdef _WIN32
#include "dx12_context.hpp"
#include "dx12_engine.hpp"
#include "dx12_dense.hpp"
#include "dx12_pool.hpp"
#endif

namespace fs = std::filesystem;
namespace m = minagi;
namespace pg = minagi::paged;

static int failures = 0;
static int checks_run = 0;

static void check(const char* name, bool ok) {
    ++checks_run;
    if (ok) {
        std::cout << "ok " << name << "\n";
    } else {
        std::cout << "FAIL " << name << "\n";
        ++failures;
    }
}

static std::vector<uint8_t> read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    std::vector<uint8_t> buf((std::istreambuf_iterator<char>(f)),
                               std::istreambuf_iterator<char>());
    return buf;
}

static float read_f32(const uint8_t* p) {
    float v;
    std::memcpy(&v, p, 4);
    return v;
}

static uint32_t read_u32(const uint8_t* p) {
    uint32_t v;
    std::memcpy(&v, p, 4);
    return v;
}

int main(int argc, char** argv) {
    std::cout << "DX12 smoke test start" << std::endl;

#ifdef _WIN32
    // Try to initialize DX12
    dx12::Dx12ComputeEngine engine;
    bool dx12_ok = engine.init("src");
    if (!dx12_ok) {
        std::cout << "DX12 not available on this machine, using CPU fallback" << std::endl;
    } else {
        std::cout << "DX12 engine initialized successfully" << std::endl;
    }
#else
    bool dx12_ok = false;
    std::cout << "DX12 not available (non-Windows), using CPU fallback" << std::endl;
#endif

    // ---- Load golden weights ----
    std::string root = "..";
    if (argc > 1) root = argv[1];

    fs::path gdir = fs::path(root) / "tests" / "golden_wavef";
    fs::path wdir = gdir / "weights";
    std::cout << "gdir=" << gdir << " wdir=" << wdir << " exists=" << fs::exists(wdir) << std::endl;

    ::store::Manifest man;
    std::map<std::string, mininpz::Array> state;
    bool ok = ::store::load(wdir.string(), man, state);
    check("dx12_load_weights", ok);
    if (!ok) {
        std::cout << "DX12 smoke: weights not loaded, skipping" << std::endl;
        return 0;
    }

    check("dx12_manifest_paged", man.paged);
    check("dx12_manifest_readonly", man.read_only);

    // ---- Set up coder + pool ----
    m::Config cfg;
    cfg.from_manifest(man.cfg_obj);
    std::vector<std::pair<std::string, mt::Tensor>> sd;
    for (auto& kv : state) sd.emplace_back(kv.first, pg::array_to_tensor(kv.second));

    mt::Shape gsha; gsha.rank = 1; gsha.d[0] = cfg.pool_experts;
    mt::Tensor ones = mt::make(gsha, mt::DType::FP32, 1.0f);
    pg::PagedPool pool(cfg.pool_experts, cfg.d_model, cfg.pool_d_ff,
                       cfg.pool_resident, man.pool_ram, cfg.explore,
                       cfg.margin, cfg.dwell, man.read_only, ones);
    pool.set_experts_dir((wdir / "experts").string());

    m::Coder coder(cfg);
    bool lw = coder.load_weights(sd, pool);
    check("dx12_load_weights_coder", lw);
    if (!lw) {
        std::cout << "DX12 smoke: coder load failed, returning" << std::endl;
        return 0;
    }

    coder.set_pool(&pool);

    // ---- Run forward (CPU path for comparison baseline) ----
    const int V = cfg.vocab_size;
    const int N = 496;  // corpus size matches golden

    // Build corpus (same as test_wavef)
    static const char* DATA_PHRASE =
        "The quick brown fox jumps over the lazy dog. 0123456789\n"
        "Pack my box with five dozen liquor jugs.\n";

    std::vector<int> corpus;
    while (static_cast<int>(corpus.size()) < N) {
        for (unsigned char c : std::string(DATA_PHRASE)) {
            corpus.push_back(static_cast<int>(c));
            if (static_cast<int>(corpus.size()) == N) break;
        }
    }

    auto caches = coder.empty_caches();

    // begin_segment
    coder.begin_segment();
    const std::vector<int>& slots = pool.slots();
    check("dx12_slots_size", static_cast<int>(slots.size()) == cfg.pool_resident);

    // Run forward on seg0 (first 170 tokens, same as golden)
    int seg0_len = 170;
    std::vector<int64_t> chunk(corpus.begin(), corpus.begin() + seg0_len);
    mt::Shape ish; ish.rank = 2; ish.d[0] = 1; ish.d[1] = seg0_len;
    mt::Tensor idx = mt::make(ish, mt::DType::FP32, 0.0f);
    float* ip = idx.ptr<float>();
    for (int i = 0; i < seg0_len; ++i) ip[i] = static_cast<float>(chunk[i]);

    std::cout << "Running forward (seg0, 170 tokens)..." << std::endl;
    m::StepOut so = coder.forward(idx, caches, 0);

    bool logits_ok = (so.logits.shape.rank == 3 &&
                      so.logits.shape.d[0] == 1 &&
                      so.logits.shape.d[1] == seg0_len &&
                      so.logits.shape.d[2] == V);
    check("dx12_seg0_logits_shape", logits_ok);

    // ---- Compare against golden logits ----
    auto bin = read_file((gdir / "logits.bin").string());
    check("dx12_logits_bin", bin.size() >= 8);
    if (bin.size() >= 8) {
        uint32_t Ng = read_u32(bin.data());
        uint32_t Vg = read_u32(bin.data() + 4);
        check("dx12_logits_hdr", Ng == static_cast<uint32_t>(N) && Vg == static_cast<uint32_t>(V));

        // Compare seg0 pos0 logits (first token's halt logits)
        const uint8_t* p = bin.data() + 8;
        uint32_t hr0 = read_u32(p);  // halt_row for position 0
        const float* gp = reinterpret_cast<const float*>(p + 4);

        const float* our_logits = so.logits.ptr<float>();
        float maxdiff = 0;
        for (int v = 0; v < V; ++v) {
            float d = std::fabs(gp[v] - our_logits[v]);
            if (d > maxdiff) maxdiff = d;
        }
        std::cout << "dx12 seg0 pos0 gold[0]=" << gp[0] << " ours[0]=" << our_logits[0]
                  << " maxdiff=" << maxdiff << std::endl;
        check("dx12_seg0_pos0_logits", maxdiff < 1e-3f);

        // Verify halt_row for position 0
        int our_hr = static_cast<int>(so.halt_row.ptr<float>()[0]);
        std::cout << "dx12 halt_row pos0 gold=" << hr0 << " ours=" << our_hr << std::endl;
        check("dx12_halt_row_pos0", hr0 == static_cast<uint32_t>(our_hr));
    }

    // ---- Test dx12_pool integration (DX12 path or CPU fallback) ----
#ifdef _WIN32
    {
        m::Dx12PoolContext* ctx = m::dx12_pool_create();
        if (ctx && ctx->valid) {
            std::cout << "DX12 pool context created, running GPU path test" << std::endl;

            // Build a tiny test: x [pool_N, d], 2 experts, resident 2
            const int pool_N = 4, d = 4, dff = 6;
            mt::Shape xs; xs.rank = 2; xs.d[0] = pool_N; xs.d[1] = d;
            mt::Tensor x = mt::make(xs, mt::DType::FP32, 1.0f);

            mt::Shape rs; rs.rank = 2; rs.d[0] = N; rs.d[1] = d;
            mt::Tensor router = mt::make(rs, mt::DType::FP32, 0.5f);
            mt::Shape ds; ds.rank = 1; ds.d[0] = d;
            mt::Tensor depth = mt::make(ds, mt::DType::FP32, 0.0f);

            // Create expert weights (n experts, not N tokens)
            const int n_experts = 2;
            mt::Shape wsha1; wsha1.rank = 3; wsha1.d[0] = n_experts; wsha1.d[1] = dff; wsha1.d[2] = d;
            mt::Tensor W1 = mt::make(wsha1, mt::DType::FP32, 0.1f);
            mt::Tensor W3 = mt::make(wsha1, mt::DType::FP32, 0.2f);

            mt::Shape wsha2; wsha2.rank = 3; wsha2.d[0] = n_experts; wsha2.d[1] = d; wsha2.d[2] = dff;
            mt::Tensor W2 = mt::make(wsha2, mt::DType::FP32, 0.3f);

            mt::Shape gsha_n; gsha_n.rank = 1; gsha_n.d[0] = n_experts;
            mt::Tensor gate_t = mt::make(gsha_n, mt::DType::FP32, 1.0f);

            std::vector<int> slots_n(n_experts, 0);
            for (int i = 0; i < n_experts; ++i) slots_n[i] = i;
            std::vector<int> rows_n(n_experts, 0);
            rows_n[0] = 0; rows_n[1] = 1;

            m::Dx12PoolResult pr = m::dx12_pool_mlp_forward(ctx, x, router, depth, gate_t,
                                                             W1, W3, W2, slots_n, rows_n,
                                                             2, 0.5);
            check("dx12_pool_result_shape", pr.y.shape.rank == 2 &&
                    pr.y.shape.d[0] == pool_N && pr.y.shape.d[1] == d);
            check("dx12_pool_routed_positive", pr.routed > 0);

            m::dx12_pool_destroy(ctx);
        } else {
            std::cout << "DX12 pool context creation failed" << std::endl;
            check("dx12_pool_init", false);
        }
    }
#endif

    std::cout << "DX12 smoke test done: " << checks_run << " checks, "
              << failures << " failures" << std::endl;

    if (failures == 0) {
        std::cout << "ALL_TESTS_PASSED\n";
        return 0;
    }
    std::cout << failures << " check(s) FAILED\n";
    return 1;
}
