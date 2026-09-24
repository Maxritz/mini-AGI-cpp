// tests/test_wavef.cpp
// Wave F integration test: paged pool + pondering + greedy decode against the
// golden reference in tests/golden_wavef/. Run with the working directory set
// to tests/ (CMAKE_WORKING_DIRECTORY); argv[1] optionally overrides the path
// to the repo root (defaults to "..").
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

// ---- binary helpers ----

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

// ---- corpus (section 8 DATA_PHRASE) ----

static const char* DATA_PHRASE =
    "The quick brown fox jumps over the lazy dog. 0123456789\n"
    "Pack my box with five dozen liquor jugs.\n";

static std::vector<int> build_corpus(int N) {
    std::vector<int> out;
    while (static_cast<int>(out.size()) < N) {
        for (unsigned char c : std::string(DATA_PHRASE)) {
            out.push_back(static_cast<int>(c));
            if (static_cast<int>(out.size()) == N) break;
        }
    }
    return out;
}

// ---- tensor helpers ----

static mt::Tensor int_tensor_2d(const std::vector<int32_t>& flat, int R, int C) {
    mt::Shape s;
    s.rank = 2;
    s.d[0] = R;
    s.d[1] = C;
    mt::Tensor t = mt::make(s, mt::DType::FP32, 0.0f);
    float* p = t.ptr<float>();
    for (size_t i = 0; i < flat.size(); ++i) p[i] = static_cast<float>(flat[i]);
    return t;
}

static mt::Tensor int_tensor_2d_long(const std::vector<int64_t>& flat, int R, int C) {
    mt::Shape s;
    s.rank = 2;
    s.d[0] = R;
    s.d[1] = C;
    mt::Tensor t = mt::make(s, mt::DType::FP32, 0.0f);
    float* p = t.ptr<float>();
    for (size_t i = 0; i < flat.size(); ++i) p[i] = static_cast<float>(flat[i]);
    return t;
}

static int32_t argmax_row(const mt::Tensor& row, int V) {
    const float* p = row.ptr<float>();
    int best = 0;
    float bv = p[0];
    for (int i = 1; i < V; ++i) {
        if (p[i] > bv) { bv = p[i]; best = i; }
    }
    return best;
}

// ---- main ----

int main(int argc, char** argv) {
    std::string root = "..";
    if (argc > 1) root = argv[1];
    std::cout << "wavef start root=" << root << std::endl;
    fs::path gdir = fs::path(root) / "tests" / "golden_wavef";
    fs::path wdir = gdir / "weights";
    std::cout << "gdir=" << gdir << " wdir=" << wdir << " exists=" << fs::exists(wdir) << std::endl;

    // ---- check1: store load key count, shapes, head==tok_emb ----
    {
        ::store::Manifest man;
        std::map<std::string, mininpz::Array> state;
        bool ok = ::store::load(wdir.string(), man, state);
        check("check1_loads", ok);
        check("check1_paged", man.paged);
        check("check1_read_only", man.read_only);
        check("check1_pool_ram4", man.pool_ram == 4);
        bool core_present = state.count("tok_emb.weight");
        bool head_present = state.count("head.weight");
        check("check1_tok_emb", core_present);
        check("check1_head", head_present);
        if (core_present && head_present) {
            const auto& a = state["tok_emb.weight"];
            const auto& b = state["head.weight"];
            bool same = (a.dtype == b.dtype && a.shape == b.shape &&
                         a.bytes == b.bytes);
            check("check1_head_is_tok_emb", same);
        }
        bool gate_present = state.count("pool.gate");
        bool sr_present = state.count("pool.segment_router.weight");
        check("check1_gate", gate_present);
        check("check1_segment_router", sr_present);
        check("check1_expert0", state.count("pool.experts.0.w1.weight") &&
               state.count("pool.experts.0.w3.weight") &&
               state.count("pool.experts.0.w2.weight"));
        check("check1_expert3_block",
              state.count("pool.experts.3.blocks.0.w1.weight") ||
              state.count("pool.experts.3.w1.weight"));
        if (core_present) {
            const auto& tok = state["tok_emb.weight"];
            check("check1_tok_shape", tok.shape.rank == 2 &&
                  static_cast<int>(tok.shape.d[1]) == man.d_model &&
                  static_cast<int>(tok.shape.d[0]) > man.d_model);
        }
        std::cout << "check1 done n_experts=" << man.n_experts << " cfg_obj_type=" << static_cast<int>(man.cfg_obj.t) << std::endl;
    }

    // ---- check2: Tiers.fetch vs npz ----
    {
        std::cout << "check2 start" << std::endl;
        ::store::Manifest man;
        std::map<std::string, mininpz::Array> state;
        bool ok = ::store::load(wdir.string(), man, state);
        check("check2_load", ok);
        if (!ok) return failures ? 1 : 0;
        fs::path edir = wdir / "experts";
        std::cout << "check2 edir=" << edir << " dm=" << man.d_model << " dff=" << man.d_ff << " ram=" << man.pool_ram << " ro=" << man.read_only << std::endl;
        pg::Tiers tiers(edir.string(), man.d_model, man.d_ff, man.pool_ram,
                        man.read_only);
        mt::Tensor w1, w3, w2;
        std::cout << "check2 calling fetch" << std::endl;
        bool got = tiers.fetch(0, w1, w3, w2);
        std::cout << "check2 fetch=" << got << " w1size=" << w1.data_.size() << std::endl;
        check("check2_tiers_fetch0", got);
        auto find = [&](const std::string& k) -> const mininpz::Array* {
            auto it = state.find(k);
            return it == state.end() ? nullptr : &it->second;
        };
        const mininpz::Array* a_w1 = find("pool.experts.0.w1.weight");
        std::cout << "check2 a_w1=" << (a_w1?"yes":"no") << (a_w1 ? (" sz=" + std::to_string(a_w1->bytes.size())) : "") << std::endl;
        if (got && a_w1) {
            bool eq = (w1.data_ == a_w1->bytes);
            std::cout << "check2 w1_match=" << eq << " dsz=" << w1.data_.size() << " asz=" << a_w1->bytes.size() << std::endl;
            check("check2_w1_match", eq);
        } else {
            check("check2_w1_match", false);
        }
        const auto& tc = tiers.counters();
        std::cout << "check2 counters reads=" << tc.reads << " wb=" << tc.writebacks << std::endl;
        check("check2_reads", tc.reads >= 1);
        check("check2_readonly_nodirty", man.read_only && tc.writebacks == 0);
        std::cout << "check2 end" << std::endl;
    }

    // ---- check3: from_manifest ----
    {
        ::store::Manifest man;
        std::map<std::string, mininpz::Array> state;
        bool ok = ::store::load(wdir.string(), man, state);
        check("check3_load", ok);
        if (!ok) return failures ? 1 : 0;
        m::Config cfg;
        bool fm = cfg.from_manifest(man.cfg_obj);
        check("check3_from_manifest", fm);
        check("check3_vocab", cfg.vocab_size == 265);
        check("check3_d_model", cfg.d_model == 24);
        check("check3_n_head", cfg.n_head == 3);
        check("check3_d_ff", cfg.d_ff == 48);
        check("check3_pool_experts", cfg.pool_experts == 6);
        check("check3_pool_d_ff", cfg.pool_d_ff == 32);
        check("check3_pool_top_k", cfg.pool_top_k == 2);
        check("check3_max_steps", cfg.max_steps == 5);
        check("check3_halt_thresh", std::fabs(cfg.halt_thresh - 0.9) < 1e-9);
        check("check3_n_prelude", cfg.n_prelude == 1);
        check("check3_n_recur", cfg.n_recur == 2);
        check("check3_pool_resident", cfg.pool_resident == 3);
    }

    // ---- check4: pool_mlp_forward forced-drop vs mt reference ----
    {
        // Build a tiny pool with 4 experts, resident 2; force a drop by setting
        // capacity_factor tiny and confirming pool_mlp_forward still returns [T,d].
        const int n = 4, d = 4, dff = 6;
        mt::Shape gsha; gsha.rank = 1; gsha.d[0] = n;
        mt::Tensor gate = mt::make(gsha, mt::DType::FP32, 0.0f);
        pg::PagedPool pool(n, d, dff, 2 /*resident*/, 4 /*ram*/, 0.15, 0.10, 4, true, gate);
        // x [2, d] all ones
        mt::Shape xs; xs.rank = 2; xs.d[0] = 2; xs.d[1] = d;
        mt::Tensor x = mt::make(xs, mt::DType::FP32, 1.0f);
        // router [n, d] zeros, depth_emb [d] zeros
        mt::Shape rs; rs.rank = 2; rs.d[0] = n; rs.d[1] = d;
        mt::Tensor router = mt::make(rs, mt::DType::FP32, 0.0f);
        mt::Tensor depth = mt::make(gsha, mt::DType::FP32, 0.0f);
        m::RouteStats st;
        mt::Tensor y = m::pool_mlp_forward(pool, x, router, depth, 2, 0.5, &st);
        check("check4_shape", y.shape.rank == 2 && y.shape.d[0] == 2 && y.shape.d[1] == d);
        check("check4_routed", st.routed > 0);
    }

    // ---- check5: begin_segment SEG0 + TIERS0 ----
    {
        m::Config cfg;
        ::store::Manifest man;
        std::map<std::string, mininpz::Array> state;
        bool ok = ::store::load(wdir.string(), man, state);
        check("check5_load", ok);
        if (!ok) return failures ? 1 : 0;
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
        check("check5_load_weights", coder.load_weights(sd, pool));
        coder.set_pool(&pool);

        // begin_segment seeds first swap_to(choose()) -> expect [5,4,3]
        const std::vector<int> expect{5, 4, 3};
        coder.begin_segment();
        const std::vector<int>& after = pool.slots();
        check("check5_seg_slots_size", static_cast<int>(after.size()) == cfg.pool_resident);
        check("check5_seg_slots_size3", static_cast<int>(after.size()) == 3);
        bool ok53 = (static_cast<int>(after.size()) == 3);
        for (int i = 0; i < 3; ++i) {
            if (after[i] != expect[i]) ok53 = false;
        }
        check("check5_seg0_543", ok53);
        const auto& tc = pool.counters();
        check("check5_tiers0", tc.reads == 3 && tc.hits == 0 &&
              tc.evictions == 0 && tc.writebacks == 0);
    }

    // ---- check6: full golden run (phase A 170/170/156, replay, 32 decode) ----
    {
        m::Config cfg;
        ::store::Manifest man;
        std::map<std::string, mininpz::Array> state;
        bool ok = ::store::load(wdir.string(), man, state);
        check("check6_load", ok);
        if (!ok) return failures ? 1 : 0;
        cfg.from_manifest(man.cfg_obj);
        check("check6_cfg_use_pool", cfg.use_pool);
        std::vector<std::pair<std::string, mt::Tensor>> sd;
        for (auto& kv : state) sd.emplace_back(kv.first, pg::array_to_tensor(kv.second));
        mt::Shape gsha; gsha.rank = 1; gsha.d[0] = cfg.pool_experts;
        mt::Tensor ones = mt::make(gsha, mt::DType::FP32, 1.0f);
        pg::PagedPool pool(cfg.pool_experts, cfg.d_model, cfg.pool_d_ff,
                           cfg.pool_resident, man.pool_ram, cfg.explore,
                           cfg.margin, cfg.dwell, man.read_only, ones);
        pool.set_experts_dir((wdir / "experts").string());
        // restore gate from sd (1 + 0.02*arange)
        {
            const mininpz::Array* ga = nullptr;
            for (auto& kv : state) if (kv.first == "pool.gate") ga = &kv.second;
            if (ga) pool.set_gate(pg::array_to_tensor(*ga));
        }
        pool.load_telemetry(man.telemetry);
        m::Coder coder(cfg);
        check("check6_load_weights", coder.load_weights(sd, pool));
        coder.set_pool(&pool);

        const int V = cfg.vocab_size;
        const int N = 496;
        const int Mdec = 32;
        std::vector<int> corpus = build_corpus(N);
        auto caches = coder.empty_caches();

        std::vector<std::pair<int, int>> crows;   // (global idx, halt_row)
        std::vector<float> lrows;                  // halt_row per token
        std::vector<float> llogits;                // [N, V] float
        std::vector<std::vector<float>> summaries;

        int pos = 0;
        const int seg_lens[3] = {170, 170, 156};
        for (int seg = 0; seg < 3; ++seg) {
            int L = seg_lens[seg];
            coder.begin_segment();
            std::vector<int64_t> chunk(corpus.begin() + pos, corpus.begin() + pos + L);
            mt::Tensor idx = int_tensor_2d_long(chunk, 1, L);
            m::StepOut so = coder.forward(idx, caches, pos);
            bool logits_ok = (so.logits.shape.rank == 3 &&
                              so.logits.shape.d[0] == 1 &&
                              so.logits.shape.d[1] == L &&
                              so.logits.shape.d[2] == V);
            check("check6_seg_logits_shape", logits_ok);
            const float* lp = so.logits.ptr<float>();
            for (int j = 0; j < L; ++j) {
                int hr = static_cast<int>(so.halt_row.ptr<float>()[j]);
                crows.push_back({pos + j, hr});
                lrows.push_back(static_cast<float>(hr));
                for (int v = 0; v < V; ++v) llogits.push_back(lp[j * V + v]);
            }
            // summary after prelude (observe ran inside forward)
            const mt::Tensor& sm = pool.summary();
            if (sm.shape.rank >= 1 && sm.shape.d[0] == cfg.d_model) {
                std::vector<float> sv(sm.shape.d[0]);
                std::memcpy(sv.data(), sm.ptr<float>(), sv.size() * sizeof(float));
                summaries.push_back(std::move(sv));
            }
            pos += L;
            if (seg == 0) {
                auto lbin = read_file((gdir / "logits.bin").string());
                const float* lgp = reinterpret_cast<const float*>(lbin.data() + 8 + 4);
                float maxdiff_l = 0;
                for (int v = 0; v < V; ++v) { float d=std::fabs(lgp[v]-llogits[v]); if(d>maxdiff_l)maxdiff_l=d; }
                std::cout << "check6 seg0 pos0 gold=" << lgp[0] << " ours=" << llogits[0] << " maxdiff=" << maxdiff_l << std::endl;
            }
        }

        // phase B: replay + 32 decode
        coder.begin_segment();
        {
            std::vector<int64_t> chunk(corpus.begin(), corpus.begin() + N);
            mt::Tensor idx = int_tensor_2d_long(chunk, 1, N);
            m::StepOut so = coder.forward(idx, caches, 0);
            bool ok = (so.logits.shape.rank == 3 && so.logits.shape.d[1] == N);
            check("check6_replay_logits", ok);
            const mt::Tensor& sm = pool.summary();
            if (sm.shape.rank >= 1 && sm.shape.d[0] == cfg.d_model) {
                std::vector<float> sv(sm.shape.d[0]);
                std::memcpy(sv.data(), sm.ptr<float>(), sv.size() * sizeof(float));
                summaries.push_back(std::move(sv));
            }
        }
        // decode
        std::vector<int> gen;
        std::vector<int64_t> prev(corpus.begin(), corpus.begin() + N);
        int cur = prev.back();
        int offset = N;
        for (int t = 0; t < Mdec; ++t) {
            std::vector<int64_t> tok{cur};
            mt::Tensor idx = int_tensor_2d_long(tok, 1, 1);
            m::StepOut so = coder.forward(idx, caches, offset);
            bool ok = (so.logits.shape.rank == 3 && so.logits.shape.d[1] == 1);
            check("check6_decode_logits", ok);
             mt::Tensor logits1d;
            {
                mt::Shape s; s.rank = 2; s.d[0] = 1; s.d[1] = V;
                logits1d = mt::make(s, mt::DType::FP32, 0.0f);
            }
            std::memcpy(logits1d.ptr<float>(), so.logits.ptr<float>(),
                        static_cast<size_t>(V) * sizeof(float));
            // prev window: most recent LAST, keep last 64
            std::vector<int64_t> window;
            int nwin = std::min(static_cast<int>(prev.size()), 64);
            window.assign(prev.end() - nwin, prev.end());
            mt::Tensor pw = int_tensor_2d_long(window, 1, static_cast<int>(window.size()));
            m::DecodeWeights dw;
            int nxt = m::pick_next(logits1d, pw, dw);
            check("check6_decode_inrange", nxt >= 0 && nxt < V);
            if (nxt >= 0 && nxt < V) {
                gen.push_back(nxt);
                prev.push_back(nxt);
                cur = nxt;
            }
            const mt::Tensor& sm = pool.summary();
            if (sm.shape.rank >= 1 && sm.shape.d[0] == cfg.d_model) {
                std::vector<float> sv(sm.shape.d[0]);
                std::memcpy(sv.data(), sm.ptr<float>(), sv.size() * sizeof(float));
                summaries.push_back(std::move(sv));
            }
            ++offset;
        }
        check("check6_n_gen", static_cast<int>(gen.size()) == Mdec);

        // compare against golden ids.txt
        {
            std::ifstream f((gdir / "ids.txt").string());
            check("check6_ids_exist", f.good());
            if (!f) return failures ? 1 : 0;
            int Nhdr, Mhdr;
            f >> Nhdr >> Mhdr;
            std::vector<int> gids;
            int x;
            while (f >> x) gids.push_back(x);
            bool ok_hdr = (Nhdr == N && Mhdr == Mdec);
            check("check6_ids_hdr", ok_hdr);
            bool ok_phase = (static_cast<int>(gids.size()) >= N &&
                             std::equal(corpus.begin(), corpus.end(), gids.begin()));
            check("check6_phaseA_ids", ok_phase);
            bool ok_gen = (static_cast<int>(gids.size()) >= N + Mdec &&
                           std::equal(gen.begin(), gen.end(), gids.begin() + N));
            check("check6_gen_ids", ok_gen);
        }

        // compare halt rows against golden logits.bin (halt_row per position)
        {
            auto bin = read_file((gdir / "logits.bin").string());
            check("check6_logits_bin", bin.size() >= 8);
            if (bin.size() < 8) return failures ? 1 : 0;
            uint32_t Ng = read_u32(bin.data());
            uint32_t Vg = read_u32(bin.data() + 4);
            check("check6_logits_hdr", Ng == static_cast<uint32_t>(N) && Vg == static_cast<uint32_t>(V));
            const uint8_t* p = bin.data() + 8;
            bool ok_hr = true;
            bool ok_lg = true;
            size_t per = 4 + static_cast<size_t>(V) * 4;
            for (int i = 0; i < N && static_cast<size_t>(8 + (i + 1) * per) <= bin.size(); ++i) {
                uint32_t hr = read_u32(p);
                if (hr != static_cast<uint32_t>(lrows[i]) && ok_hr) {
                    std::cout << "  halt_row mismatch at " << i << " gold=" << hr
                              << " got=" << lrows[i] << "\n";
                    ok_hr = false;
                }
                const float* gp = reinterpret_cast<const float*>(p + 4);
                const float* cp = &llogits[i * V];
                for (int v = 0; v < V; ++v) {
                    if (std::fabs(gp[v] - cp[v]) > 1e-4f) { ok_lg = false; break; }
                }
                p += per;
            }
            check("check6_halt_rows", ok_hr);
            check("check6_logits_match", ok_lg);
        }

        // compare summaries.bin (N+1+M = 36 summaries, each d_model=24 floats)
        {
            auto bin = read_file((gdir / "summaries.bin").string());
            check("check6_sum_bin", bin.size() >= 4);
            if (bin.size() < 4) return failures ? 1 : 0;
            uint32_t count = read_u32(bin.data());
            int expect_sum = 3 + 1 + Mdec;  // 36
            check("check6_sum_count", count == static_cast<uint32_t>(expect_sum));
            check("check6_sum_ours", static_cast<int>(summaries.size()) == expect_sum);
            if (count == static_cast<uint32_t>(summaries.size()) && count > 0) {
                size_t per = static_cast<size_t>(cfg.d_model) * 4;
                bool ok = true;
                const uint8_t* p = bin.data() + 4;
                for (uint32_t i = 0; i < count && ok; ++i) {
                    const float* gp = reinterpret_cast<const float*>(p);
                    const std::vector<float>& sv = summaries[i];
                    for (int j = 0; j < cfg.d_model; ++j) {
                        if (std::fabs(gp[j] - sv[j]) > 2e-4f) { ok = false; break; }
                    }
                    p += per;
                }
                check("check6_summaries_match", ok);
            } else {
                check("check6_summaries_match", false);
            }
        }

        // trace halt rows: every phase-A token halts at step 5
        bool all5 = std::all_of(lrows.begin(), lrows.end(),
                               [](float f) { return static_cast<int>(f) == 5; });
        check("check6_trace_all5", all5);
    }

    // ---- check7: save_paged round-trip ----
    {
        m::Config cfg;
        ::store::Manifest man;
        std::map<std::string, mininpz::Array> state;
        bool ok = ::store::load(wdir.string(), man, state);
        check("check7_load", ok);
        if (!ok) return failures ? 1 : 0;
        cfg.from_manifest(man.cfg_obj);
        fs::path tmp = fs::path(root) / "build" / "wavef_paged_save";
        fs::remove_all(tmp);
        std::vector<std::pair<std::string, mt::Tensor>> sd;
        for (auto& kv : state) sd.emplace_back(kv.first, pg::array_to_tensor(kv.second));
        // keep only core + router (no expert tensors) for the round-trip surface
        std::vector<std::pair<std::string, mt::Tensor>> surf;
        for (auto& kv : sd) {
            const std::string& k = kv.first;
            if (k.find("pool.experts.") == std::string::npos) surf.push_back(kv);
        }
        mt::Shape gsha; gsha.rank = 1; gsha.d[0] = cfg.pool_experts;
        mt::Tensor ones = mt::make(gsha, mt::DType::FP32, 1.0f);
        pg::PagedPool pool(cfg.pool_experts, cfg.d_model, cfg.pool_d_ff,
                           cfg.pool_resident, man.pool_ram, cfg.explore,
                           cfg.margin, cfg.dwell, man.read_only, ones);
        ::store::SaveResult sr;
        bool sv = ::store::save_paged(surf, pool, man.cfg_obj, tmp.string(), 7, 0.5, &sr);
        check("check7_save_paged", sv);
        // round-trip load
        ::store::Manifest m2;
        std::map<std::string, mininpz::Array> st2;
        bool l2 = ::store::load(tmp.string(), m2, st2);
        check("check7_reload", l2);
        if (l2) {
            check("check7_paged", m2.paged);
            check("check7_read_only", m2.read_only);
            check("check7_pool_ram", m2.pool_ram == 4);
            check("check7_telemetry", m2.telemetry.t == mini::JsonValue::Type::Obj);
            check("check7_cfg_obj", m2.cfg_obj.t == mini::JsonValue::Type::Obj);
            // expert files still present (copied) -> experts dir non-empty
            fs::path edir = tmp / "experts";
            bool has_experts = fs::exists(edir) &&
                std::filesystem::directory_iterator(edir) != std::filesystem::directory_iterator{};
            // we did NOT write experts; save_paged enumerates existing ones only.
            // (tmp has no experts dir populated) -> expect empty expert list.
            check("check7_no_expert_files", static_cast<int>(m2.experts.size()) == 0);
        }
        fs::remove_all(tmp);
    }

    // ---- check8: Tiers read/write + flush (RW, non-read_only) ----
    {
        fs::path tmp = fs::path(root) / "build" / "wavef_tiers_rw";
        fs::remove_all(tmp);
        fs::create_directories(tmp);
        pg::Tiers tiers(tmp.string(), 24, 32, 4, false /*read_only*/);
        mt::Shape s2; s2.rank = 2; s2.d[0] = 32; s2.d[1] = 24;
        mt::Tensor w1 = mt::make(s2, mt::DType::FP32, 0.5f);
        mt::Shape s3; s3.rank = 2; s3.d[0] = 24; s3.d[1] = 32;
        mt::Tensor w3 = mt::make(s3, mt::DType::FP32, 0.25f);
        mt::Shape s4; s4.rank = 2; s4.d[0] = 32; s4.d[1] = 32;
        mt::Tensor w2 = mt::make(s4, mt::DType::FP32, 0.1f);
        tiers.put(42, w1, w3, w2, true /*dirty*/);
        tiers.flush();
        // file exists
        std::string fp = (tmp / "e00042.npz").string();
        check("check8_file_written", fs::exists(fp));
        // reload
        mt::Tensor r1, r3, r2;
        bool got = tiers.fetch(42, r1, r3, r2);
        check("check8_refetch", got);
        if (got) {
            check("check8_w1", r1.data_ == w1.data_);
            check("check8_w3", r3.data_ == w3.data_);
            check("check8_w2", r2.data_ == w2.data_);
        }
        const auto& tc = tiers.counters();
        check("check8_wb", tc.writebacks >= 1);
        fs::remove_all(tmp);
    }

    // ---- check9: LRU micro (evict least-recently-used) ----
    {
        fs::path tmp = fs::path(root) / "build" / "wavef_lru";
        fs::remove_all(tmp);
        fs::create_directories(tmp);
        pg::Tiers tiers(tmp.string(), 4, 4, 2 /*ram_capacity*/, false);
        auto mk = [](float v) {
            mt::Shape s; s.rank = 2; s.d[0] = 4; s.d[1] = 4;
            return mt::make(s, mt::DType::FP32, v);
        };
        tiers.put(0, mk(0.0f), mk(0.0f), mk(0.0f), false);
        tiers.put(1, mk(1.0f), mk(1.0f), mk(1.0f), false);
        // access 0 (make it MRU)
        mt::Tensor a, b, c;
        tiers.fetch(0, a, b, c);
        // insert 2 -> should evict 1 (LRU), keep 0
        tiers.put(2, mk(2.0f), mk(2.0f), mk(2.0f), false);
        mt::Tensor d, e, f;
        bool got2 = tiers.fetch(2, d, e, f);
        bool got0 = tiers.fetch(0, d, e, f);
        bool gone1 = !tiers.fetch(1, d, e, f);
        check("check9_put2", got2);
        check("check9_keep0", got0);
        check("check9_evict1", gone1);
        fs::remove_all(tmp);
    }

    // ---- check10: Tier-2 self-consistency (choose_by_demand) ----
    {
        ::store::Manifest man;
        std::map<std::string, mininpz::Array> state;
        bool ok = ::store::load(wdir.string(), man, state);
        check("check10_load", ok);
        if (!ok) return failures ? 1 : 0;
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
        {
            m::Coder coder(cfg);
            check("check10_load_weights", coder.load_weights(sd, pool));
            coder.set_pool(&pool);
            coder.begin_segment();  // swap_to -> warm residents [5,4,3]
        }
        // build keys for self-consistency (reads expert files).
        int nkey = pool.build_keys(false);
        check("check10_keys_built", nkey == cfg.pool_experts);
        // demand = gate.abs(); want = top-k by demand priority (expert ids).
        const float* gp = pool.gate().ptr<float>();
        std::vector<std::pair<double, int>> gs;
        for (int i = 0; i < cfg.pool_experts; ++i)
            gs.emplace_back(std::fabs(static_cast<double>(gp[i])), i);
        std::sort(gs.begin(), gs.end(),
                  [](const std::pair<double, int>& a, const std::pair<double, int>& b) {
                      if (a.first != b.first) return a.first > b.first;
                      return a.second < b.second;
                  });
        std::vector<double> want(cfg.pool_resident, 0.0);
        for (int i = 0; i < cfg.pool_resident; ++i) want[i] = static_cast<double>(gs[i].second);
        mt::Tensor gates_abs = mt::make(gsha, mt::DType::FP32, 0.0f);
        float* ap = gates_abs.ptr<float>();
        for (int i = 0; i < cfg.pool_experts; ++i) ap[i] = std::fabs(gp[i]);
        mt::Tensor everb = mt::make(gsha, mt::DType::FP32, 1.0f);
        mt::Tensor since = mt::make(gsha, mt::DType::FP32, 0.0f);
        mt::Tensor last_seen = mt::make(gsha, mt::DType::FP32, 0.0f);
        std::vector<int> out;
        int rc = pool.choose_by_demand(gates_abs, everb, since, last_seen, want, out);
        check("check10_choose_demand_rc", rc == 0);
        check("check10_choose_demand_size", static_cast<int>(out.size()) == cfg.pool_resident);
        // warm-card: the returned order must be the top-k by gate demand.
        std::vector<int> topk;
        for (int i = 0; i < cfg.pool_resident; ++i) topk.push_back(gs[i].second);
        std::sort(topk.begin(), topk.end());
        std::vector<int> got_sorted = out;
        std::sort(got_sorted.begin(), got_sorted.end());
        check("check10_choose_demand_set", topk == got_sorted);
    }

    // ---- check11: failure paths ----
    {
        m::Config cfg;
        ::store::Manifest man;
        // non-existent dir
        std::map<std::string, mininpz::Array> state;
        bool bad = ::store::load((fs::path(root) / "nonexistent_dir_xyz").string(), man, state);
        check("check11_bad_load", !bad);
        // empty idx tensor -> empty StepOut
        ::store::Manifest man2;
        std::map<std::string, mininpz::Array> state2;
        bool ok = ::store::load(wdir.string(), man2, state2);
        check("check11_ok", ok);
        cfg.from_manifest(man2.cfg_obj);
        mt::Shape gsha; gsha.rank = 1; gsha.d[0] = cfg.pool_experts;
        mt::Tensor ones = mt::make(gsha, mt::DType::FP32, 1.0f);
        pg::PagedPool pool(cfg.pool_experts, cfg.d_model, cfg.pool_d_ff,
                           cfg.pool_resident, man2.pool_ram, cfg.explore,
                           cfg.margin, cfg.dwell, man2.read_only, ones);
        m::Coder coder(cfg);
        std::vector<std::pair<std::string, mt::Tensor>> sd;
        for (auto& kv : state2) sd.emplace_back(kv.first, pg::array_to_tensor(kv.second));
        bool lw = coder.load_weights(sd, pool);
        check("check11_load_weights", lw);
        // empty idx [1,0] -> rank-0 StepOut
        mt::Shape es; es.rank = 2; es.d[0] = 1; es.d[1] = 0;
        mt::Tensor empty = mt::make(es, mt::DType::FP32, 0.0f);
        auto caches = coder.empty_caches();
        m::StepOut so = coder.forward(empty, caches, 0);
        check("check11_empty_forward", so.logits.shape.rank == 0);
    }

    if (failures == 0) {
        std::cout << "ALL_TESTS_PASSED\n";
        return 0;
    }
    std::cout << failures << " check(s) FAILED\n";
    return 1;
}
