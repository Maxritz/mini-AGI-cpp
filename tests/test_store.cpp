// tests/test_store.cpp
// Store module tests: classification, bf16 packing, save/load round-trip,
// stale-file cleanup, best_val/summarise, atomicity.
#include "store.hpp"
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static int failures = 0;

static void check(const char* name, bool ok) {
    if (ok) {
        std::cout << "ok " << name << "\n";
    } else {
        std::cout << "FAIL " << name << "\n";
        ++failures;
    }
}

// ---- array builders ----
static mininpz::Array arr_f4(std::vector<size_t> dims, float fill) {
    mininpz::Shape sh;
    sh.rank = dims.size();
    for (size_t i = 0; i < dims.size(); ++i) sh.d[i] = dims[i];
    mininpz::Array a;
    a.dtype = mininpz::DType::F4;
    a.shape = sh;
    size_t n = sh.numel();
    a.bytes.resize(n * 4);
    for (size_t i = 0; i < n; ++i) {
        float v = fill;
        std::memcpy(a.bytes.data() + i * 4, &v, 4);
    }
    return a;
}

static mininpz::Array arr_i2(const std::vector<int16_t>& data) {
    mininpz::Shape sh;
    sh.rank = 1;
    sh.d[0] = data.size();
    mininpz::Array a;
    a.dtype = mininpz::DType::I2;
    a.shape = sh;
    a.bytes.resize(data.size() * 2);
    std::memcpy(a.bytes.data(), data.data(), data.size() * 2);
    return a;
}

static mininpz::Array arr_f8_scalar(double v) {
    mininpz::Shape sh;  // rank 0
    mininpz::Array a;
    a.dtype = mininpz::DType::F8;
    a.shape = sh;
    a.bytes.resize(8);
    std::memcpy(a.bytes.data(), &v, 8);
    return a;
}

static bool bytes_equal(const mininpz::Array& a, const mininpz::Array& b) {
    return a.dtype == b.dtype && a.shape == b.shape && a.bytes == b.bytes;
}

static bool has_tmp_file(const fs::path& dir) {
    for (auto& p : fs::recursive_directory_iterator(dir)) {
        if (!p.is_regular_file()) continue;
        std::string n = p.path().filename().string();
        if (n.find(".tmp") != std::string::npos) return true;
    }
    return false;
}

static bool path_exists(const fs::path& p) { return fs::exists(p); }

static std::map<std::string, mininpz::Array> make_state() {
    std::map<std::string, mininpz::Array> state;
    state["core.tok_emb.weight"] = arr_f4({265, 512}, 0.5f);
    state["core.blocks.0.ln1.weight"] = arr_f4({512}, 0.1f);
    state["lm_head.weight"] = arr_f4({512, 265}, 0.2f);
    state["pool.segment_router.weight"] = arr_f4({2, 4}, 0.3f);
    state["pool.gate"] = arr_f4({4}, 0.4f);
    state["pool.depth_emb"] = arr_f4({8, 4}, 0.25f);
    // expert 2: three weights + two Adam moment arrays
    state["pool.experts.2.w1.weight"] = arr_f4({1408, 512}, 0.25f);
    state["pool.experts.2.w3.weight"] = arr_f4({1408, 512}, 0.26f);
    state["pool.experts.2.w2.weight"] = arr_f4({512, 1408}, 0.27f);
    const std::vector<int16_t> mom = {16256, -16384, 16457, 14979, 18304, 11996, -16576};
    state["pool.experts.2.w1_m"] = arr_i2(mom);
    state["pool.experts.2.w1_v"] = arr_i2(mom);
    // expert 7: a deeper block leaf (b0_w1)
    state["pool.experts.7.blocks.0.w1.weight"] = arr_f4({1408, 512}, 0.75f);
    // trunk optimizer bundles
    state["lm_head|m"] = arr_i2({16256, 16457});
    state["lm_head|v"] = arr_i2({16256, 16457});
    state["lm_head|t"] = arr_f8_scalar(3.0);
    return state;
}

int main() {
    // ---- classification ----
    check("is_expert bean", store::is_expert("pool.experts.12.w1.weight"));
    check("is_expert no pool", !store::is_expert("experts.5.w1.weight"));
    check("is_expert gate", !store::is_expert("pool.gate"));
    check("is_router seg", store::is_router("pool.segment_router.weight"));
    check("is_router depth", store::is_router("pool.depth_emb"));
    check("is_router neg", !store::is_router("core.tok_emb.weight"));
    check("index 12", store::expert_index("pool.experts.12.w1.weight") == 12);
    check("index 3", store::expert_index("pool.experts.3.blocks.1.w3.weight") == 3);
    check("leaf w1", store::expert_leaf("pool.experts.12.w1.weight") == "w1");
    check("leaf b0_w1", store::expert_leaf("pool.experts.12.blocks.0.w1.weight") == "b0_w1");
    check("leaf b1_w3", store::expert_leaf("pool.experts.3.blocks.1.w3.weight") == "b1_w3");
    check("leaf moment", store::expert_leaf("pool.experts.2.w1_m") == "w1_m");
    check("is_moment m", store::is_moment("w1_m"));
    check("is_moment v", store::is_moment("w1_v"));
    check("is_moment |m", store::is_moment("lm_head|m"));
    check("is_moment |t", !store::is_moment("lm_head|t"));
    check("is_moment weight", !store::is_moment("w1"));
    check("is_optim |m", store::is_optim("lm_head|m"));
    check("is_optim |t", store::is_optim("lm_head|t"));
    check("is_optim neg", !store::is_optim("w1"));

    // ---- bf16 packing ----
    const std::vector<float> fs = {1.0f, -2.0f, 3.140625f, 1e-3f, 65504.0f, 1e-10f, -0.75f};
    const std::vector<int16_t> expect = {16256, -16384, 16457, 14979, 18304, 11996, -16576};
    std::vector<int16_t> packed = store::pack_bf16(fs);
    check("pack_bf16 exact bits", packed == expect);
    {
        const std::vector<float> roundtrip = store::unpack_bf16(expect);
        const std::vector<float> target = {
            1.0f, -2.0f, 3.140625f, 0.00099945068359375f,
            65536.0f, 1.000444171950221e-10f, -0.75f};
        bool ok = roundtrip.size() == target.size();
        for (size_t i = 0; ok && i < target.size(); ++i) {
            ok = std::fabs(roundtrip[i] - target[i]) <= 1e-6f;
        }
        check("unpack_bf16 approx", ok);
    }

    // ---- save ----
    fs::path tmp = fs::temp_directory_path() / fs::path("minagi_store_test");
    std::error_code ec;
    fs::remove_all(tmp, ec);
    fs::create_directories(tmp);

    std::map<std::string, mininpz::Array> state = make_state();
    std::map<std::string, std::string> cfg;
    cfg["d_model"] = "512";
    cfg["d_ff"] = "1408";
    std::map<int, double> gates = {{2, 1.0}, {7, 0.9}};
    store::SaveResult out;
    bool ok = store::save(tmp.string(), state, 42, 0.5, cfg, gates, out);
    check("save returns true", ok);
    check("save n_experts", ok && out.n_experts == 2);
    check("save total_bytes>0", ok && out.total_bytes > 0);
    check("core.npz exists", path_exists(tmp / "core.npz"));
    check("routers.npz exists", path_exists(tmp / "routers.npz"));
    check("optim.npz exists", path_exists(tmp / "optim.npz"));
    check("manifest.json exists", path_exists(tmp / "manifest.json"));
    check("e00002 exists", path_exists(tmp / "experts" / "e00002.npz"));
    check("e00007 exists", path_exists(tmp / "experts" / "e00007.npz"));
    check("no tmp files after save", !has_tmp_file(tmp));

    // ---- manifest fields ----
    {
        std::ifstream ifs(tmp / "manifest.json", std::ios::binary);
        std::string text((std::istreambuf_iterator<char>(ifs)),
                         std::istreambuf_iterator<char>());
        mini::JsonValue root;
        check("manifest parses", mini::json_parse(text, root));
        double d = 0;
        if (mini::json_parse(text, root)) {
            const mini::JsonValue* v = mini::json_find(root, "n_experts");
            check("manifest n_experts", v && mini::json_try_num(*v, d) && (int)d == 2);
            v = mini::json_find(root, "d_model");
            check("manifest d_model", v && mini::json_try_num(*v, d) && (int)d == 512);
            v = mini::json_find(root, "d_ff");
            check("manifest d_ff", v && mini::json_try_num(*v, d) && (int)d == 1408);
            v = mini::json_find(root, "experts");
            if (v && v->t == mini::JsonValue::Type::Arr) {
                check("manifest experts size", v->a.size() == 2);
                if (v->a.size() == 2) {
                    const mini::JsonValue* f0 = mini::json_find(v->a[0], "id");
                    const mini::JsonValue* f1 = mini::json_find(v->a[1], "id");
                    bool first2 = f0 && mini::json_try_num(*f0, d) && (int)d == 2;
                    bool second7 = f1 && mini::json_try_num(*f1, d) && (int)d == 7;
                    check("manifest experts sorted ids", first2 && second7);
                    const mini::JsonValue* g7 = mini::json_find(v->a[1], "gate");
                    check("manifest gate",
                          g7 && mini::json_try_num(*g7, d) && std::fabs(d - 0.9) < 1e-9);
                    const mini::JsonValue* p0 = mini::json_find(v->a[0], "params");
                    bool params_ok = p0 && mini::json_try_num(*p0, d) &&
                                     (long long)d == 1408LL * 512 * 3 &&
                                     d == 2162688.0;
                    check("manifest params excludes moments", params_ok);
                }
            }
        }
    }

    // ---- best_val / load ----
    check("best_val", ok && std::fabs(store::best_val(tmp.string()) - 0.5) < 1e-12);
    check("best_val empty dir is inf",
          std::isinf(store::best_val((fs::temp_directory_path() / "no_such_dir_xyz").string())));

    store::Manifest man;
    std::map<std::string, mininpz::Array> state2;
    ok = store::load(tmp.string(), man, state2);
    check("load returns true", ok);
    if (ok) {
        check("load step", man.step == 42);
        check("load val", man.val == 0.5);
        check("load n_experts", man.n_experts == 2);
        check("load d_model", man.d_model == 512);
        check("load d_ff", man.d_ff == 1408);
        check("load removed 0", man.removed_expert_files == 0);
        check("load cfg", man.cfg.count("d_ff") && man.cfg.at("d_ff") == "1408");
    }
    check("load core", state2.count("core.tok_emb.weight") == 1);
    check("load router", state2.count("pool.segment_router.weight") == 1 &&
                             state2.count("pool.depth_emb") == 1);
    check("load expert w1", state2.count("pool.experts.2.w1.weight") == 1);
    check("load expert w2", state2.count("pool.experts.2.w2.weight") == 1);
    check("load blocks leaf", state2.count("pool.experts.7.blocks.0.w1.weight") == 1);
    check("load skips expert moments",
          state2.count("pool.experts.2.w1_m") == 0 &&
              state2.count("pool.experts.2.w1_v") == 0);
    check("load skips trunk moments",
          state2.count("lm_head|m") == 0 && state2.count("lm_head|t") == 0);
    check("load values byte-equal",
          bytes_equal(state2["core.tok_emb.weight"], state["core.tok_emb.weight"]) &&
              bytes_equal(state2["pool.experts.7.blocks.0.w1.weight"],
                          state["pool.experts.7.blocks.0.w1.weight"]));

    // direct expert file inspection
    {
        std::vector<mininpz::NpzEntry> entries;
        ok = mininpz::read_npz((tmp / "experts" / "e00007.npz").string(), entries);
        check("e00007 single leaf",
              ok && entries.size() == 1 && entries[0].name == "b0_w1.npy");
    }
    {
        std::vector<mininpz::NpzEntry> entries;
        ok = mininpz::read_npz((tmp / "experts" / "e00002.npz").string(), entries);
        bool moment_names_ok = ok && entries.size() == 5;
        for (const auto& e : entries) {
            if (e.name != "w1.npy" && e.name != "w3.npy" && e.name != "w2.npy" &&
                e.name != "w1_m.npy" && e.name != "w1_v.npy") {
                moment_names_ok = false;
            }
        }
        check("e00002 moment leaves", moment_names_ok);
    }
    {
        std::vector<mininpz::NpzEntry> entries;
        ok = mininpz::read_npz((tmp / "optim.npz").string(), entries);
        bool names_ok = ok && entries.size() == 3;
        for (const auto& e : entries) {
            if (e.name != "lm_head|m.npy" && e.name != "lm_head|v.npy" &&
                e.name != "lm_head|t.npy") {
                names_ok = false;
            }
        }
        check("optim npz names", names_ok);
    }

    check("summarise", store::summarise(tmp.string()).find("step=42") != std::string::npos &&
                           store::summarise(tmp.string()).find("val=0.5") != std::string::npos);

    // ---- stale expert-file cleanup ----
    fs::path tmp2 = fs::temp_directory_path() / fs::path("minagi_store_test2");
    fs::remove_all(tmp2, ec);
    fs::create_directories(tmp2 / "experts");
    std::vector<mininpz::NpzEntry> junk = {{"junk", arr_f4({2, 2}, 1.0f)}};
    ok = mininpz::write_npz((tmp2 / "experts" / "e00099.npz").string(), junk) &&
         mininpz::write_npz((tmp2 / "experts" / "e00012.npz").string(), junk);
    bool saved2 = store::save(tmp2.string(), state, 42, 0.5, cfg, gates, out);
    check("stale save", ok && saved2);
    check("stale removed", saved2 && !path_exists(tmp2 / "experts" / "e00099.npz") &&
                               !path_exists(tmp2 / "experts" / "e00012.npz"));
    {
        std::ifstream ifs(tmp2 / "manifest.json", std::ios::binary);
        std::string text((std::istreambuf_iterator<char>(ifs)),
                         std::istreambuf_iterator<char>());
        mini::JsonValue root;
        if (mini::json_parse(text, root)) {
            const mini::JsonValue* v = mini::json_find(root, "removed_expert_files");
            double d = 0;
            check("manifest removed=2",
                  v && mini::json_try_num(*v, d) && (int)d == 2);
        } else {
            check("manifest removed=2", false);
        }
    }

    // ---- re-save atomicity ----
    state["extra.bias"] = arr_f4({4}, 0.05f);
    ok = store::save(tmp.string(), state, 43, 0.6, cfg, gates, out);
    check("resave ok", ok && out.n_experts == 2);
    check("resave no tmp", !has_tmp_file(tmp));
    store::Manifest man2;
    std::map<std::string, mininpz::Array> state3;
    ok = store::load(tmp.string(), man2, state3);
    check("resave load", ok && man2.step == 43 && man2.val == 0.6);
    check("resave has extra", ok && state3.count("extra.bias") == 1);

    std::cout << (failures == 0 ? "ALL_TESTS_PASSED" : "SOME_TESTS_FAILED") << "\n";
    return failures == 0 ? 0 : 1;
}