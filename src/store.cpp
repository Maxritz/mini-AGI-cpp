// src/store.cpp
// Weights-directory store implementation. Port of minagi/store.py (non-paged
// path). A model's full state is a flat map param-name -> Array; saving groups
// it into core.npz / routers.npz / optim.npz / experts/e%05d.npz and writes a
// manifest.json (atomically, via manifest.json.tmp + rename). Loading rebuilds
// the flat map exactly as Python's store.load does: expert _m/_v moment arrays
// are optimizer state and are NOT part of the state dict.
#include "store.hpp"
#include "paged.hpp"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <set>
#include <stdexcept>

namespace fs = std::filesystem;

namespace store {

namespace {

bool starts_with(const std::string& s, const std::string& p) {
    return s.size() >= p.size() && s.compare(0, p.size(), p) == 0;
}

bool ends_with(const std::string& s, const std::string& p) {
    return s.size() >= p.size() && s.compare(s.size() - p.size(), p.size(), p) == 0;
}

std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> out;
    size_t start = 0;
    for (size_t i = 0; i <= s.size(); ++i) {
        if (i == s.size() || s[i] == sep) {
            out.push_back(s.substr(start, i - start));
            start = i + 1;
        }
    }
    return out;
}

}  // namespace

bool is_expert(const std::string& key) {
    return starts_with(key, "pool.") && key.find(".experts.") != std::string::npos;
}

bool is_router(const std::string& key) {
    return ends_with(key, "router.weight") || ends_with(key, "depth_emb");
}

bool is_moment(const std::string& name) {
    return ends_with(name, "_m") || ends_with(name, "_v") ||
           ends_with(name, "|m") || ends_with(name, "|v");
}

bool is_optim(const std::string& name) {
    // Adam state bundles, "<param>|m" / "|v" / "|t". The "|t" step counter is
    // optimizer state too even though is_moment() treats it as not-a-moment.
    return ends_with(name, "|m") || ends_with(name, "|v") || ends_with(name, "|t");
}

int expert_index(const std::string& key) {
    // "pool.experts.12.w1.weight" -> 12
    auto pos = key.find(".experts.");
    if (pos == std::string::npos) return -1;
    pos += std::string(".experts.").size();
    int idx = 0;
    while (pos < key.size() && std::isdigit(static_cast<unsigned char>(key[pos]))) {
        idx = idx * 10 + (key[pos] - '0');
        ++pos;
    }
    return idx;
}

std::string expert_leaf(const std::string& key) {
    // Mirrors Python expert_leaf:
    //   pool.experts.12.w1.weight           -> w1
    //   pool.experts.12.blocks.0.w1.weight  -> b0_w1
    // Split on dots and apply Python's mapping to the fragment past the id.
    // Python drops the last element of the fragment (the ".weight") and
    // returns tail[0], or b{tail[1]}_{tail[2]} when tail[0] == "blocks".
    // Evaluating the FULL fragment (without dropping) gives identical results
    // for every weight key while also handling moment keys ("w1_m" -> "w1_m",
    // "pool.experts.2.blocks.0.w1_m" -> "b0_w1_m"), which Python never does.
    std::vector<std::string> parts = split(key, '.');
    auto it = std::find(parts.begin(), parts.end(), "experts");
    if (it == parts.end()) return key;
    auto i = it - parts.begin() + 2;  // past the id
    if (i <= 0 || i >= static_cast<long long>(parts.size())) return key;
    const std::string& a0 = parts[static_cast<size_t>(i)];
    if (a0 == "blocks" && i + 3 <= static_cast<long long>(parts.size())) {
        return "b" + parts[static_cast<size_t>(i) + 1] + "_" +
               parts[static_cast<size_t>(i) + 2];
    }
    return a0;
}

std::string expert_dir(const std::string& dir) {
    return (fs::path(dir) / "experts").string();
}

static size_t elt_size(mininpz::DType dt) {
    switch (dt) {
        case mininpz::DType::U8: return 1;
        case mininpz::DType::I2: return 2;
        case mininpz::DType::I4: return 4;
        case mininpz::DType::I8: return 8;
        case mininpz::DType::F4: return 4;
        case mininpz::DType::F8: return 8;
    }
    return 0;
}

static size_t array_nbytes(const mininpz::Array& a) {
    return a.shape.numel() * elt_size(a.dtype);
}

// ---- bf16 ----

// fp32 -> bfloat16 bits with round-to-nearest-even on the low 16 mantissa
// bits. The classic add-0x7FFF trick matches torch's fp32->bf16 conversion.
static uint16_t float_to_bf16_bits(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(float));
    uint32_t rounded = (bits + ((1u << 15) - 1u + ((bits >> 16) & 1u))) & 0xffff0000u;
    return static_cast<uint16_t>(rounded >> 16);
}

static float bf16_bits_to_float(uint16_t bf) {
    uint32_t bits = static_cast<uint32_t>(bf) << 16;
    float f;
    std::memcpy(&f, &bits, sizeof(float));
    return f;
}

std::vector<int16_t> pack_bf16(const std::vector<float>& vals) {
    std::vector<int16_t> out(vals.size());
    for (size_t i = 0; i < vals.size(); ++i) {
        out[i] = static_cast<int16_t>(float_to_bf16_bits(vals[i]));
    }
    return out;
}

std::vector<float> unpack_bf16(const std::vector<int16_t>& bits) {
    std::vector<float> out(bits.size());
    for (size_t i = 0; i < bits.size(); ++i) {
        out[i] = bf16_bits_to_float(static_cast<uint16_t>(bits[i]));
    }
    return out;
}

// ---- Helpers ----

static std::string fmt_expert_file(int id) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "experts/e%05d.npz", id);
    return std::string(buf);
}

// numpy's savez stores each member as "<name>.npy"; accept that suffix when
// reading, and return bare names untouched when reading our own (or old) files.
static std::string strip_npy(const std::string& name) {
    if (ends_with(name, ".npy") && name.size() > 4) {
        return name.substr(0, name.size() - 4);
    }
    return name;
}

// ---- Save ----

bool save(const std::string& dir,
          const std::map<std::string, mininpz::Array>& state,
          int step,
          double val,
          const std::map<std::string, std::string>& cfg,
          const std::map<int, double>& gates,
          SaveResult& out) {
    out.n_experts = 0;
    out.total_bytes = 0;

    try {
        if (!fs::exists(dir)) fs::create_directories(dir);
        fs::path experts_dir = fs::path(dir) / "experts";
        if (!fs::exists(experts_dir)) fs::create_directories(experts_dir);

        std::vector<std::string> core_keys, router_keys;
        std::map<int, std::vector<mininpz::NpzEntry>> expert_entries;
        std::vector<mininpz::NpzEntry> optim_entries;

        for (const auto& kv : state) {
            const std::string& key = kv.first;
            const mininpz::Array& arr = kv.second;

            if (is_expert(key)) {
                int eid = expert_index(key);
                std::string leaf = expert_leaf(key);
                expert_entries[eid].push_back({leaf + ".npy", arr});
            } else if (is_optim(key)) {
                // trunk Adam state: "<param>|m" / "|v" / "|t"
                optim_entries.push_back({key + ".npy", arr});
            } else if (is_router(key)) {
                router_keys.push_back(key);
            } else {
                core_keys.push_back(key);
            }
        }

        std::sort(core_keys.begin(), core_keys.end());
        std::sort(router_keys.begin(), router_keys.end());

        std::vector<mininpz::NpzEntry> core_entries, router_entries;
        // numpy's np.savez stores every member under "<name>.npy"; mirror that
        // so a C++-written directory is byte-identical to a Python one.
        for (const auto& k : core_keys) core_entries.push_back({k + ".npy", state.at(k)});
        for (const auto& k : router_keys) router_entries.push_back({k + ".npy", state.at(k)});
        std::sort(optim_entries.begin(), optim_entries.end(),
                  [](const mininpz::NpzEntry& a, const mininpz::NpzEntry& b) {
                      return a.name < b.name;
                  });

        fs::path core_path = fs::path(dir) / "core.npz";
        fs::path router_path = fs::path(dir) / "routers.npz";
        fs::path optim_path = fs::path(dir) / "optim.npz";

        if (!mininpz::write_npz(core_path.string(), core_entries)) return false;
        if (!mininpz::write_npz(router_path.string(), router_entries)) return false;
        if (!mininpz::write_npz(optim_path.string(), optim_entries)) return false;

        // Write expert files
        std::set<int> written_expert_ids;
        std::vector<Manifest::ExpertInfo> expert_infos;
        for (auto& e : expert_entries) {
            int eid = e.first;
            written_expert_ids.insert(eid);
            std::string rel = fmt_expert_file(eid);
            fs::path full = fs::path(dir) / rel;
            std::sort(e.second.begin(), e.second.end(),
                      [](const mininpz::NpzEntry& a, const mininpz::NpzEntry& b) {
                          return a.name < b.name;
                      });
            if (!mininpz::write_npz(full.string(), e.second)) return false;

            Manifest::ExpertInfo info;
            info.id = eid;
            info.file = fs::path(rel).filename().string();
            info.remotes = 0;
            info.params = 0;
            info.bytes = static_cast<long long>(fs::file_size(full));
            info.moments = false;
            auto git = gates.find(eid);
            info.gate = (git != gates.end()) ? git->second : 1.0;

            for (const auto& ent : e.second) {
                if (is_moment(strip_npy(ent.name))) {
                    info.moments = true;
                } else {
                    info.params += static_cast<int>(ent.arr.shape.numel());
                }
            }
            expert_infos.push_back(info);
        }
        std::sort(expert_infos.begin(), expert_infos.end(),
                  [](const Manifest::ExpertInfo& a, const Manifest::ExpertInfo& b) {
                      return a.id < b.id;
                  });

        // Remove stale expert files: any experts/e*.npz not written this save.
        int removed = 0;
        if (fs::exists(experts_dir)) {
            for (auto& p : fs::directory_iterator(experts_dir)) {
                if (!p.is_regular_file()) continue;
                std::string fname = p.path().filename().string();
                if (!starts_with(fname, "e") || !ends_with(fname, ".npz")) continue;
                std::string numpart = fname.substr(1, fname.size() - 5);
                if (numpart.empty() ||
                    !std::all_of(numpart.begin(), numpart.end(),
                                 [](char c) {
                                     return std::isdigit(static_cast<unsigned char>(c));
                                 })) {
                    continue;
                }
                int id = std::stoi(numpart);
                if (written_expert_ids.find(id) == written_expert_ids.end()) {
                    fs::remove(p.path());
                    ++removed;
                }
            }
        }

        long long total = 0;
        total += static_cast<long long>(fs::file_size(core_path));
        total += static_cast<long long>(fs::file_size(router_path));
        total += static_cast<long long>(fs::file_size(optim_path));
        for (const auto& ei : expert_infos) {
            total += ei.bytes;
        }

        // d_ff/d_model come from the FIRST (smallest-id) expert's w1 weight,
        // exactly like Python: sorted(by_expert)[0]["w1"].shape.
        int d_model = 0, d_ff = 0;
        if (!expert_infos.empty()) {
            int first_id = expert_infos[0].id;
            auto it = expert_entries.find(first_id);
            if (it != expert_entries.end()) {
                for (const auto& ent : it->second) {
                    if (strip_npy(ent.name) == "w1" && ent.arr.shape.rank >= 2) {
                        d_ff = static_cast<int>(ent.arr.shape.d[0]);
                        d_model = static_cast<int>(ent.arr.shape.d[1]);
                        break;
                    }
                }
            }
        }
        if (d_model == 0 && !core_entries.empty()) {
            const mininpz::Shape& sh = core_entries[0].arr.shape;
            if (sh.rank > 0) d_model = static_cast<int>(sh.d[sh.rank - 1]);
            if (sh.rank >= 2) d_ff = static_cast<int>(sh.d[0]);
        }

        // Build manifest JSON
        mini::JsonValue cfg_obj = mini::JsonValue::make_obj(
            std::map<std::string, mini::JsonValue>());
        for (const auto& c : cfg) {
            cfg_obj.o[c.first] = mini::JsonValue::make_str(c.second);
        }

        std::vector<mini::JsonValue> core_arr;
        for (const auto& k : core_keys) core_arr.push_back(mini::JsonValue::make_str(k));
        std::vector<mini::JsonValue> router_arr;
        for (const auto& k : router_keys) router_arr.push_back(mini::JsonValue::make_str(k));

        std::vector<mini::JsonValue> experts_arr;
        for (const auto& ei : expert_infos) {
            std::map<std::string, mini::JsonValue> eo;
            eo["id"] = mini::JsonValue::make_num(static_cast<double>(ei.id));
            eo["file"] = mini::JsonValue::make_str(ei.file);
            eo["remotes"] = mini::JsonValue::make_num(static_cast<double>(ei.remotes));
            eo["params"] = mini::JsonValue::make_num(static_cast<double>(ei.params));
            eo["bytes"] = mini::JsonValue::make_num(static_cast<double>(ei.bytes));
            eo["moments"] = mini::JsonValue::make_bool(ei.moments);
            eo["gate"] = mini::JsonValue::make_num(ei.gate);
            experts_arr.push_back(mini::JsonValue::make_obj(eo));
        }

        std::map<std::string, mini::JsonValue> manifest_obj;
        manifest_obj["step"] = mini::JsonValue::make_num(static_cast<double>(step));
        manifest_obj["val"] = mini::JsonValue::make_num(val);
        manifest_obj["cfg"] = cfg_obj;
        manifest_obj["n_experts"] = mini::JsonValue::make_num(
            static_cast<double>(expert_infos.size()));
        manifest_obj["d_model"] = mini::JsonValue::make_num(static_cast<double>(d_model));
        manifest_obj["d_ff"] = mini::JsonValue::make_num(static_cast<double>(d_ff));
        manifest_obj["core_tensors"] = mini::JsonValue::make_arr(core_arr);
        manifest_obj["router_tensors"] = mini::JsonValue::make_arr(router_arr);
        manifest_obj["experts"] = mini::JsonValue::make_arr(experts_arr);
        manifest_obj["total_bytes"] = mini::JsonValue::make_num(static_cast<double>(total));
        manifest_obj["removed_expert_files"] = mini::JsonValue::make_num(
            static_cast<double>(removed));

        mini::JsonValue root = mini::JsonValue::make_obj(manifest_obj);
        std::string json_text = mini::json_dump(root, 1);

        fs::path manifest_path = fs::path(dir) / "manifest.json";
        fs::path tmp_path = fs::path(dir) / "manifest.json.tmp";
        {
            std::ofstream ofs(tmp_path, std::ios::binary | std::ios::trunc);
            if (!ofs) return false;
            ofs.write(json_text.data(), static_cast<std::streamsize>(json_text.size()));
            if (!ofs) return false;
        }
        fs::rename(tmp_path, manifest_path);

        out.n_experts = static_cast<int>(expert_infos.size());
        out.total_bytes = total;
        return true;
    } catch (...) {
        return false;
    }
}

// ---- Paged-path save (section 1 _save_paged) ----
// Writes ONLY core.npz + routers.npz + manifest.json. Expert files are never
// rewritten (read from existing experts/*.npz/.npy by store::load). The
// manifest's experts array enumerates the expert files currently on disk.
bool save_paged(const std::vector<std::pair<std::string, mt::Tensor>>& sd,
                const minagi::paged::PagedPool& pool,
                const mini::JsonValue& cfg_obj_arg,
                const std::string& dir, int step, double val,
                SaveResult* out) {
    SaveResult local;
    if (!out) out = &local;
    out->n_experts = 0;
    out->total_bytes = 0;
    try {
        if (!fs::exists(dir)) fs::create_directories(dir);
        fs::path experts_dir = fs::path(dir) / "experts";
        if (!fs::exists(experts_dir)) fs::create_directories(experts_dir);

        std::vector<std::string> core_keys, router_keys;
        std::vector<mininpz::NpzEntry> core_entries, router_entries;
        long long total = 0;
        for (const auto& kv : sd) {
            const std::string& key = kv.first;
            if (minagi::paged::PagedPool::is_moment_key(key)) continue;  // skip optim state
            mininpz::Array arr = minagi::paged::tensor_to_array(kv.second);
            if (minagi::paged::PagedPool::is_router_key(key)) {
                router_keys.push_back(key);
                router_entries.push_back({key + ".npy", arr});
                total += static_cast<long long>(array_nbytes(arr));
            } else {
                core_keys.push_back(key);
                core_entries.push_back({key + ".npy", arr});
                total += static_cast<long long>(array_nbytes(arr));
            }
        }
        std::sort(core_keys.begin(), core_keys.end());
        std::sort(router_keys.begin(), router_keys.end());
        std::sort(core_entries.begin(), core_entries.end(),
                  [](const mininpz::NpzEntry& a, const mininpz::NpzEntry& b) {
                      return a.name < b.name;
                  });
        std::sort(router_entries.begin(), router_entries.end(),
                  [](const mininpz::NpzEntry& a, const mininpz::NpzEntry& b) {
                      return a.name < b.name;
                  });

        fs::path core_path = fs::path(dir) / "core.npz";
        fs::path router_path = fs::path(dir) / "routers.npz";
        if (!mininpz::write_npz(core_path.string(), core_entries)) return false;
        if (!mininpz::write_npz(router_path.string(), router_entries)) return false;
        total += static_cast<long long>(fs::file_size(core_path));
        total += static_cast<long long>(fs::file_size(router_path));

        // Enumerate existing expert files on disk.
        std::vector<Manifest::ExpertInfo> expert_infos;
        if (fs::exists(experts_dir)) {
            for (auto& p : fs::directory_iterator(experts_dir)) {
                if (!p.is_regular_file()) continue;
                std::string fname = p.path().filename().string();
                if (!ends_with(fname, ".npz")) continue;
                int eid = expert_index("pool.experts." +
                    fname.substr(1, fname.size() - 5) + ".w1.weight");
                if (eid < 0) continue;
                Manifest::ExpertInfo info;
                info.id = eid;
                info.file = fname;
                info.remotes = 0;
                info.params = 0;
                info.bytes = static_cast<long long>(fs::file_size(p.path()));
                std::vector<mininpz::NpzEntry> ents;
                if (mininpz::read_npz(p.path().string(), ents)) {
                    for (const auto& e : ents) {
                        std::string n = strip_npy(e.name);
                        if (is_moment(n)) { info.moments = true; }
                        else { info.params += static_cast<int>(e.arr.shape.numel()); }
                    }
                }
                double g = 1.0;
                pool.lookup_gate(eid, g);
                info.gate = g;
                expert_infos.push_back(info);
            }
        }
        std::sort(expert_infos.begin(), expert_infos.end(),
                  [](const Manifest::ExpertInfo& a, const Manifest::ExpertInfo& b) {
                      return a.id < b.id;
                  });

        int d_model = 0;
        for (const auto& e : core_entries) {
            if (strip_npy(e.name) == "tok_emb.weight" &&
                e.arr.shape.rank >= 2) {
                d_model = static_cast<int>(e.arr.shape.d[1]);
                break;
            }
        }

        mini::JsonValue cfg_val = (cfg_obj_arg.t == mini::JsonValue::Type::Obj)
            ? cfg_obj_arg
            : mini::JsonValue::make_obj(std::map<std::string, mini::JsonValue>());

        std::vector<mini::JsonValue> core_arr, router_arr, experts_arr;
        for (const auto& k : core_keys) core_arr.push_back(mini::JsonValue::make_str(k));
        for (const auto& k : router_keys) router_arr.push_back(mini::JsonValue::make_str(k));
        for (const auto& ei : expert_infos) {
            std::map<std::string, mini::JsonValue> eo;
            eo["id"] = mini::JsonValue::make_num(static_cast<double>(ei.id));
            eo["file"] = mini::JsonValue::make_str(ei.file);
            eo["remotes"] = mini::JsonValue::make_num(static_cast<double>(ei.remotes));
            eo["params"] = mini::JsonValue::make_num(static_cast<double>(ei.params));
            eo["bytes"] = mini::JsonValue::make_num(static_cast<double>(ei.bytes));
            eo["moments"] = mini::JsonValue::make_bool(ei.moments);
            eo["gate"] = mini::JsonValue::make_num(ei.gate);
            experts_arr.push_back(mini::JsonValue::make_obj(eo));
        }

        std::map<std::string, mini::JsonValue> manifest_obj;
        manifest_obj["step"] = mini::JsonValue::make_num(static_cast<double>(step));
        manifest_obj["val"] = mini::JsonValue::make_num(val);
        manifest_obj["cfg"] = cfg_val;
        manifest_obj["n_experts"] = mini::JsonValue::make_num(static_cast<double>(expert_infos.size()));
        manifest_obj["d_model"] = mini::JsonValue::make_num(static_cast<double>(d_model));
        manifest_obj["d_ff"] = mini::JsonValue::make_num(0.0);
        manifest_obj["core_tensors"] = mini::JsonValue::make_arr(core_arr);
        manifest_obj["router_tensors"] = mini::JsonValue::make_arr(router_arr);
        manifest_obj["experts"] = mini::JsonValue::make_arr(experts_arr);
        manifest_obj["total_bytes"] = mini::JsonValue::make_num(static_cast<double>(total));
        manifest_obj["removed_expert_files"] = mini::JsonValue::make_num(0.0);
        manifest_obj["paged"] = mini::JsonValue::make_bool(true);
        pool.emit_manifest_lifecycle(manifest_obj);
        manifest_obj["telemetry"] = pool.telemetry_json();

        mini::JsonValue root = mini::JsonValue::make_obj(manifest_obj);
        std::string json_text = mini::json_dump(root, 1);
        fs::path manifest_path = fs::path(dir) / "manifest.json";
        fs::path tmp_path = fs::path(dir) / "manifest.json.tmp";
        {
            std::ofstream ofs(tmp_path, std::ios::binary | std::ios::trunc);
            if (!ofs) return false;
            ofs.write(json_text.data(), static_cast<std::streamsize>(json_text.size()));
            if (!ofs) return false;
        }
        fs::rename(tmp_path, manifest_path);

        out->n_experts = static_cast<int>(expert_infos.size());
        out->total_bytes = total;
        return true;
    } catch (...) {
        return false;
    }
}

// ---- Load ----

bool load(const std::string& dir,
          Manifest& manifest,
          std::map<std::string, mininpz::Array>& state) {
    state.clear();
    try {
        fs::path manifest_path = fs::path(dir) / "manifest.json";
        if (!fs::exists(manifest_path)) return false;
        std::ifstream ifs(manifest_path, std::ios::binary);
        if (!ifs) return false;
        std::stringstream ss;
        ss << ifs.rdbuf();
        std::string text = ss.str();

        mini::JsonValue root;
        if (!mini::json_parse(text, root)) return false;

        const mini::JsonValue* v;
        if ((v = mini::json_find(root, "step"))) {
            double d;
            if (mini::json_try_num(*v, d)) manifest.step = static_cast<int>(d);
        }
        if ((v = mini::json_find(root, "val"))) {
            double d;
            if (mini::json_try_num(*v, d)) manifest.val = d;
        }
        if ((v = mini::json_find(root, "cfg"))) {
            for (const auto& kv : v->o) {
                std::string s;
                if (mini::json_try_str(kv.second, s)) manifest.cfg[kv.first] = s;
            }
        }
        if ((v = mini::json_find(root, "n_experts"))) {
            double d;
            if (mini::json_try_num(*v, d)) manifest.n_experts = static_cast<int>(d);
        }
        if ((v = mini::json_find(root, "d_model"))) {
            double d;
            if (mini::json_try_num(*v, d)) manifest.d_model = static_cast<int>(d);
        }
        if ((v = mini::json_find(root, "d_ff"))) {
            double d;
            if (mini::json_try_num(*v, d)) manifest.d_ff = static_cast<int>(d);
        }
        if ((v = mini::json_find(root, "core_tensors"))) {
            for (const auto& e : v->a) {
                std::string s;
                if (mini::json_try_str(e, s)) manifest.core_tensors.push_back(s);
            }
        }
        if ((v = mini::json_find(root, "router_tensors"))) {
            for (const auto& e : v->a) {
                std::string s;
                if (mini::json_try_str(e, s)) manifest.router_tensors.push_back(s);
            }
        }
        if ((v = mini::json_find(root, "experts"))) {
            for (const auto& e : v->a) {
                Manifest::ExpertInfo ei;
                const mini::JsonValue* f;
                if ((f = mini::json_find(e, "id"))) {
                    double d;
                    if (mini::json_try_num(*f, d)) ei.id = static_cast<int>(d);
                }
                if ((f = mini::json_find(e, "file"))) {
                    std::string s;
                    if (mini::json_try_str(*f, s)) ei.file = s;
                }
                if ((f = mini::json_find(e, "remotes"))) {
                    double d;
                    if (mini::json_try_num(*f, d)) ei.remotes = static_cast<int>(d);
                }
                if ((f = mini::json_find(e, "params"))) {
                    double d;
                    if (mini::json_try_num(*f, d)) ei.params = static_cast<int>(d);
                }
                if ((f = mini::json_find(e, "bytes"))) {
                    double d;
                    if (mini::json_try_num(*f, d)) ei.bytes = static_cast<long long>(d);
                }
                if ((f = mini::json_find(e, "moments"))) ei.moments = f->b;
                if ((f = mini::json_find(e, "gate"))) {
                    double d;
                    if (mini::json_try_num(*f, d)) ei.gate = d;
                }
                manifest.experts.push_back(ei);
            }
        }
        if ((v = mini::json_find(root, "total_bytes"))) {
            double d;
            if (mini::json_try_num(*v, d)) manifest.total_bytes = static_cast<long long>(d);
        }
        if ((v = mini::json_find(root, "removed_expert_files"))) {
            double d;
            if (mini::json_try_num(*v, d)) manifest.removed_expert_files = static_cast<int>(d);
        }

        // Paged-path fields (section 1): cfg_obj (typed), telemetry block,
        // paged / read_only / pool_ram flags, pool_lifecycle arrays.
        if ((v = mini::json_find(root, "cfg_obj"))) {
            manifest.cfg_obj = *v;
        } else if ((v = mini::json_find(root, "cfg"))) {
            manifest.cfg_obj = *v;
        }
        if ((v = mini::json_find(root, "telemetry"))) {
            manifest.telemetry = *v;
        }
        if ((v = mini::json_find(root, "paged"))) {
            if (v->t == mini::JsonValue::Type::Bool) manifest.paged = v->b;
            else { double d; if (mini::json_try_num(*v, d)) manifest.paged = (d != 0.0); }
        }
        if ((v = mini::json_find(root, "read_only"))) {
            if (v->t == mini::JsonValue::Type::Bool) manifest.read_only = v->b;
            else { double d; if (mini::json_try_num(*v, d)) manifest.read_only = (d != 0.0); }
        }
        if ((v = mini::json_find(root, "pool_ram"))) {
            double d;
            if (mini::json_try_num(*v, d)) manifest.pool_ram = static_cast<int>(d);
        }
        if ((v = mini::json_find(root, "pool_ever"))) {
            for (const auto& e : v->a) {
                if (e.t == mini::JsonValue::Type::Bool) manifest.pool_ever.push_back(e.b);
                else { double d; if (mini::json_try_num(e, d)) manifest.pool_ever.push_back(d != 0.0); }
            }
        }
        if ((v = mini::json_find(root, "pool_since"))) {
            for (const auto& e : v->a) {
                double d;
                if (mini::json_try_num(e, d)) manifest.pool_since.push_back(d);
            }
        }

        fs::path core_path = fs::path(dir) / "core.npz";
        if (fs::exists(core_path)) {
            std::vector<mininpz::NpzEntry> entries;
            if (!mininpz::read_npz(core_path.string(), entries)) return false;
            for (const auto& e : entries) {
                state[strip_npy(e.name)] = e.arr;
            }
        }

        fs::path router_path = fs::path(dir) / "routers.npz";
        if (fs::exists(router_path)) {
            std::vector<mininpz::NpzEntry> entries;
            if (!mininpz::read_npz(router_path.string(), entries)) return false;
            for (const auto& e : entries) {
                state[strip_npy(e.name)] = e.arr;
            }
        }

        // optim.npz holds optimizer state ("<param>|m" etc); Python's load()
        // applies it through the optimizer, not the state dict, so it is not
        // added here.

        for (const auto& ei : manifest.experts) {
            fs::path exp_path = fs::path(dir) / "experts" / ei.file;
            if (!fs::exists(exp_path)) return false;
            // Npz bundle (standard). Legacy checkpoints may store a single flat
            // .npy file whose basename is the leaf name; fall back to read_npy.
            std::vector<mininpz::NpzEntry> entries;
            if (mininpz::read_npz(exp_path.string(), entries)) {
                for (const auto& e : entries) {
                    const std::string name = strip_npy(e.name);
                    if (is_moment(name)) continue;
                    std::string full_key;
                    if (starts_with(name, "b") && name.size() > 1 &&
                        name[1] >= '0' && name[1] <= '9') {
                        size_t underscore = name.find('_');
                        if (underscore != std::string::npos) {
                            std::string blocknum = name.substr(1, underscore - 1);
                            std::string leaf_name = name.substr(underscore + 1);
                            full_key = "pool.experts." + std::to_string(ei.id) +
                                       ".blocks." + blocknum + "." + leaf_name + ".weight";
                        } else {
                            full_key = "pool.experts." + std::to_string(ei.id) + "." + name;
                        }
                    } else {
                        full_key = "pool.experts." + std::to_string(ei.id) + "." + name +
                                   ".weight";
                    }
                    state[full_key] = e.arr;
                }
            } else {
                mininpz::Array arr;
                if (!mininpz::read_npy(exp_path.string(), arr)) return false;
                // legacy flat .npy: basename is the leaf.
                std::string leaf = exp_path.stem().string();
                const std::string name = leaf;
                if (!is_moment(name)) {
                    state["pool.experts." + std::to_string(ei.id) + "." + name + ".weight"] = arr;
                }
            }
        }

        return true;
    } catch (...) {
        return false;
    }
}

double best_val(const std::string& dir) {
    try {
        fs::path manifest_path = fs::path(dir) / "manifest.json";
        if (!fs::exists(manifest_path)) return std::numeric_limits<double>::infinity();
        std::ifstream ifs(manifest_path, std::ios::binary);
        if (!ifs) return std::numeric_limits<double>::infinity();
        std::stringstream ss;
        ss << ifs.rdbuf();
        std::string text = ss.str();
        mini::JsonValue root;
        if (!mini::json_parse(text, root)) return std::numeric_limits<double>::infinity();
        const mini::JsonValue* v = mini::json_find(root, "val");
        if (!v) return std::numeric_limits<double>::infinity();
        double val = 0.0;
        if (!mini::json_try_num(*v, val)) return std::numeric_limits<double>::infinity();
        if (val < 0.0) return std::numeric_limits<double>::infinity();
        return val;
    } catch (...) {
        return std::numeric_limits<double>::infinity();
    }
}

std::string summarise(const std::string& dir) {
    try {
        Manifest m;
        std::map<std::string, mininpz::Array> state;
        if (!load(dir, m, state)) return "step=0 val=inf n_experts=0 bytes=0";
        std::ostringstream oss;
        oss << "step=" << m.step << " val=" << m.val << " n_experts=" << m.n_experts
            << " bytes=" << m.total_bytes;
        return oss.str();
    } catch (...) {
        return "step=0 val=inf n_experts=0 bytes=0";
    }
}

}  // namespace store