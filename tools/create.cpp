// tools/create.cpp
// CLI tool: create a fresh mini-AGI model directory from config.
// Equivalent to: python3 minagi/create.py --out <dir> --seed <seed>
//
// Uses recur.hpp's minagi::Config (NOT config.hpp — different struct).
#include "../src/tensor.hpp"
#include "../src/init.hpp"
#include "../src/model_create.hpp"
#include "../src/store.hpp"
#include "../src/recur.hpp"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <cstring>

namespace fs = std::filesystem;

static void usage() {
    std::cerr << "Usage: minagi_create --out <dir> [--seed <N>] [--config <yaml>] [--force]\n";
    std::cerr << "  --out   Output weights directory (created if missing, --force to overwrite)\n";
    std::cerr << "  --seed  RNG seed (default 0)\n";
    std::cerr << "  --config  Path to config.yaml (default: config.yaml in CWD)\n";
    std::cerr << "  --force  Overwrite if directory exists\n";
}

// Minimal YAML parser for the subset we need:
//   model:
//     d_model: 24
//   pool:
//     experts: 6
static minagi::Config parse_config_file(const std::string& path) {
    minagi::Config cfg;  // all defaults
    std::ifstream f(path);
    if (!f) return cfg;
    std::string line, section;
    while (std::getline(f, line)) {
        size_t hash = line.find('#');
        if (hash != std::string::npos) line = line.substr(0, hash);
        size_t start = line.find_first_not_of(" \t");
        if (start == std::string::npos) continue;
        size_t end = line.find_last_not_of(" \t");
        line = line.substr(start, end - start + 1);
        if (line.empty()) continue;
        if (line.back() == ':') {
            section = line.substr(0, line.size() - 1);
            continue;
        }
        size_t colon = line.find(':');
        if (colon == std::string::npos) continue;
        std::string key = line.substr(0, colon);
        std::string val = line.substr(colon + 1);
        start = val.find_first_not_of(" \t");
        end = val.find_last_not_of(" \t");
        if (start != std::string::npos)
            val = val.substr(start, end - start + 1);
        std::string full = section.empty() ? key : (section + "." + key);
        if (full == "model.d_model") cfg.d_model = std::stoi(val);
        else if (full == "model.n_head") cfg.n_head = std::stoi(val);
        else if (full == "model.d_ff") cfg.d_ff = std::stoi(val);
        else if (full == "model.n_prelude") cfg.n_prelude = std::stoi(val);
        else if (full == "model.n_recur") cfg.n_recur = std::stoi(val);
        else if (full == "model.n_coda") cfg.n_coda = std::stoi(val);
        else if (full == "model.max_steps") cfg.max_steps = std::stoi(val);
        else if (full == "model.min_steps") cfg.min_steps = std::stoi(val);
        else if (full == "model.halt_prior") cfg.halt_prior = std::stod(val);
        else if (full == "model.halt_thresh") cfg.halt_thresh = std::stod(val);
        else if (full == "model.tied_embeddings") cfg.tie_embeddings = (val == "true" || val == "1");
        else if (full == "pool.experts") cfg.pool_experts = std::stoi(val);
        else if (full == "pool.d_ff") cfg.pool_d_ff = std::stoi(val);
        else if (full == "pool.depth") cfg.pool_depth = std::stoi(val);
        else if (full == "pool.top_k") cfg.pool_top_k = std::stoi(val);
        else if (full == "pool.resident") cfg.pool_resident = std::stoi(val);
        else if (full == "pool.capacity_factor") cfg.pool_capacity_factor = std::stod(val);
    }
    return cfg;
}

int main(int argc, char** argv) {
    std::string out_dir, cfg_path;
    uint64_t seed = 0;
    bool force = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) out_dir = argv[++i];
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = std::stoull(argv[++i]);
        else if (strcmp(argv[i], "--config") == 0 && i + 1 < argc) cfg_path = argv[++i];
        else if (strcmp(argv[i], "--force") == 0) force = true;
        else { usage(); return 1; }
    }

    if (out_dir.empty()) { usage(); return 1; }

    // Load config
    minagi::Config cfg;
    if (cfg_path.empty()) cfg_path = "config.yaml";
    if (fs::exists(cfg_path)) {
        cfg = parse_config_file(cfg_path);
    } else {
        std::cout << "Warning: " << cfg_path << " not found, using defaults\n";
    }

    // Handle existing directory
    if (fs::exists(out_dir)) {
        if (!force) {
            std::cerr << "Error: " << out_dir << " exists. Use --force to overwrite.\n";
            return 1;
        }
        fs::remove_all(out_dir);
    }
    fs::create_directories(out_dir);

    // Build model
    minagi::PcgRng rng(seed);
    auto model_tensors = minagi::build_model(cfg, rng);
    auto state = minagi::tensors_to_state(model_tensors);

    // Build gates map for manifest
    std::map<int, double> gates;
    for (int i = 0; i < cfg.pool_experts; ++i) {
        gates[i] = 1.0;
    }

    // Build cfg string map for manifest
    std::map<std::string, std::string> cfg_map = {
        {"d_model", std::to_string(cfg.d_model)},
        {"n_head", std::to_string(cfg.n_head)},
        {"d_ff", std::to_string(cfg.d_ff)},
        {"vocab_size", std::to_string(cfg.vocab_size)},
        {"block", std::to_string(cfg.block)},
        {"tie_embeddings", cfg.tie_embeddings ? "true" : "false"},
        {"use_pool", cfg.use_pool ? "true" : "false"},
        {"pool_experts", std::to_string(cfg.pool_experts)},
        {"pool_d_ff", std::to_string(cfg.pool_d_ff)},
        {"pool_depth", std::to_string(cfg.pool_depth)},
        {"pool_top_k", std::to_string(cfg.pool_top_k)},
        {"pool_capacity_factor", std::to_string(cfg.pool_capacity_factor)},
        {"pool_max", std::to_string(cfg.pool_max)},
        {"n_prelude", std::to_string(cfg.n_prelude)},
        {"n_recur", std::to_string(cfg.n_recur)},
        {"n_coda", std::to_string(cfg.n_coda)},
        {"max_steps", std::to_string(cfg.max_steps)},
        {"min_steps", std::to_string(cfg.min_steps)},
        {"halt_prior", std::to_string(cfg.halt_prior)},
        {"halt_thresh", std::to_string(cfg.halt_thresh)},
        {"rope_theta", std::to_string(cfg.rope_theta)},
        {"pool_resident", std::to_string(cfg.pool_resident)},
    };

    // Save model
    store::SaveResult result;
    if (!store::save(out_dir, state, /*step=*/-1, /*val=*/0.0, cfg_map, gates, result)) {
        std::cerr << "Error: failed to save model to " << out_dir << "\n";
        return 1;
    }

    int64_t total_params = 0;
    for (const auto& kv : state) {
        const auto& arr = kv.second;
        int64_t n = 1;
        for (size_t i = 0; i < arr.shape.rank; ++i) n *= static_cast<int64_t>(arr.shape.d[i]);
        total_params += n;
    }

    int per = cfg.pool_depth * (2 * cfg.d_model * cfg.pool_d_ff + cfg.d_model * cfg.d_model);
    std::cout << "\n  " << cfg.pool_experts << " experts x " << per << " parameters "
              << "(" << cfg.pool_d_ff << " hidden units, depth " << cfg.pool_depth << ")\n";
    std::cout << "  pool " << (cfg.pool_experts * per / 1e6) << "M, resident "
              << cfg.pool_resident << " = " << (cfg.pool_resident * per / 1e6) << "M in VRAM\n";
    std::cout << "  total " << (total_params / 1e6) << "M parameters\n";
    std::cout << "  Written to " << out_dir << " (" << result.n_experts << " experts)\n";

    return 0;
}
