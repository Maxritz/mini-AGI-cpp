// tools/train.cpp
// CPU training loop for mini-AGI: forward -> cross-entropy loss -> backward -> AdamW update
//
// Equivalent to minagi/live.py. Streams characters from a text file,
// builds chunks, trains the recurrence (prelude + n_recur blocks) with
// halting, growth/prune of expert pool, and AdamW optimizer.
//
// Usage:
//   minagi_train --data <corpus.txt> [--steps N] [--lr F] [--chunk N]
//                [--save_every N] [--out <dir>] [--seed N]
//
// If --data is omitted, a small synthetic dataset is generated.
#include "../src/tensor.hpp"
#include "../src/init.hpp"
#include "../src/recur.hpp"
#include "../src/model.hpp"
#include "../src/store.hpp"
#include "../src/optimizer.hpp"
#include "../src/backward.hpp"
#include "../src/paged.hpp"
#include "../src/mininpz.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace {

struct Args {
    std::string data_path;
    std::string out_dir;
    int steps = 1000;
    float lr = 3e-4f;
    int chunk = 8;
    int save_every = 8;
    int seed = 42;
};

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--data") == 0 && i + 1 < argc) a.data_path = argv[++i];
        else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) a.out_dir = argv[++i];
        else if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc) a.steps = atoi(argv[++i]);
        else if (strcmp(argv[i], "--lr") == 0 && i + 1 < argc) a.lr = static_cast<float>(atof(argv[++i]));
        else if (strcmp(argv[i], "--chunk") == 0 && i + 1 < argc) a.chunk = atoi(argv[++i]);
        else if (strcmp(argv[i], "--save_every") == 0 && i + 1 < argc) a.save_every = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) a.seed = atoi(argv[++i]);
    }
    return a;
}

std::vector<int32_t> generate_synthetic_data(int vocab, int n_tokens, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int32_t> dist(0, vocab - 1);
    std::vector<int32_t> data(n_tokens);
    for (int i = 0; i < n_tokens; ++i) data[i] = dist(rng);
    return data;
}

}  // namespace

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);

    minagi::Config cfg;
    cfg.d_model = 64;
    cfg.n_head = 4;
    cfg.n_layer = 2;
    cfg.n_recur = 2;
    cfg.n_prelude = 1;
    cfg.n_coda = 0;
    cfg.max_steps = 4;
    cfg.min_steps = 1;
    cfg.halt_thresh = 0.85;
    cfg.d_ff = 128;
    cfg.vocab_size = 256;
    cfg.use_pool = true;
    cfg.pool_experts = 8;
    cfg.pool_d_ff = 128;
    cfg.pool_depth = 1;
    cfg.pool_top_k = 2;
    cfg.pool_resident = 4;

    minagi::Coder coder(cfg);
    coder.random_init(static_cast<unsigned>(args.seed));

    // Set up PagedPool when using pooled experts
    minagi::paged::PagedPool* pool_ptr = nullptr;
    std::unique_ptr<minagi::paged::PagedPool> pool;
    std::string experts_base = args.out_dir.empty()
        ? "trained_model/experts"
        : (args.out_dir + "/experts");
    if (cfg.use_pool) {
        mt::Shape gsha; gsha.rank = 1; gsha.d[0] = cfg.pool_experts;
        mt::Tensor gate = mt::make(gsha, mt::DType::FP32, 1.0f);
        pool = std::make_unique<minagi::paged::PagedPool>(
            cfg.pool_experts, cfg.d_model, cfg.pool_d_ff, cfg.pool_resident,
            cfg.pool_resident, 0.15, 0.10, 4, false, gate);
        pool_ptr = pool.get();
        coder.set_pool(pool_ptr);

        // Write initial expert weight files to a persistent "experts" dir
        std::filesystem::create_directories(experts_base);
        minagi::PcgRng init_rng(static_cast<unsigned>(args.seed));
        for (int eid = 0; eid < cfg.pool_experts; ++eid) {
            mt::Shape w1s; w1s.rank = 2; w1s.d[0] = cfg.pool_d_ff; w1s.d[1] = cfg.d_model;
            mt::Tensor w1 = minagi::init_normal(w1s, init_rng, 0.0f, 0.02f);
            mt::Tensor w3 = minagi::init_normal(w1s, init_rng, 0.0f, 0.02f);
            mt::Shape w2s; w2s.rank = 2; w2s.d[0] = cfg.d_model; w2s.d[1] = cfg.pool_d_ff;
            mt::Tensor w2 = minagi::init_normal(w2s, init_rng, 0.0f, 0.02f);

            // e%05d.npz: must match Tiers::file_path / PagedPool::expert_uid or
            // reload can never find these experts (STATUS train gap #3).
            char ename[32];
            std::snprintf(ename, sizeof(ename), "e%05d.npz", eid);
            std::string path = experts_base + "/" + ename;
            std::vector<mininpz::NpzEntry> entries;
            entries.push_back({"w1.npy", minagi::paged::tensor_to_array(w1)});
            entries.push_back({"w3.npy", minagi::paged::tensor_to_array(w3)});
            entries.push_back({"w2.npy", minagi::paged::tensor_to_array(w2)});
            mininpz::write_npz(path, entries);
        }
        pool_ptr->set_experts_dir(experts_base);
        // Initial swap: a fresh pool has slots=[-1]*resident and zero weights.
        // Without this, all routing collapses to expert 0's zeros and no pool
        // gradient ever flows. (No reselect protocol in this toy loop.)
        coder.begin_segment();
    }

    std::vector<int32_t> tokens;
    if (!args.data_path.empty()) {
        std::ifstream f(args.data_path, std::ios::binary);
        if (!f) {
            std::cerr << "Cannot open data file: " << args.data_path << "\n";
            return 1;
        }
        std::vector<char> buf((std::istreambuf_iterator<char>(f)),
                              std::istreambuf_iterator<char>());
        for (char c : buf) {
            tokens.push_back(static_cast<int32_t>(static_cast<uint8_t>(c)) % cfg.vocab_size);
        }
    } else {
        tokens = generate_synthetic_data(cfg.vocab_size, args.steps * args.chunk + args.chunk + 10, args.seed);
    }

    if (tokens.size() < static_cast<size_t>(args.chunk + 2)) {
        std::cerr << "Not enough data. Need at least " << (args.chunk + 2)
                  << " tokens, got " << tokens.size() << "\n";
        return 1;
    }

    minagi::AdamW opt(args.lr, 0.0f, 0.9f, 0.999f, 1e-8f, 1.0f);

    std::vector<model::KVCache> caches = coder.empty_caches();

    int pos = 0;
    float total_loss = 0.0f;

    std::cout << "Starting training: " << args.steps << " steps, chunk=" << args.chunk
              << ", lr=" << args.lr << ", vocab=" << cfg.vocab_size << "\n";

    for (int step = 0; step < args.steps; ++step) {
        // Reset KV caches each step (no cross-sequence attention)
        caches = coder.empty_caches();
        int start = pos % (static_cast<int>(tokens.size()) - args.chunk - 1);
        if (start < 0) start = 0;

        // Build input [1, chunk] and target [1, chunk] (next-token prediction)
        mt::Shape shp; shp.rank = 2; shp.d[0] = 1; shp.d[1] = args.chunk;
        mt::Tensor idx = mt::make(shp, mt::DType::FP32, 0.0f);
        mt::Tensor tg = mt::make(shp, mt::DType::FP32, 0.0f);
        for (int i = 0; i < args.chunk; ++i) {
            idx.set_flat(i, static_cast<float>(tokens[start + i]));
            tg.set_flat(i, static_cast<float>(tokens[start + i + 1]));
        }
        pos = (start + args.chunk) % (static_cast<int>(tokens.size()) - args.chunk - 1);

        int64_t pos_offset = static_cast<int64_t>(start);

        minagi::backward::Gradients grads;
        std::cout << "Step " << step << " forward..." << std::flush;
        float loss = minagi::backward::compute_loss_and_grads(
            cfg, coder, idx, tg, pos_offset,
            caches, grads);

        total_loss += loss;

        // Optimizer step for each gradient
        bool has_nan = false;
        for (auto& [name, grad] : grads.grads) {
            const float* gp = grad.ptr<float>();
            for (int64_t i = 0; i < grad.shape.numel(); ++i) {
                if (std::isnan(gp[i]) || std::isinf(gp[i])) {
                    has_nan = true;
                    std::cerr << "NaN/Inf in grad: " << name << " idx=" << i << " val=" << gp[i] << "\n";
                    break;
                }
            }
            if (has_nan) break;
        }
        if (has_nan) {
            std::cerr << "NaN gradients detected, skipping optimizer step\n";
        } else {
            // Build paired params for opt.step
            std::vector<std::pair<std::string, std::pair<mt::Tensor*, mt::Tensor*>>> paired;
            for (auto& [name, grad] : grads.grads) {
                mt::Tensor* param = coder.get_param(name);
                if (param) {
                    paired.emplace_back(name, std::make_pair(param, &grad));
                }
            }
            opt.step(paired);

            // Expert + gate stepping (pool params live in the PagedPool, not
            // the coder). Expert grads are keyed pool.experts.{uid}.w*.weight
            // with per-uid AdamW moments (stable across swaps); the resident
            // row is stepped via copy-out/update/copy-back and persisted by
            // pool flush before checkpointing.
            if (pool_ptr) {
                const std::vector<int>& pslots = pool_ptr->slots();
                auto step_expert_row = [&](const std::string& leaf,
                                           mt::Tensor& resident, int64_t row_numel) {
                    for (size_t s = 0; s < pslots.size(); ++s) {
                        const int pos = pslots[s];
                        if (pos < 0) continue;
                        const int uid = pool_ptr->expert_uid(pos);
                        if (uid < 0) continue;
                        const std::string key = "pool.experts." +
                                                std::to_string(uid) + "." + leaf;
                        mt::Tensor* g = grads.get(key);
                        if (!g) continue;
                        mt::Tensor row;
                        pool_ptr->row_of(resident, static_cast<int>(s), row_numel, row);
                        opt.update_param(row, *g, key);
                        minagi::paged::PagedPool::set_row(resident, static_cast<int>(s), row);
                    }
                };
                const int64_t row12 =
                    static_cast<int64_t>(cfg.pool_d_ff) * cfg.d_model;
                step_expert_row("w1.weight", pool_ptr->mutable_w1(), row12);
                step_expert_row("w3.weight", pool_ptr->mutable_w3(), row12);
                step_expert_row("w2.weight", pool_ptr->mutable_w2(), row12);
                // Gate: copy-step-restore (moments keyed "pool.gate").
                mt::Tensor* gg = grads.get("pool.gate");
                if (gg) {
                    mt::Tensor gate = pool_ptr->gate();
                    opt.update_param(gate, *gg, "pool.gate");
                    pool_ptr->set_gate(gate);
                }
            }
        }

        if (step % 10 == 0 || step == args.steps - 1) {
            float avg_loss = total_loss / static_cast<float>(step + 1);
            std::cout << "Step " << step << "/" << args.steps
                      << " loss=" << avg_loss << "\n";
        }

        // Growth: every 50 steps, try adding an expert
        if (pool_ptr && (step + 1) % 50 == 0 && pool_ptr->n_experts() < 64) {
            minagi::PcgRng rng(static_cast<uint64_t>(step * 7919 + 42));
            int n_prev = pool_ptr->n_experts();
            pool_ptr->grow(1, step, 0.001, 0.02f, rng, experts_base);
            // Grow per-site routers to match (they index into pool experts)
            for (int r = 0; r < cfg.n_recur; ++r) {
                std::string key = "recur." + std::to_string(r) + ".mlp.router.weight";
                mt::Tensor* router = coder.get_param(key);
                if (router && router->shape.d[0] < pool_ptr->n_experts()) {
                    mt::Tensor new_router;
                    mt::Shape s = router->shape;
                    s.d[0] = pool_ptr->n_experts();
                    new_router = mt::make(s, mt::DType::FP32, 0.0f);
                    float* dst = new_router.ptr<float>();
                    const float* src = router->ptr<float>();
                    for (int i = 0; i < router->shape.d[0]; ++i) {
                        std::memcpy(dst + i * router->shape.d[1],
                                    src + i * router->shape.d[1],
                                    router->shape.d[1] * sizeof(float));
                    }
                    *router = new_router;
                }
            }
            // Grow segment_router to match
            {
                const mt::Tensor& sr = pool_ptr->segment_router();
                if (sr.shape.d[0] < pool_ptr->n_experts()) {
                    mt::Tensor new_sr;
                    mt::Shape s; s.rank = 2;
                    s.d[0] = pool_ptr->n_experts();
                    s.d[1] = sr.shape.d[1];
                    new_sr = mt::make(s, mt::DType::FP32, 0.0f);
                    float* dst = new_sr.ptr<float>();
                    const float* src = sr.ptr<float>();
                    for (int i = 0; i < sr.shape.d[0]; ++i) {
                        std::memcpy(dst + i * sr.shape.d[1],
                                    src + i * sr.shape.d[1],
                                    sr.shape.d[1] * sizeof(float));
                    }
                    pool_ptr->set_segment_router(new_sr);
                }
            }
            std::cout << "Growth: " << n_prev << " -> "
                      << pool_ptr->n_experts() << " experts\n";
        }

        // Pruning: every 200 steps, remove stale experts
        if (pool_ptr && (step + 1) % 200 == 0 && pool_ptr->n_experts() > 1) {
            int n_prev = pool_ptr->n_experts();
            int removed = pool_ptr->prune(step, 100, /*protect=*/1);
            if (removed > 0) {
                std::cout << "Prune: " << n_prev << " -> "
                          << pool_ptr->n_experts() << " experts (removed "
                          << removed << ")\n";
            }
        }

        // Checkpoint save
        if ((step + 1) % args.save_every == 0) {
            std::string save_dir = args.out_dir.empty()
                ? ("ckpt_step" + std::to_string(step + 1))
                : (args.out_dir + "/step" + std::to_string(step + 1));

            // Flush stepped residents to experts_base FIRST so the checkpoint
            // copies current (not initial) expert weights. Without this the
            // experts train in RAM and the checkpoint silently keeps step-0.
            if (pool_ptr) pool_ptr->flush();

            // Set pool's experts_dir to checkpoint dir and copy expert files there
            if (pool_ptr) {
                std::string ckpt_experts = save_dir + "/experts";
                std::filesystem::create_directories(ckpt_experts);
                // Copy all expert files from base dir to checkpoint dir
                if (std::filesystem::exists(experts_base)) {
                    for (auto& p : std::filesystem::directory_iterator(experts_base)) {
                        if (p.is_regular_file() && p.path().extension() == ".npz") {
                            std::filesystem::copy_file(p.path(),
                                ckpt_experts + "/" + p.path().filename().string(),
                                std::filesystem::copy_options::overwrite_existing);
                        }
                    }
                }
                pool_ptr->set_experts_dir(ckpt_experts);
            }

            // Collect state dict from coder (params that have gradients)
            std::vector<std::pair<std::string, mt::Tensor>> sd;
            for (const auto& [name, grad] : grads.grads) {
                const mt::Tensor* param = coder.get_param(name);
                if (param) {
                    sd.emplace_back(name, *param);
                }
            }
            // Add pool params that don't get gradients in the simplified backward
            if (pool_ptr) {
                sd.emplace_back("pool.gate", pool_ptr->gate());
                // Ensure segment_router has the right number of rows
                const mt::Tensor& sr = pool_ptr->segment_router();
                if (sr.shape.d[0] < pool_ptr->n_experts()) {
                    // Grow segment_router to match current expert count
                    mt::Tensor new_sr;
                    mt::Shape s; s.rank = 2;
                    s.d[0] = pool_ptr->n_experts();
                    s.d[1] = sr.shape.d[1];
                    new_sr = mt::make(s, mt::DType::FP32, 0.0f);
                    // Copy existing rows
                    float* dst = new_sr.ptr<float>();
                    const float* src = sr.ptr<float>();
                    for (int i = 0; i < sr.shape.d[0]; ++i) {
                        std::memcpy(dst + i * sr.shape.d[1],
                                    src + i * sr.shape.d[1],
                                    sr.shape.d[1] * sizeof(float));
                    }
                    // New rows get zeros
                    sd.emplace_back("pool.segment_router.weight", new_sr);
                } else {
                    sd.emplace_back("pool.segment_router.weight", sr);
                }
            }

            // Build cfg_obj JSON with actual config fields
            mini::JsonValue cfg_obj;
            cfg_obj.t = mini::JsonValue::Type::Obj;
            cfg_obj.o["d_model"] = mini::JsonValue::make_num(static_cast<double>(cfg.d_model));
            cfg_obj.o["n_head"] = mini::JsonValue::make_num(static_cast<double>(cfg.n_head));
            cfg_obj.o["n_layer"] = mini::JsonValue::make_num(static_cast<double>(cfg.n_layer));
            cfg_obj.o["n_prelude"] = mini::JsonValue::make_num(static_cast<double>(cfg.n_prelude));
            cfg_obj.o["n_recur"] = mini::JsonValue::make_num(static_cast<double>(cfg.n_recur));
            cfg_obj.o["n_coda"] = mini::JsonValue::make_num(static_cast<double>(cfg.n_coda));
            cfg_obj.o["d_ff"] = mini::JsonValue::make_num(static_cast<double>(cfg.d_ff));
            cfg_obj.o["vocab_size"] = mini::JsonValue::make_num(static_cast<double>(cfg.vocab_size));
            cfg_obj.o["block"] = mini::JsonValue::make_num(static_cast<double>(cfg.block));
            cfg_obj.o["tie_embeddings"] = mini::JsonValue::make_bool(cfg.tie_embeddings);
            cfg_obj.o["use_pool"] = mini::JsonValue::make_bool(cfg.use_pool);
            cfg_obj.o["pool_experts"] = mini::JsonValue::make_num(static_cast<double>(pool_ptr->n_experts()));
            cfg_obj.o["pool_d_ff"] = mini::JsonValue::make_num(static_cast<double>(cfg.pool_d_ff));
            cfg_obj.o["pool_depth"] = mini::JsonValue::make_num(static_cast<double>(cfg.pool_depth));
            cfg_obj.o["pool_top_k"] = mini::JsonValue::make_num(static_cast<double>(cfg.pool_top_k));
            cfg_obj.o["pool_capacity_factor"] = mini::JsonValue::make_num(cfg.pool_capacity_factor);
            cfg_obj.o["pool_max"] = mini::JsonValue::make_num(static_cast<double>(pool_ptr->n_experts()));
            cfg_obj.o["pool_resident"] = mini::JsonValue::make_num(static_cast<double>(cfg.pool_resident));
            cfg_obj.o["max_steps"] = mini::JsonValue::make_num(static_cast<double>(cfg.max_steps));
            cfg_obj.o["min_steps"] = mini::JsonValue::make_num(static_cast<double>(cfg.min_steps));
            cfg_obj.o["halt_prior"] = mini::JsonValue::make_num(cfg.halt_prior);
            cfg_obj.o["halt_thresh"] = mini::JsonValue::make_num(cfg.halt_thresh);
            cfg_obj.o["ponder_beta"] = mini::JsonValue::make_num(cfg.ponder_beta);
            cfg_obj.o["train_steps_mean"] = mini::JsonValue::make_num(cfg.train_steps_mean);
            cfg_obj.o["rope_theta"] = mini::JsonValue::make_num(cfg.rope_theta);
            cfg_obj.o["explore"] = mini::JsonValue::make_num(cfg.explore);
            cfg_obj.o["margin"] = mini::JsonValue::make_num(cfg.margin);
            cfg_obj.o["dwell"] = mini::JsonValue::make_num(static_cast<double>(cfg.dwell));

            // Add optimizer moments
            for (const auto& kv : opt.state_dict()) {
                sd.push_back(kv);
            }

            // Build gates map
            std::map<int, double> gates;
            if (pool_ptr) {
                const mt::Tensor& g = pool_ptr->gate();
                for (int i = 0; i < g.shape.d[0]; ++i) {
                    gates[i] = static_cast<double>(g.ptr<float>()[i]);
                }
            }

            store::SaveResult result;
            if (pool_ptr) {
                std::map<std::string, std::string> cfg_map;
                if (store::save_paged(sd, *pool_ptr, cfg_obj, save_dir, step, 0.0, &result)) {
                    std::cout << "Saved checkpoint to " << save_dir
                              << " (" << result.n_experts << " experts)\n";
                } else {
                    std::cerr << "Warning: failed to save checkpoint to " << save_dir << "\n";
                }
            } else {
                std::cerr << "Warning: checkpoint save requires a PagedPool (use_pool=true)\n";
            }
        }
    }

    std::cout << "Training complete. Final avg loss: "
              << (total_loss / static_cast<float>(args.steps)) << "\n";

    return 0;
}
