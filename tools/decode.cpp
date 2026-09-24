// tools/decode.cpp
// CLI tool: load a trained model and generate text.
// Equivalent to: python3 -c "from minagi.recur import load; m=load('weights/'); ..."
//
// Usage: minagi_decode --model <weights_dir> --prompt "hello" --n 64
//        minagi_decode --model <weights_dir> --file input.txt --n 128
//
// Uses recur.hpp's minagi::Config (NOT config.hpp). The full inference
// loop (segment reading, replay, decode) is extracted from test_wavef.cpp.
#include "../src/tensor.hpp"
#include "../src/init.hpp"
#include "../src/model_create.hpp"
#include "../src/store.hpp"
#include "../src/paged.hpp"
#include "../src/recur.hpp"
#include "../src/decode.hpp"
#include "../src/backward.hpp"
#include "../src/optimizer.hpp"
#include "../src/tokenizer.hpp"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <cstring>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace m = minagi;
namespace pg = minagi::paged;

static void usage() {
    std::cerr << "Usage: minagi_decode --model <dir> (--prompt \"text\" | --file <path>) [--n N] [--seed N]\n";
    std::cerr << "       minagi_decode --model <dir> --prompt \"text\" --n N [--stream-learn] [--lr LR] [--save-every N] [--out DIR]\n";
    std::cerr << "  --model      Weights directory (with manifest.json + *.npz)\n";
    std::cerr << "  --prompt     Text to start decoding from\n";
    std::cerr << "  --file       File path to read prompt text from\n";
    std::cerr << "  --n          Number of tokens to generate (default 64)\n";
    std::cerr << "  --seed       (unused, deterministic decode)\n";
    std::cerr << "  --stream-learn  Enable streaming learner mode: compute loss + backward + optimizer\n";
    std::cerr << "                  step after each generated token, updating model in-place\n";
    std::cerr << "  --lr         Learning rate for streaming (default 1e-4)\n";
    std::cerr << "  --save-every Save checkpoint every N tokens (streaming only, default 0=off)\n";
    std::cerr << "  --out        Output directory for streaming checkpoints (default: model_dir/stream)\n";
}

static std::string read_file_text(const std::string& path) {
    std::ifstream f(path);
    if (!f) return {};
    std::string s((std::istreambuf_iterator<char>(f)),
                   std::istreambuf_iterator<char>());
    return s;
}

int main(int argc, char** argv) {
    std::string model_dir, prompt_text, file_path;
    int n_gen = 64;
    bool stream_learn = false;
    float lr = 1e-4f;
    int save_every = 0;
    std::string out_dir;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) model_dir = argv[++i];
        else if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) prompt_text = argv[++i];
        else if (strcmp(argv[i], "--file") == 0 && i + 1 < argc) file_path = argv[++i];
        else if (strcmp(argv[i], "--n") == 0 && i + 1 < argc) n_gen = std::stoi(argv[++i]);
        else if (strcmp(argv[i], "--stream-learn") == 0) stream_learn = true;
        else if (strcmp(argv[i], "--lr") == 0 && i + 1 < argc) lr = std::stof(argv[++i]);
        else if (strcmp(argv[i], "--save-every") == 0 && i + 1 < argc) save_every = std::stoi(argv[++i]);
        else if (strcmp(argv[i], "--out") == 0 && i + 1 < argc) out_dir = argv[++i];
        else { usage(); return 1; }
    }

    if (model_dir.empty()) { usage(); return 1; }
    if (prompt_text.empty() && file_path.empty()) { usage(); return 1; }
    if (file_path.size()) prompt_text = read_file_text(file_path);

    // --- Load model ---
    ::store::Manifest man;
    std::map<std::string, mininpz::Array> state;
    if (!::store::load(model_dir, man, state)) {
        std::cerr << "Error: could not load model from " << model_dir << "\n";
        return 1;
    }

    minagi::Config cfg;
    if (!cfg.from_manifest(man.cfg_obj)) {
        std::cerr << "Error: could not parse config from manifest\n";
        return 1;
    }

    std::vector<std::pair<std::string, mt::Tensor>> sd;
    for (auto& kv : state) sd.emplace_back(kv.first, pg::array_to_tensor(kv.second));

    mt::Shape gsha; gsha.rank = 1; gsha.d[0] = cfg.pool_experts;
    mt::Tensor ones = mt::make(gsha, mt::DType::FP32, 1.0f);
    pg::PagedPool pool(cfg.pool_experts, cfg.d_model, cfg.pool_d_ff,
                       cfg.pool_resident, man.pool_ram, cfg.explore,
                       cfg.margin, cfg.dwell, man.read_only, ones);
    fs::path wdir = fs::path(model_dir);
    pool.set_experts_dir((wdir / "experts").string());
    {
        const mininpz::Array* ga = nullptr;
        for (auto& kv : state) if (kv.first == "pool.gate") ga = &kv.second;
        if (ga) pool.set_gate(pg::array_to_tensor(*ga));
    }
    pool.load_telemetry(man.telemetry);

    m::Coder coder(cfg);
    if (!coder.load_weights(sd, pool)) {
        std::cerr << "Error: load_weights failed\n";
        return 1;
    }
    coder.set_pool(&pool);

    // Optimizer for streaming learner mode
    minagi::AdamW opt(lr, 0.01f, 0.9f, 0.999f, 1e-8f);

    // --- Tokenize prompt ---
    minagi::ByteTokenizer tok;
    std::vector<int32_t> ids = tok.encode(prompt_text);
    if (ids.empty()) {
        std::cerr << "Error: prompt is empty\n";
        return 1;
    }

    const int V = cfg.vocab_size;
    auto caches = coder.empty_caches();
    std::vector<int64_t> prev;
    for (int32_t id : ids) prev.push_back(id);

    // --- Phase A: read prompt in segments ---
    coder.begin_segment();
    int offset = 0;
    while (offset < static_cast<int>(prev.size())) {
        int remaining = static_cast<int>(prev.size()) - offset;
        int L = std::min(remaining, cfg.block);
        std::vector<int64_t> chunk(prev.begin() + offset, prev.begin() + offset + L);

        // Encode ids as FP32 tensor (forward reads int from float via cast)
        mt::Shape s; s.rank = 2; s.d[0] = 1; s.d[1] = L;
        mt::Tensor idx = mt::make(s, mt::DType::FP32, 0.0f);
        float* p = idx.ptr<float>();
        for (int i = 0; i < L; ++i) p[i] = static_cast<float>(chunk[i]);

        m::StepOut so = coder.forward(idx, caches, offset);
        offset += L;
    }

    // --- Phase B: Greedy decode ---
    int cur = static_cast<int>(prev.back());
    for (int t = 0; t < n_gen; ++t) {
        mt::Shape s; s.rank = 2; s.d[0] = 1; s.d[1] = 1;
        mt::Tensor idx = mt::make(s, mt::DType::FP32, 0.0f);
        idx.ptr<float>()[0] = static_cast<float>(cur);
        m::StepOut so = coder.forward(idx, caches, offset);

        if (so.logits.shape.rank == 0) break;
        if (so.logits.shape.rank != 3) break;

        mt::Shape ls; ls.rank = 2; ls.d[0] = 1; ls.d[1] = V;
        mt::Tensor logits1d = mt::make(ls, mt::DType::FP32, 0.0f);
        std::memcpy(logits1d.ptr<float>(), so.logits.ptr<float>(),
                    static_cast<size_t>(V) * sizeof(float));

        // adaptation window: last 64 tokens
        std::vector<int64_t> window;
        int nwin = std::min(static_cast<int>(prev.size()), 64);
        window.assign(prev.end() - nwin, prev.end());
        mt::Shape ws; ws.rank = 2; ws.d[0] = 1; ws.d[1] = nwin;
        mt::Tensor pw = mt::make(ws, mt::DType::FP32, 0.0f);
        float* wp = pw.ptr<float>();
        for (int i = 0; i < nwin; ++i) wp[i] = static_cast<float>(window[i]);

        m::DecodeWeights dw;
        int nxt = m::pick_next(logits1d, pw, dw);
        if (nxt < 0 || nxt >= V) break;

        prev.push_back(nxt);
        cur = nxt;
        ++offset;

        // output character
        std::vector<int32_t> one{static_cast<int32_t>(nxt)};
        std::cout << tok.decode(one);
        std::cout.flush();

        // --- Streaming learner: compute loss + backward + optimizer step ---
        if (stream_learn) {
            // Build training target: next-token prediction from the adaptation window
            // Target = the token we just generated (nxt)
            // Input = last cfg.block tokens from prev (or fewer if < block)
            int ctx_len = std::min(static_cast<int>(prev.size()) - 1, cfg.block);
            if (ctx_len >= 2) {
                // Reset caches and re-run forward to get fresh backward cache
                auto stream_caches = coder.empty_caches();
                coder.begin_segment();

                // Input: tokens [prev[-ctx_len], ..., prev[-1]]
                // Target: [prev[-ctx_len+1], ..., prev[-1+1]] = [prev[-ctx_len+1], ..., nxt-1]
                mt::Shape shp; shp.rank = 2; shp.d[0] = 1; shp.d[1] = ctx_len;
                mt::Tensor idx = mt::make(shp, mt::DType::FP32, 0.0f);
                mt::Tensor tg = mt::make(shp, mt::DType::FP32, 0.0f);
                float* ip = idx.ptr<float>();
                float* tp = tg.ptr<float>();
                for (int i = 0; i < ctx_len; ++i) {
                    ip[i] = static_cast<float>(prev[prev.size() - ctx_len + i]);
                    tp[i] = static_cast<float>(prev[prev.size() - ctx_len + i + 1]);
                }

                minagi::backward::Gradients grads;
                float loss = minagi::backward::compute_loss_and_grads(
                    cfg, coder, idx, tg, 0, stream_caches, grads);

                // NaN check
                bool has_nan = false;
                for (auto& [name, grad] : grads.grads) {
                    const float* gp = grad.ptr<float>();
                    for (int64_t i = 0; i < grad.shape.numel(); ++i) {
                        if (std::isnan(gp[i]) || std::isinf(gp[i])) {
                            has_nan = true;
                            std::cerr << "NaN/Inf in grad: " << name << "\n";
                            break;
                        }
                    }
                    if (has_nan) break;
                }
                if (!has_nan) {
                    std::vector<std::pair<std::string, std::pair<mt::Tensor*, mt::Tensor*>>> paired;
                    for (auto& [name, grad] : grads.grads) {
                        mt::Tensor* param = coder.get_param(name);
                        if (param) {
                            paired.emplace_back(name, std::make_pair(param, &grad));
                        }
                    }
                    opt.step(paired);
                }
            }
        }

        // Checkpoint save (streaming mode)
        if (stream_learn && save_every > 0 && (t + 1) % save_every == 0) {
            std::string save_dir = out_dir.empty()
                ? (model_dir + "/stream")
                : out_dir;
            fs::create_directories(save_dir + "/experts");

            // Collect all coder params for saving
            std::vector<std::pair<std::string, mt::Tensor>> sd;
            sd.emplace_back("tok_emb.weight", coder.tok_emb());
            sd.emplace_back("ln_f.weight", coder.ln_f_w());
            sd.emplace_back("adapter.weight", coder.adapter_w());
            sd.emplace_back("halt.weight", coder.halt_w());
            sd.emplace_back("halt.bias", coder.halt_b());
            for (int r = 0; r < cfg.n_prelude + cfg.n_recur + cfg.n_coda; ++r) {
                const auto& b = coder.blocks()[r];
                std::string prefix;
                if (r < cfg.n_prelude) prefix = "prelude." + std::to_string(r) + ".";
                else if (r < cfg.n_prelude + cfg.n_recur) prefix = "recur." + std::to_string(r - cfg.n_prelude) + ".";
                else prefix = "coda." + std::to_string(r - cfg.n_prelude - cfg.n_recur) + ".";
                sd.emplace_back(prefix + "ln1.weight", b.ln1);
                sd.emplace_back(prefix + "attn.qkv.weight", b.qkv);
                sd.emplace_back(prefix + "attn.proj.weight", b.proj);
                sd.emplace_back(prefix + "ln2.weight", b.ln2);
                if (b.w1.shape.rank > 0) {
                    sd.emplace_back(prefix + "mlp.w1.weight", b.w1);
                    sd.emplace_back(prefix + "mlp.w3.weight", b.w3);
                    sd.emplace_back(prefix + "mlp.w2.weight", b.w2);
                }
                sd.emplace_back(prefix + "mlp.router.weight", b.router);
                sd.emplace_back(prefix + "mlp.depth_emb", b.depth_emb);
            }
            // Add pool gate
            sd.emplace_back("pool.gate", pool.gate());

            store::SaveResult result;
            if (pool.n_experts() > 0) {
                store::save_paged(sd, pool, man.cfg_obj, save_dir,
                                  t + 1, 0.0f, &result);
            }
            std::cout << "\n[saved stream checkpoint to " << save_dir
                      << " (" << result.n_experts << " experts)]\n";
        }
    }

    std::cout << "\n";
    return 0;
}
