// tools/gguf_convert.cpp
// Convert a llama.cpp GGUF (v3, Qwen/llama-style) weights file to mini-AGI's
// paged directory format (core.npz + routers.npz + manifest.json) via
// store::save_paged. Supported GGML tensor types: F32, F16, Q4_K, Q6_K.
//
// Usage:
//   minagi_gguf_convert --input <model.gguf> --output_dir <dir> [--max_blocks N]
//
// The converted model runs with use_pool=false (dense SwiGLU at every block).
// Layout mapping:
//   token_embd.weight            -> tok_emb.weight   [V, d]
//   output.weight                -> skipped (tied head, identical to token_embd)
//   blk.0.*                      -> prelude.0.*      (attention + dense SwiGLU)
//   blk.b.* (b >= 1, b-1 < N)    -> recur.{b-1}.*    (dense SwiGLU)
//   output_norm.weight           -> ln_f.weight
//   attn_q/attn_k/attn_v         -> fused attn.qkv.weight [3d, d]
//   (emergent q/k-norm weights are skipped)
//
// GGUF stores 2D weights as [out, in] row-major (matches the runtime's
// matmul(x, transpose_2d(w)) with w [out, in]), so no transposes are needed.
// Fused qkv: head h reads k/v rows at d + h*hd / 2d + h*hd (see
// recur.cpp::attn_residual); the kv projectors have only n_kv heads, so head h
// copies the kv-projection rows (h / rep)*hd with rep = H / n_kv.

#include "../src/store.hpp"
#include "../src/paged.hpp"
#include "../src/tensor.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <thread>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

// ---- error reporting ----
static void die(const std::string& msg) {
    std::cerr << "gguf_convert: " << msg << std::endl;
    std::exit(1);
}

// ---- GGUF value types ----
enum GVal : uint32_t {
    VAL_U8 = 0, VAL_I8 = 1, VAL_U16 = 2, VAL_I16 = 3,
    VAL_U32 = 4, VAL_I32 = 5, VAL_F32 = 6, VAL_BOOL = 7,
    VAL_STRING = 8, VAL_ARRAY = 9, VAL_U64 = 10, VAL_I64 = 11,
    VAL_F64 = 12
};

// GGML tensor types we support.
enum GType : uint32_t {
    TYPE_F32 = 0, TYPE_F16 = 1,
    TYPE_Q4_K = 12, TYPE_Q6_K = 14
};

// ---- little-endian readers ----
static uint64_t rd_u64(std::ifstream& f) {
    uint64_t v = 0;
    f.read(reinterpret_cast<char*>(&v), 8);
    return v;
}
static uint32_t rd_u32(std::ifstream& f) {
    uint32_t v = 0;
    f.read(reinterpret_cast<char*>(&v), 4);
    return v;
}
static uint16_t rd_u16(std::ifstream& f) {
    uint16_t v = 0;
    f.read(reinterpret_cast<char*>(&v), 2);
    return v;
}
static uint8_t rd_u8(std::ifstream& f) {
    uint8_t v = 0;
    f.read(reinterpret_cast<char*>(&v), 1);
    return v;
}
static uint16_t read_le16(const uint8_t* p) {
    return static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
}

static std::string rd_str(std::ifstream& f) {
    uint64_t n = rd_u64(f);
    if (n > (1u << 30)) die("gguf string length implausible");
    std::string s(static_cast<size_t>(n), '\0');
    if (n > 0) f.read(&s[0], static_cast<std::streamsize>(n));
    return s;
}

static float half_to_float(uint16_t h) {
    const uint32_t sign = (h >> 15) & 1;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t man = h & 0x3FF;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign << 31;
        } else {
            exp = 127 - 15 + 1;
            while (!(man & 0x400)) { man <<= 1; exp--; }
            man &= 0x3FF;
            bits = (sign << 31) | (exp << 23) | (man << 13);
        }
    } else if (exp == 0x1F) {
        bits = (sign << 31) | 0x7F800000u | (man << 13);
    } else {
        bits = (sign << 31) | ((exp - 15 + 127) << 23) | (man << 13);
    }
    float out;
    std::memcpy(&out, &bits, 4);
    return out;
}

// ---- Q4_K / Q6_K dequantization (port of ggml-quants.c) ----
// block_q4_K: fp16 d, fp16 dmin, scales[12], qs[128] -> 144 bytes (256 elems).
// block_q6_K: ql[128], qh[64], int8 scales[16], fp16 d -> 210 bytes (256 elems).

static inline uint8_t get_q4k_sc(int j, const uint8_t* q) {
    if (j < 4) return q[j] & 63;
    return (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
}
static inline uint8_t get_q4k_mn(int j, const uint8_t* q) {
    if (j < 4) return q[j + 4] & 63;
    return (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
}
static void deq_q4k_block(const uint8_t* b, float* y) {
    const float d = half_to_float(read_le16(b));
    const float dmin = half_to_float(read_le16(b + 2));
    const uint8_t* sc = b + 4;
    const uint8_t* q = b + 16;
    for (int j = 0; j < 256; j += 64) {
        const int is = j / 32;
        const float d1 = d * static_cast<float>(get_q4k_sc(is + 0, sc));
        const float m1 = dmin * static_cast<float>(get_q4k_mn(is + 0, sc));
        const float d2 = d * static_cast<float>(get_q4k_sc(is + 1, sc));
        const float m2 = dmin * static_cast<float>(get_q4k_mn(is + 1, sc));
        for (int l = 0; l < 32; ++l) y[l] = d1 * static_cast<float>(q[l] & 0xF) - m1;
        for (int l = 0; l < 32; ++l) y[32 + l] = d2 * static_cast<float>(q[l] >> 4) - m2;
        y += 64;
        q += 32;
    }
}
static void deq_q6k_block(const uint8_t* b, float* y) {
    const float d = half_to_float(read_le16(b + 208));
    const uint8_t* ql = b;
    const uint8_t* qh = b + 128;
    const int8_t* sc = reinterpret_cast<const int8_t*>(b + 192);
    for (int n = 0; n < 256; n += 128) {
        for (int l = 0; l < 32; ++l) {
            const int is = l / 16;
            const float d1 = d * static_cast<float>(sc[is + 0]);
            const float d2 = d * static_cast<float>(sc[is + 2]);
            const float d3 = d * static_cast<float>(sc[is + 4]);
            const float d4 = d * static_cast<float>(sc[is + 6]);
            const int q1 = static_cast<int8_t>(static_cast<uint8_t>((ql[l] & 0xF) |
                (((qh[l] >> 0) & 3) << 4))) - 32;
            const int q2 = static_cast<int8_t>(static_cast<uint8_t>((ql[l + 32] & 0xF) |
                (((qh[l] >> 2) & 3) << 4))) - 32;
            const int q3 = static_cast<int8_t>(static_cast<uint8_t>((ql[l] >> 4) |
                (((qh[l] >> 4) & 3) << 4))) - 32;
            const int q4 = static_cast<int8_t>(static_cast<uint8_t>((ql[l + 32] >> 4) |
                (((qh[l] >> 6) & 3) << 4))) - 32;
            y[l + 0] = d1 * static_cast<float>(q1);
            y[l + 32] = d2 * static_cast<float>(q2);
            y[l + 64] = d3 * static_cast<float>(q3);
            y[l + 96] = d4 * static_cast<float>(q4);
        }
        y += 128;
        ql += 64;
        qh += 32;
        sc += 8;
    }
}

// ---- tensor record + size helpers ----
struct GgufTensor {
    std::string name;
    std::vector<uint64_t> dims;
    uint32_t type = 0;
    uint64_t offset = 0;
};

static uint64_t tensor_ne(const GgufTensor& t) {
    uint64_t ne = 1;
    for (uint64_t d : t.dims) ne *= d;
    return ne;
}

static uint64_t tensor_bytes(const GgufTensor& t) {
    const uint64_t ne = tensor_ne(t);
    switch (t.type) {
        case TYPE_F32: return ne * 4;
        case TYPE_F16: return ne * 2;
        case TYPE_Q4_K:
            if (ne % 256 != 0) die("Q4_K tensor '" + t.name + "' has non-256-multiple element count");
            return (ne / 256) * 144;
        case TYPE_Q6_K:
            if (ne % 256 != 0) die("Q6_K tensor '" + t.name + "' has non-256-multiple element count");
            return (ne / 256) * 210;
        default:
            die("unsupported GGML tensor type " + std::to_string(t.type) +
                " for '" + t.name + "'");
            return 0;
    }
}

static std::vector<float> dequant(std::ifstream& f, uint64_t base,
                                  const GgufTensor& t, const std::string& what) {
    const uint64_t ne = tensor_ne(t);
    const uint64_t bytes = tensor_bytes(t);
    std::vector<float> out(static_cast<size_t>(ne));
    f.seekg(static_cast<std::streamoff>(base + t.offset));
    if (t.type == TYPE_F32) {
        f.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(bytes));
    } else if (t.type == TYPE_F16) {
        std::vector<uint16_t> blob(static_cast<size_t>(ne));
        f.read(reinterpret_cast<char*>(blob.data()), static_cast<std::streamsize>(bytes));
        for (size_t i = 0; i < out.size(); ++i) out[i] = half_to_float(blob[i]);
    } else {
        std::vector<uint8_t> blob(static_cast<size_t>(bytes));
        f.read(reinterpret_cast<char*>(blob.data()), static_cast<std::streamsize>(bytes));
        float* p = out.data();
        // one quantized block covers 256 floats: 144 raw bytes (Q4_K) or
        // 210 raw bytes (Q6_K). Stepping `i` by the *block* size (not 256)
        // is load-bearing: a 256 stride silently misaligns every block.
        const size_t stride = (t.type == TYPE_Q4_K) ? 144 : 210;
        const size_t n_blocks = static_cast<size_t>(bytes) / stride;
        const bool is_q4k = (t.type == TYPE_Q4_K);
        size_t nthreads = std::thread::hardware_concurrency();
        if (nthreads == 0) nthreads = 8;
        if (nthreads > n_blocks) nthreads = n_blocks;
        const uint8_t* bd = blob.data();
        auto worker = [&](size_t b0, size_t b1) {
            for (size_t b = b0; b < b1; ++b) {
                if (is_q4k) deq_q4k_block(bd + b * stride, p + b * 256);
                else deq_q6k_block(bd + b * stride, p + b * 256);
            }
        };
        if (nthreads <= 1) {
            worker(0, n_blocks);
        } else {
            std::vector<std::thread> th;
            th.reserve(nthreads);
            for (size_t w = 0; w < nthreads; ++w) {
                const size_t b0 = (n_blocks * w) / nthreads;
                const size_t b1 = (n_blocks * (w + 1)) / nthreads;
                th.emplace_back(worker, b0, b1);
            }
            for (auto& t2 : th) t2.join();
        }
    }
    if (!f.good()) die("short read dequantizing " + what + " (" + t.name + ")");
    // sanity scan: any NaN/Inf means the parse offsets or block layout are
    // wrong; dying loudly is better than saving a corrupt checkpoint.
    {
        size_t nbad = 0;
        double mn = 0.0, mx = 0.0;
        if (!out.empty()) {
            mn = mx = out[0];
            for (float v : out) {
                if (!std::isfinite(v)) { ++nbad; continue; }
                if (v < mn) mn = v;
                if (v > mx) mx = v;
            }
        }
        if (nbad > 0) {
            die("non-finite values (" + std::to_string(nbad) + " of " +
                std::to_string(out.size()) + ") dequantizing " + what +
                " (" + t.name + ")");
        }
        std::cout << "gguf_convert:   [" << what << "] n=" << out.size()
                  << " min=" << mn << " max=" << mx << std::endl;
    }
    return out;
}

static void require2(const GgufTensor* t, uint64_t r0, uint64_t r1,
                     const std::string& what) {
    if (!t) die("missing GGUF tensor for " + what);
    if (t->dims.size() != 2 || t->dims[0] != r0 || t->dims[1] != r1) {
        die("unexpected shape for " + what + ": expected [" +
            std::to_string(r0) + "," + std::to_string(r1) + "]");
    }
}

static void require1(const GgufTensor* t, uint64_t r0,
                     const std::string& what) {
    if (!t) die("missing GGUF tensor for " + what);
    if (t->dims.size() != 1 || t->dims[0] != r0) {
        die("unexpected shape for " + what + ": expected [" +
            std::to_string(r0) + "]");
    }
}

// ---- tensor construction (fp32, 1-D or 2-D) ----
static mt::Tensor make_tensor(std::vector<float>&& vals, int64_t r0, int64_t r1 = 1) {
    mt::Tensor t;
    t.dtype = mt::DType::FP32;
    if (r1 > 1) {
        t.shape.rank = 2;
        t.shape.d[0] = r0;
        t.shape.d[1] = r1;
    } else {
        t.shape.rank = 1;
        t.shape.d[0] = r0;
    }
    const size_t n = vals.size() * sizeof(float);
    t.data_.resize(n);
    std::memcpy(t.data_.data(), vals.data(), n);
    return t;
}

// ---- typed config object ----
static mini::JsonValue typed_cfg(int vocab_size, int n_layer, int n_head,
                                 int d_model, int block, int d_ff,
                                 double rope_theta, int max_blocks) {
    std::map<std::string, mini::JsonValue> o;
    o["vocab_size"] = mini::JsonValue::make_num(static_cast<double>(vocab_size));
    o["n_layer"] = mini::JsonValue::make_num(static_cast<double>(n_layer));
    o["n_head"] = mini::JsonValue::make_num(static_cast<double>(n_head));
    o["d_model"] = mini::JsonValue::make_num(static_cast<double>(d_model));
    o["block"] = mini::JsonValue::make_num(static_cast<double>(block));
    o["d_ff"] = mini::JsonValue::make_num(static_cast<double>(d_ff));
    o["rope_theta"] = mini::JsonValue::make_num(rope_theta);
    o["tie_embeddings"] = mini::JsonValue::make_bool(true);
    o["use_pool"] = mini::JsonValue::make_num(0.0);
    o["pool_experts"] = mini::JsonValue::make_num(0.0);
    o["pool_d_ff"] = mini::JsonValue::make_num(0.0);
    o["pool_depth"] = mini::JsonValue::make_num(1.0);
    o["pool_top_k"] = mini::JsonValue::make_num(0.0);
    o["pool_capacity_factor"] = mini::JsonValue::make_num(1.5);
    o["pool_max"] = mini::JsonValue::make_num(0.0);
    o["pool_aux"] = mini::JsonValue::make_num(0.01);
    o["pool_resident"] = mini::JsonValue::make_num(0.0);
    o["n_prelude"] = mini::JsonValue::make_num(1.0);
    // block 0 is the prelude; the rest are recur sites, so recur count is
    // max_blocks - 1 (not max_blocks -- that counted a phantom recur site).
    o["n_recur"] = mini::JsonValue::make_num(static_cast<double>(max_blocks - 1));
    o["n_coda"] = mini::JsonValue::make_num(0.0);
    o["max_steps"] = mini::JsonValue::make_num(1.0);
    o["min_steps"] = mini::JsonValue::make_num(1.0);
    o["train_steps_mean"] = mini::JsonValue::make_num(0.0);
    o["bptt_window"] = mini::JsonValue::make_num(4.0);
    o["ponder_beta"] = mini::JsonValue::make_num(0.01);
    o["halt_prior"] = mini::JsonValue::make_num(0.4);
    o["halt_thresh"] = mini::JsonValue::make_num(0.5);
    return mini::JsonValue::make_obj(o);
}

int main(int argc, char** argv) {
    using clk = std::chrono::steady_clock;
    const auto start_tp = clk::now();
    auto elapsed = [&start_tp]() {
        return static_cast<double>(
                   std::chrono::duration_cast<std::chrono::milliseconds>(clk::now() -
                                                                        start_tp)
                       .count()) /
               1000.0;
    };

    std::string input, out_dir;
    std::cout.setf(std::ios::unitbuf);  // unbuffered: live progress in logs
    int max_blocks = 2;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](const char* flag) -> std::string {
            if (i + 1 >= argc) die(std::string("missing value for ") + flag);
            return argv[++i];
        };
        if (a == "--input") input = need("--input");
        else if (a == "--output_dir") out_dir = need("--output_dir");
        else if (a == "--max_blocks") {
            std::string v = need("--max_blocks");
            max_blocks = std::atoi(v.c_str());
        } else die("unknown argument '" + a + "'");
    }
    if (input.empty()) die("--input <gguf> is required");
    if (out_dir.empty()) die("--output_dir <dir> is required");
    if (max_blocks < 1) die("--max_blocks must be >= 1");

    std::ifstream f(input, std::ios::binary);
    if (!f.is_open()) die("cannot open '" + input + "'");
    f.seekg(0, std::ios::end);
    const uint64_t file_size = static_cast<uint64_t>(f.tellg());
    f.seekg(0);

    if (rd_u32(f) != 0x46554747u) die("not a GGUF file");
    const uint32_t version = rd_u32(f);
    if (version != 2 && version != 3) die("unsupported GGUF version " + std::to_string(version));
    const uint64_t tensor_count = rd_u64(f);
    const uint64_t kv_count = rd_u64(f);

    std::map<std::string, double> nums;
    std::map<std::string, uint64_t> arr_counts;
    std::map<std::string, std::string> strs;
    std::vector<std::string> tokens;
    std::vector<int32_t> token_types;

    int depth = 0;
    // read a single GGUF value (recursively for arrays)
    auto read_value = [&](uint32_t atype, const std::string& key,
                          auto&& self_ref) -> void {
        switch (atype) {
            case VAL_U8: nums[key] = static_cast<double>(rd_u8(f)); break;
            case VAL_I8: {
                int8_t v;
                f.read(reinterpret_cast<char*>(&v), 1);
                nums[key] = static_cast<double>(v);
                break;
            }
            case VAL_U16: nums[key] = static_cast<double>(rd_u16(f)); break;
            case VAL_I16: {
                int16_t v;
                f.read(reinterpret_cast<char*>(&v), 2);
                nums[key] = static_cast<double>(v);
                break;
            }
            case VAL_U32: nums[key] = static_cast<double>(rd_u32(f)); break;
            case VAL_I32: {
                int32_t v;
                f.read(reinterpret_cast<char*>(&v), 4);
                nums[key] = static_cast<double>(v);
                break;
            }
            case VAL_F32: {
                float v;
                f.read(reinterpret_cast<char*>(&v), 4);
                nums[key] = static_cast<double>(v);
                break;
            }
            case VAL_BOOL: nums[key] = static_cast<double>(rd_u8(f)); break;
            case VAL_U64: nums[key] = static_cast<double>(rd_u64(f)); break;
            case VAL_I64: {
                int64_t v;
                f.read(reinterpret_cast<char*>(&v), 8);
                nums[key] = static_cast<double>(v);
                break;
            }
            case VAL_F64: {
                double v;
                f.read(reinterpret_cast<char*>(&v), 8);
                nums[key] = v;
                break;
            }
            case VAL_STRING: strs[key] = rd_str(f); break;
            case VAL_ARRAY: {
                const uint32_t etype = rd_u32(f);
                const uint64_t n = rd_u64(f);
                arr_counts[key] = n;
                if (key == "tokenizer.ggml.tokens" && etype == VAL_STRING) {
                    tokens.reserve(static_cast<size_t>(n));
                    for (uint64_t i = 0; i < n; ++i) tokens.push_back(rd_str(f));
                } else if (key == "tokenizer.ggml.token_type" && etype == VAL_I32) {
                    token_types.reserve(static_cast<size_t>(n));
                    for (uint64_t i = 0; i < n; ++i) {
                        int32_t v;
                        f.read(reinterpret_cast<char*>(&v), 4);
                        token_types.push_back(v);
                    }
                } else {
                    for (uint64_t i = 0; i < n; ++i) {
                        if (depth > 4) die("gguf kv array too deeply nested");
                        ++depth;
                        self_ref(etype, key, self_ref);
                        --depth;
                    }
                }
                break;
            }
            default:
                die("unsupported GGUF KV value type " + std::to_string(atype));
        }
    };

    for (uint64_t i = 0; i < kv_count; ++i) {
        std::string key = rd_str(f);
        uint32_t vtype = rd_u32(f);
        read_value(vtype, key, read_value);
    }

    std::vector<GgufTensor> tensors;
    tensors.reserve(static_cast<size_t>(tensor_count));
    for (uint64_t i = 0; i < tensor_count; ++i) {
        GgufTensor t;
        t.name = rd_str(f);
        if (t.name.empty()) die("empty tensor name in GGUF info");
        uint32_t nd = rd_u32(f);
        if (nd == 0 || nd > 4) die("implausible n_dims=" + std::to_string(nd));
        t.dims.resize(nd);
        for (uint32_t k = 0; k < nd; ++k) t.dims[k] = rd_u64(f);
        t.type = rd_u32(f);
        t.offset = rd_u64(f);
        tensors.push_back(std::move(t));
    }
    if (!f.good()) die("truncated GGUF header");

    // base = file_size - (max tensor end). Empirically this equals the
    // aligned data offset that llama.cpp uses for the first tensor.
    uint64_t max_end = 0;
    for (const auto& t : tensors) {
        max_end = std::max(max_end, t.offset + tensor_bytes(t));
    }
    if (max_end == 0 || max_end > file_size) die("implausible GGUF tensor area");
    const uint64_t base = file_size - max_end;
    for (const auto& t : tensors) {
        if (base + t.offset + tensor_bytes(t) > file_size) {
            die("tensor '" + t.name + "' extends past end of file");
        }
    }

    auto find = [&](const std::string& name) -> const GgufTensor* {
        for (const auto& t : tensors) if (t.name == name) return &t;
        return nullptr;
    };

    // ---- derive model config from the KV metadata ----
    auto num = [&](const char* key, double fallback) -> double {
        auto it = nums.find(key);
        return it == nums.end() ? fallback : it->second;
    };
    const int d_model = static_cast<int>(num("qwen3.embedding_length", 0));
    const int n_layer = static_cast<int>(num("qwen3.block_count", 0));
    const int n_head = static_cast<int>(num("qwen3.attention.head_count", 0));
    const int n_kv = static_cast<int>(num("qwen3.attention.head_count_kv", 8));
    const int d_ff = static_cast<int>(num("qwen3.feed_forward_length", 4 * d_model));
    const int ctx = static_cast<int>(num("qwen3.context_length", 4096));
    double rope_theta = num("qwen3.rope.freq_base", 1000000.0);
    const double eps_v = num("qwen3.layer_norm_rms_epsilon", 0.0);
    const double eps = eps_v > 0.0 ? eps_v : 1e-6;
    (void)eps;
    if (d_model <= 0 || n_layer <= 0 || n_head <= 0 || d_model % n_head != 0)
        die("missing/implausible model dims from GGUF metadata");
    int vocab_size = static_cast<int>(num("qwen3.vocab_size", static_cast<double>(tokens.size())));
    if (vocab_size <= 0) vocab_size = static_cast<int>(tokens.size());
    const int hd = d_model / n_head;
    if (static_cast<int64_t>(n_head) * hd != d_model) die("bad n_head/d_model");
    if (n_kv <= 0 || n_head % n_kv != 0) die("bad kv head grouping");
    const int rep = n_head / n_kv;
    if (static_cast<int64_t>(n_kv) * hd * d_model == 0) die("bad shapes");

    // output.weight (tied head) is skipped: identical to token_embd.weight.
    if (find("output.weight")) {
        std::cout << "gguf_convert: skipping tied output.weight\n";
    }

    // build the state dict (core) and the router extras (routers.npz) ----
    std::vector<std::pair<std::string, mt::Tensor>> sd;
    std::vector<std::pair<std::string, mt::Tensor>> router_sd;
    auto add = [&sd](const std::string& key, mt::Tensor&& t) {
        sd.push_back({key, std::move(t)});
    };

    const GgufTensor* tok = find("token_embd.weight");
    require2(tok, static_cast<uint64_t>(d_model), static_cast<uint64_t>(vocab_size),
             "token_embd.weight");
    {
        std::vector<float> w = dequant(f, base, *tok, "token embeddings");
        std::cout << "gguf_convert: token_embd.weight -> tok_emb.weight"
                  << " (" << vocab_size << " x " << d_model << ")"
                  << " [" << elapsed() << "s]\n";
        add("tok_emb.weight", make_tensor(std::move(w), vocab_size, d_model));
    }

    const GgufTensor* lnf = find("output_norm.weight");
    require1(lnf, static_cast<uint64_t>(d_model), "output_norm.weight");
    {
        std::vector<float> w = dequant(f, base, *lnf, "final norm");
        add("ln_f.weight", make_tensor(std::move(w), d_model));
    }

    for (int b = 0; b < max_blocks; ++b) {
        const int gguf_blk = b;
        const std::string src = "blk." + std::to_string(gguf_blk) + ".";
        const std::string dst = (b == 0) ? "prelude.0" : ("recur." + std::to_string(b - 1));

        const GgufTensor* ln1 = find(src + "attn_norm.weight");
        require1(ln1, static_cast<uint64_t>(d_model), "attn_norm.weight");
        const GgufTensor* q = find(src + "attn_q.weight");
        const GgufTensor* k = find(src + "attn_k.weight");
        const GgufTensor* v = find(src + "attn_v.weight");
        require2(q, static_cast<uint64_t>(d_model), static_cast<uint64_t>(d_model),
                 "attn_q.weight");
        require2(k, static_cast<uint64_t>(d_model), static_cast<uint64_t>(n_kv * hd),
                 "attn_k.weight");
        require2(v, static_cast<uint64_t>(d_model), static_cast<uint64_t>(n_kv * hd),
                 "attn_v.weight");
        const GgufTensor* proj = find(src + "attn_output.weight");
        require2(proj, static_cast<uint64_t>(d_model), static_cast<uint64_t>(d_model),
                 "attn_output.weight");
        const GgufTensor* ln2 = find(src + "ffn_norm.weight");
        require1(ln2, static_cast<uint64_t>(d_model), "ffn_norm.weight");
        const GgufTensor* w1 = find(src + "ffn_gate.weight");
        const GgufTensor* w3 = find(src + "ffn_up.weight");
        const GgufTensor* w2 = find(src + "ffn_down.weight");
        require2(w1, static_cast<uint64_t>(d_model), static_cast<uint64_t>(d_ff),
                 "ffn_gate.weight");
        require2(w3, static_cast<uint64_t>(d_model), static_cast<uint64_t>(d_ff),
                 "ffn_up.weight");
        require2(w2, static_cast<uint64_t>(d_ff), static_cast<uint64_t>(d_model),
                 "ffn_down.weight");

        std::cout << "gguf_convert: blk." << gguf_blk << " -> " << dst
                  << " [" << elapsed() << "s]\n";

        std::vector<float> ln1w = dequant(f, base, *ln1, "attn_norm");
        add(dst + ".ln1.weight", make_tensor(std::move(ln1w), d_model));

        // fused qkv
        std::vector<float> qw = dequant(f, base, *q, "attn_q");
        std::vector<float> kw = dequant(f, base, *k, "attn_k");
        std::vector<float> vw = dequant(f, base, *v, "attn_v");
        std::vector<float> qkv(static_cast<size_t>(3) * d_model * d_model, 0.0f);
        std::memcpy(qkv.data(), qw.data(),
                    static_cast<size_t>(d_model) * d_model * sizeof(float));
        for (int h = 0; h < n_head; ++h) {
            const int src_row = (h / rep) * hd;
            const int kdst = d_model + h * hd;
            const int vdst = 2 * d_model + h * hd;
            std::memcpy(qkv.data() + static_cast<size_t>(kdst) * d_model,
                        kw.data() + static_cast<size_t>(src_row) * d_model,
                        static_cast<size_t>(hd) * d_model * sizeof(float));
            std::memcpy(qkv.data() + static_cast<size_t>(vdst) * d_model,
                        vw.data() + static_cast<size_t>(src_row) * d_model,
                        static_cast<size_t>(hd) * d_model * sizeof(float));
        }
        add(dst + ".attn.qkv.weight",
            make_tensor(std::move(qkv), static_cast<int64_t>(3) * d_model, d_model));

        std::vector<float> pw = dequant(f, base, *proj, "attn_output");
        add(dst + ".attn.proj.weight", make_tensor(std::move(pw), d_model, d_model));

        std::vector<float> ln2w = dequant(f, base, *ln2, "ffn_norm");
        add(dst + ".ln2.weight", make_tensor(std::move(ln2w), d_model));

        std::vector<float> w1w = dequant(f, base, *w1, "ffn_gate");
        add(dst + ".mlp.w1.weight", make_tensor(std::move(w1w), d_ff, d_model));
        std::vector<float> w3w = dequant(f, base, *w3, "ffn_up");
        add(dst + ".mlp.w3.weight", make_tensor(std::move(w3w), d_ff, d_model));
        std::vector<float> w2w = dequant(f, base, *w2, "ffn_down");
        add(dst + ".mlp.w2.weight", make_tensor(std::move(w2w), d_model, d_ff));
    }

    // adapter: [d, 2d] identity => cat([h, x]) mixes both inputs 1:1.
    {
        std::vector<float> ad(static_cast<size_t>(d_model) * 2 * d_model, 0.0f);
        for (int r = 0; r < d_model; ++r) {
            ad[static_cast<size_t>(r) * (2 * d_model) + r] = 1.0f;
            ad[static_cast<size_t>(r) * (2 * d_model) + d_model + r] = 1.0f;
        }
        add("adapter.weight",
            make_tensor(std::move(ad), d_model, static_cast<int64_t>(2) * d_model));
    }
    // halt: never halts mid-sequence (max_steps==1 forces halting at the end).
    // load_weights expects rank-2 [1, d] (trained-checkpoint layout).
    {
        std::vector<float> hw(static_cast<size_t>(d_model), 0.0f);
        add("halt.weight", make_tensor(std::move(hw), 1, d_model));
        std::vector<float> hb(1, 0.0f);
        add("halt.bias", make_tensor(std::move(hb), 1));
    }
    // pool metadata keys: inference never needs them, but they must NOT be
    // flagged as routers (store::is_router matches "router.weight"). Store
    // them in routers.npz under harmless names (kept out of core.npz).
    {
        mt::Tensor gg;
        gg.dtype = mt::DType::FP32;
        gg.shape.rank = 1;
        gg.shape.d[0] = 0;
        router_sd.push_back({"pool.gate.meta", std::move(gg)});
        mt::Shape s;
        s.rank = 2;
        s.d[0] = 0;
        s.d[1] = d_model;
        mt::Tensor t0;
        t0.dtype = mt::DType::FP32;
        t0.shape = s;
        router_sd.push_back({"pool.segment_router.meta", std::move(t0)});
    }

    // ---- PagedPool (0 experts; harmless) + save ----
    {
        mt::Shape onesh;
        onesh.rank = 1;
        onesh.d[0] = 0;
        mt::Tensor ones = mt::make(onesh, mt::DType::FP32, 1.0f);
        minagi::paged::PagedPool pool(0, d_model, 0, 0, 4, 0.15, 0.10, 4, true, ones);

        mini::JsonValue cfg_obj = typed_cfg(vocab_size, n_layer, n_head, d_model,
                                            ctx, d_ff, rope_theta, max_blocks);
        store::SaveResult res;
        if (!store::save_paged(sd, pool, cfg_obj, out_dir, 0, 0.0, &res,
                               &router_sd)) {
            die("save_paged failed for '" + out_dir + "'");
        }
        std::cout << "gguf_convert: wrote " << out_dir << " ("
                  << res.total_bytes << " bytes, " << max_blocks
                  << " blocks, " << tokens.size() << " vocabs)\n";
    }
    return 0;
}