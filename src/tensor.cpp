#include "tensor.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace mt {

// bfloat16 conversion with round-to-nearest-even
bfloat16 bfloat16::from_float(float f) {
    bfloat16 result;
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));

    uint32_t sign = (bits >> 31) & 0x1;
    uint32_t exp = (bits >> 23) & 0xFF;
    uint32_t mantissa = bits & 0x7FFFFF;

    if (exp == 0xFF) {
        // Inf or NaN
        if (mantissa == 0) {
            // Inf
            result.bits = static_cast<uint16_t>((sign << 15) | 0x7F80);
        } else {
            // NaN
            result.bits = 0x7FC0;
        }
        return result;
    }

    // Round 23-bit fraction to 7 bits, round-to-nearest-even.
    uint32_t frac = mantissa >> 16;
    uint32_t round_bit = (mantissa >> 15) & 0x1;
    uint32_t sticky = mantissa & 0x7FFF;
    if (round_bit && (sticky != 0 || (frac & 0x1))) {
        frac += 1;
        if (frac > 0x7F) {
            frac = 0;
            exp += 1;
            if (exp > 0xFF) {
                // Overflow to Inf
                result.bits = static_cast<uint16_t>((sign << 15) | 0x7F80);
                return result;
            }
        }
    }

    result.bits = static_cast<uint16_t>((sign << 15) | (exp << 7) | frac);
    return result;
}

float bfloat16::to_float() const {
    uint32_t sign = (bits >> 15) & 0x1;
    uint32_t exp = (bits >> 7) & 0xFF;
    uint32_t mantissa = bits & 0x7F;

    uint32_t result_bits = (sign << 31) | (exp << 23) | (mantissa << 16);
    float result;
    std::memcpy(&result, &result_bits, sizeof(result));
    return result;
}

// Tensor helpers
double Tensor::at_flat(int64_t i) const {
    if (dtype == DType::FP32) {
        float val;
        std::memcpy(&val, data_.data() + i * 4, sizeof(float));
        return static_cast<double>(val);
    } else {
        bfloat16 bf;
        std::memcpy(&bf, data_.data() + i * 2, sizeof(uint16_t));
        return static_cast<double>(bf.to_float());
    }
}

float Tensor::atf(int64_t i) const {
    return static_cast<float>(at_flat(i));
}

void Tensor::set_flat(int64_t i, double v) {
    if (dtype == DType::FP32) {
        float val = static_cast<float>(v);
        std::memcpy(data_.data() + i * 4, &val, sizeof(float));
    } else {
        bfloat16 bf = bfloat16::from_float(static_cast<float>(v));
        std::memcpy(data_.data() + i * 2, &bf.bits, sizeof(uint16_t));
    }
}

template <class T>
T* Tensor::ptr() {
    if constexpr (std::is_same_v<T, float>) {
        if (dtype != DType::FP32) return nullptr;
    } else if constexpr (std::is_same_v<T, bfloat16>) {
        if (dtype != DType::BF16) return nullptr;
    }
    return reinterpret_cast<T*>(data_.data());
}

template <class T>
const T* Tensor::ptr() const {
    if constexpr (std::is_same_v<T, float>) {
        if (dtype != DType::FP32) return nullptr;
    } else if constexpr (std::is_same_v<T, bfloat16>) {
        if (dtype != DType::BF16) return nullptr;
    }
    return reinterpret_cast<const T*>(data_.data());
}

// Explicit instantiations
template float* Tensor::ptr<float>();
template bfloat16* Tensor::ptr<bfloat16>();
template const float* Tensor::ptr<float>() const;
template const bfloat16* Tensor::ptr<bfloat16>() const;

// Factory functions
Tensor make(const Shape& s, DType dt, float fill) {
    Tensor t;
    t.dtype = dt;
    t.shape = s;
    size_t total_bytes = static_cast<size_t>(s.numel()) * ((dt == DType::FP32) ? 4 : 2);
    t.data_.resize(total_bytes);
    float fill_val = fill;
    if (dt == DType::FP32) {
        for (int64_t i = 0; i < s.numel(); ++i) {
            std::memcpy(t.data_.data() + i * 4, &fill_val, sizeof(float));
        }
    } else {
        bfloat16 bf = bfloat16::from_float(fill);
        for (int64_t i = 0; i < s.numel(); ++i) {
            std::memcpy(t.data_.data() + i * 2, &bf.bits, sizeof(uint16_t));
        }
    }
    return t;
}

Tensor make_like(const Tensor& t, float fill) {
    return make(t.shape, t.dtype, fill);
}

Tensor make_zeros(const Shape& s, DType dt) {
    return make(s, dt, 0.0f);
}

// Conversion functions
Tensor to_bf16(const Tensor& t) {
    Tensor result;
    result.dtype = DType::BF16;
    result.shape = t.shape;
    result.data_.resize(static_cast<size_t>(t.shape.numel()) * 2);
    for (int64_t i = 0; i < t.shape.numel(); ++i) {
        float val = t.atf(i);
        bfloat16 bf = bfloat16::from_float(val);
        std::memcpy(result.data_.data() + i * 2, &bf.bits, sizeof(uint16_t));
    }
    return result;
}

Tensor to_fp32(const Tensor& t) {
    Tensor result;
    result.dtype = DType::FP32;
    result.shape = t.shape;
    result.data_.resize(static_cast<size_t>(t.shape.numel()) * 4);
    for (int64_t i = 0; i < t.shape.numel(); ++i) {
        float val = t.atf(i);
        std::memcpy(result.data_.data() + i * 4, &val, sizeof(float));
    }
    return result;
}

// Elementwise operations
Tensor add(const Tensor& a, const Tensor& b) {
    Tensor result;
    result.dtype = DType::FP32;
    result.shape = a.shape;
    result.data_.resize(static_cast<size_t>(a.shape.numel()) * 4);
    for (int64_t i = 0; i < a.shape.numel(); ++i) {
        float val = static_cast<float>(a.at_flat(i) + b.at_flat(i));
        std::memcpy(result.data_.data() + i * 4, &val, sizeof(float));
    }
    return result;
}

Tensor sub(const Tensor& a, const Tensor& b) {
    Tensor result;
    result.dtype = DType::FP32;
    result.shape = a.shape;
    result.data_.resize(static_cast<size_t>(a.shape.numel()) * 4);
    for (int64_t i = 0; i < a.shape.numel(); ++i) {
        float val = static_cast<float>(a.at_flat(i) - b.at_flat(i));
        std::memcpy(result.data_.data() + i * 4, &val, sizeof(float));
    }
    return result;
}

Tensor mul(const Tensor& a, const Tensor& b) {
    Tensor result;
    result.dtype = DType::FP32;
    result.shape = a.shape;
    result.data_.resize(static_cast<size_t>(a.shape.numel()) * 4);
    for (int64_t i = 0; i < a.shape.numel(); ++i) {
        float val = static_cast<float>(a.at_flat(i) * b.at_flat(i));
        std::memcpy(result.data_.data() + i * 4, &val, sizeof(float));
    }
    return result;
}

Tensor div(const Tensor& a, const Tensor& b) {
    Tensor result;
    result.dtype = DType::FP32;
    result.shape = a.shape;
    result.data_.resize(static_cast<size_t>(a.shape.numel()) * 4);
    for (int64_t i = 0; i < a.shape.numel(); ++i) {
        float val = static_cast<float>(a.at_flat(i) / b.at_flat(i));
        std::memcpy(result.data_.data() + i * 4, &val, sizeof(float));
    }
    return result;
}

Tensor add_s(const Tensor& a, float s) {
    Tensor result;
    result.dtype = DType::FP32;
    result.shape = a.shape;
    result.data_.resize(static_cast<size_t>(a.shape.numel()) * 4);
    for (int64_t i = 0; i < a.shape.numel(); ++i) {
        float val = static_cast<float>(a.at_flat(i) + s);
        std::memcpy(result.data_.data() + i * 4, &val, sizeof(float));
    }
    return result;
}

Tensor mul_s(const Tensor& a, float s) {
    Tensor result;
    result.dtype = DType::FP32;
    result.shape = a.shape;
    result.data_.resize(static_cast<size_t>(a.shape.numel()) * 4);
    for (int64_t i = 0; i < a.shape.numel(); ++i) {
        float val = static_cast<float>(a.at_flat(i) * s);
        std::memcpy(result.data_.data() + i * 4, &val, sizeof(float));
    }
    return result;
}

// Unary operations
Tensor exp_(const Tensor& a) {
    Tensor result;
    result.dtype = DType::FP32;
    result.shape = a.shape;
    result.data_.resize(static_cast<size_t>(a.shape.numel()) * 4);
    for (int64_t i = 0; i < a.shape.numel(); ++i) {
        float val = static_cast<float>(std::exp(a.at_flat(i)));
        std::memcpy(result.data_.data() + i * 4, &val, sizeof(float));
    }
    return result;
}

Tensor sqrt_(const Tensor& a) {
    Tensor result;
    result.dtype = DType::FP32;
    result.shape = a.shape;
    result.data_.resize(static_cast<size_t>(a.shape.numel()) * 4);
    for (int64_t i = 0; i < a.shape.numel(); ++i) {
        float val = static_cast<float>(std::sqrt(a.at_flat(i)));
        std::memcpy(result.data_.data() + i * 4, &val, sizeof(float));
    }
    return result;
}

Tensor rsqrt_(const Tensor& a) {
    Tensor result;
    result.dtype = DType::FP32;
    result.shape = a.shape;
    result.data_.resize(static_cast<size_t>(a.shape.numel()) * 4);
    for (int64_t i = 0; i < a.shape.numel(); ++i) {
        float val = static_cast<float>(1.0 / std::sqrt(a.at_flat(i)));
        std::memcpy(result.data_.data() + i * 4, &val, sizeof(float));
    }
    return result;
}

Tensor log_(const Tensor& a) {
    Tensor result;
    result.dtype = DType::FP32;
    result.shape = a.shape;
    result.data_.resize(static_cast<size_t>(a.shape.numel()) * 4);
    for (int64_t i = 0; i < a.shape.numel(); ++i) {
        double val = a.at_flat(i);
        if (val < 1e-12) val = 1e-12;
        float fval = static_cast<float>(std::log(val));
        std::memcpy(result.data_.data() + i * 4, &fval, sizeof(float));
    }
    return result;
}

// Reduction operations
Tensor sum_all(const Tensor& a) {
    double sum = 0.0;
    for (int64_t i = 0; i < a.shape.numel(); ++i) {
        sum += a.at_flat(i);
    }
    Shape s;
    s.rank = 0;
    Tensor result;
    result.dtype = DType::FP32;
    result.shape = s;
    result.data_.resize(4);
    float val = static_cast<float>(sum);
    std::memcpy(result.data_.data(), &val, sizeof(float));
    return result;
}

Tensor mean_all(const Tensor& a) {
    double sum = 0.0;
    for (int64_t i = 0; i < a.shape.numel(); ++i) {
        sum += a.at_flat(i);
    }
    double mean = sum / static_cast<double>(a.shape.numel());
    Shape s;
    s.rank = 0;
    Tensor result;
    result.dtype = DType::FP32;
    result.shape = s;
    result.data_.resize(4);
    float val = static_cast<float>(mean);
    std::memcpy(result.data_.data(), &val, sizeof(float));
    return result;
}

Tensor sum_axis(const Tensor& a, size_t axis) {
    if (axis >= a.shape.rank) {
        return a;
    }
    int64_t dim = a.shape.d[axis];
    int64_t outer = 1;
    for (size_t i = 0; i < axis; ++i) {
        outer *= a.shape.d[i];
    }
    int64_t inner = 1;
    for (size_t i = axis + 1; i < a.shape.rank; ++i) {
        inner *= a.shape.d[i];
    }

    Shape result_shape;
    result_shape.rank = a.shape.rank - 1;
    size_t ri = 0;
    for (size_t i = 0; i < a.shape.rank; ++i) {
        if (i != axis) {
            result_shape.d[ri++] = a.shape.d[i];
        }
    }

    Tensor result;
    result.dtype = DType::FP32;
    result.shape = result_shape;
    result.data_.resize(static_cast<size_t>(result_shape.numel()) * 4);

    for (int64_t o = 0; o < outer; ++o) {
        for (int64_t inn = 0; inn < inner; ++inn) {
            double sum = 0.0;
            for (int64_t d = 0; d < dim; ++d) {
                int64_t idx = o * dim * inner + d * inner + inn;
                sum += a.at_flat(idx);
            }
            int64_t ridx = o * inner + inn;
            float val = static_cast<float>(sum);
            std::memcpy(result.data_.data() + ridx * 4, &val, sizeof(float));
        }
    }
    return result;
}

// Softmax
Tensor softmax(const Tensor& a, size_t axis) {
    if (axis >= a.shape.rank) {
        return a;
    }
    int64_t dim = a.shape.d[axis];
    int64_t outer = 1;
    for (size_t i = 0; i < axis; ++i) {
        outer *= a.shape.d[i];
    }
    int64_t inner = 1;
    for (size_t i = axis + 1; i < a.shape.rank; ++i) {
        inner *= a.shape.d[i];
    }

    Tensor result;
    result.dtype = DType::FP32;
    result.shape = a.shape;
    result.data_.resize(static_cast<size_t>(a.shape.numel()) * 4);

    for (int64_t o = 0; o < outer; ++o) {
        for (int64_t inn = 0; inn < inner; ++inn) {
            // Find max
            float max_val = -std::numeric_limits<float>::infinity();
            for (int64_t d = 0; d < dim; ++d) {
                int64_t idx = o * dim * inner + d * inner + inn;
                float val = a.atf(idx);
                if (val > max_val) max_val = val;
            }
            // Compute exp and sum
            double sum = 0.0;
            std::vector<float> exps(dim);
            for (int64_t d = 0; d < dim; ++d) {
                int64_t idx = o * dim * inner + d * inner + inn;
                float val = a.atf(idx);
                exps[d] = static_cast<float>(std::exp(static_cast<double>(val) - max_val));
                sum += exps[d];
            }
            // Normalize
            for (int64_t d = 0; d < dim; ++d) {
                int64_t idx = o * dim * inner + d * inner + inn;
                float val = static_cast<float>(exps[d] / sum);
                std::memcpy(result.data_.data() + idx * 4, &val, sizeof(float));
            }
        }
    }
    return result;
}

// RMSNorm
Tensor rms_norm(const Tensor& x, const Tensor& weight, float eps) {
    int64_t rows = x.shape.d[0];
    int64_t cols = x.shape.d[1];

    Tensor result;
    result.dtype = DType::FP32;
    result.shape = x.shape;
    result.data_.resize(static_cast<size_t>(x.shape.numel()) * 4);

    for (int64_t r = 0; r < rows; ++r) {
        double sum_sq = 0.0;
        for (int64_t c = 0; c < cols; ++c) {
            double val = x.at_flat(r * cols + c);
            sum_sq += val * val;
        }
        double mean_sq = sum_sq / static_cast<double>(cols);
        double scale = 1.0 / std::sqrt(mean_sq + static_cast<double>(eps));
        for (int64_t c = 0; c < cols; ++c) {
            double val = x.at_flat(r * cols + c);
            double w = weight.at_flat(c);
            float result_val = static_cast<float>(val * scale * w);
            std::memcpy(result.data_.data() + (r * cols + c) * 4, &result_val, sizeof(float));
        }
    }
    return result;
}

// Matmul
Tensor matmul(const Tensor& A, const Tensor& B) {
    int64_t M = A.shape.d[0];
    int64_t K = A.shape.d[1];
    int64_t N = B.shape.d[1];

    Shape result_shape;
    result_shape.rank = 2;
    result_shape.d[0] = M;
    result_shape.d[1] = N;

    Tensor result;
    result.dtype = DType::FP32;
    result.shape = result_shape;
    result.data_.resize(static_cast<size_t>(M * N) * 4);

    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            double sum = 0.0;
            for (int64_t k = 0; k < K; ++k) {
                sum += A.at_flat(i * K + k) * B.at_flat(k * N + j);
            }
            float val = static_cast<float>(sum);
            std::memcpy(result.data_.data() + (i * N + j) * 4, &val, sizeof(float));
        }
    }
    return result;
}

Tensor matmul_bf16(const Tensor& A, const Tensor& B) {
    int64_t M = A.shape.d[0];
    int64_t K = A.shape.d[1];
    int64_t N = B.shape.d[1];

    Shape result_shape;
    result_shape.rank = 2;
    result_shape.d[0] = M;
    result_shape.d[1] = N;

    Tensor result;
    result.dtype = DType::FP32;
    result.shape = result_shape;
    result.data_.resize(static_cast<size_t>(M * N) * 4);

    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            double sum = 0.0;
            for (int64_t k = 0; k < K; ++k) {
                sum += A.at_flat(i * K + k) * B.at_flat(k * N + j);
            }
            float val = static_cast<float>(sum);
            std::memcpy(result.data_.data() + (i * N + j) * 4, &val, sizeof(float));
        }
    }
    return result;
}

// GELU with tanh approximation
Tensor gelu(const Tensor& a) {
    Tensor result;
    result.dtype = DType::FP32;
    result.shape = a.shape;
    result.data_.resize(static_cast<size_t>(a.shape.numel()) * 4);

    const float sqrt_2_over_pi = static_cast<float>(std::sqrt(2.0 / 3.14159265358979323846));
    for (int64_t i = 0; i < a.shape.numel(); ++i) {
        double x = a.at_flat(i);
        double inner = sqrt_2_over_pi * (x + 0.044715 * x * x * x);
        double val = 0.5 * x * (1.0 + std::tanh(inner));
        float fval = static_cast<float>(val);
        std::memcpy(result.data_.data() + i * 4, &fval, sizeof(float));
    }
    return result;
}

// RoPE
Tensor rope(const Tensor& x, int64_t head_dim, const Tensor& freqs_cos, const Tensor& freqs_sin, int64_t pos_offset) {
    int64_t T = x.shape.d[0];
    int64_t D = x.shape.d[1];
    int64_t n_head = D / head_dim;

    Tensor result = x; // Copy

    for (int64_t t = 0; t < T; ++t) {
        for (int64_t h = 0; h < n_head; ++h) {
            for (int64_t i = 0; i < head_dim / 2; ++i) {
                int64_t d0 = h * head_dim + 2 * i;
                int64_t d1 = h * head_dim + 2 * i + 1;
                int64_t pos = t + pos_offset;

                double x0 = x.at_flat(t * D + d0);
                double x1 = x.at_flat(t * D + d1);
                double cos_val = freqs_cos.at_flat(pos);
                double sin_val = freqs_sin.at_flat(pos);

                double new_d0 = x0 * cos_val - x1 * sin_val;
                double new_d1 = x1 * cos_val + x0 * sin_val;

                result.set_flat(t * D + d0, new_d0);
                result.set_flat(t * D + d1, new_d1);
            }
        }
    }
    return result;
}

// TopK operations
Tensor argtopk_flatten(const Tensor& scores, int k) {
    int64_t nf = scores.shape.d[0];
    int64_t n_experts = scores.shape.d[1];

    Shape result_shape;
    result_shape.rank = 2;
    result_shape.d[0] = nf;
    result_shape.d[1] = k;

    Tensor result;
    result.dtype = DType::FP32;
    result.shape = result_shape;
    result.data_.resize(static_cast<size_t>(nf * k) * 4);

    for (int64_t r = 0; r < nf; ++r) {
        std::vector<std::pair<double, int64_t>> vals;
        for (int64_t c = 0; c < n_experts; ++c) {
            vals.emplace_back(scores.at_flat(r * n_experts + c), c);
        }

        // Partial sort for top-k
        std::partial_sort(vals.begin(), vals.begin() + k, vals.end(),
            [](const std::pair<double, int64_t>& a, const std::pair<double, int64_t>& b) {
                if (a.first != b.first) return a.first > b.first;
                return a.second < b.second;
            });

        for (int i = 0; i < k; ++i) {
            float val = static_cast<float>(vals[i].second);
            std::memcpy(result.data_.data() + (r * k + i) * 4, &val, sizeof(float));
        }
    }
    return result;
}

std::pair<Tensor, Tensor> topk(const Tensor& scores, int k) {
    int64_t nf = scores.shape.d[0];
    int64_t n_experts = scores.shape.d[1];

    Shape result_shape;
    result_shape.rank = 2;
    result_shape.d[0] = nf;
    result_shape.d[1] = k;

    Tensor values;
    values.dtype = DType::FP32;
    values.shape = result_shape;
    values.data_.resize(static_cast<size_t>(nf * k) * 4);

    Tensor indices;
    indices.dtype = DType::FP32;
    indices.shape = result_shape;
    indices.data_.resize(static_cast<size_t>(nf * k) * 4);

    for (int64_t r = 0; r < nf; ++r) {
        std::vector<std::pair<double, int64_t>> vals;
        for (int64_t c = 0; c < n_experts; ++c) {
            vals.emplace_back(scores.at_flat(r * n_experts + c), c);
        }

        std::partial_sort(vals.begin(), vals.begin() + k, vals.end(),
            [](const std::pair<double, int64_t>& a, const std::pair<double, int64_t>& b) {
                if (a.first != b.first) return a.first > b.first;
                return a.second < b.second;
            });

        for (int i = 0; i < k; ++i) {
            float v = static_cast<float>(vals[i].first);
            float idx = static_cast<float>(vals[i].second);
            std::memcpy(values.data_.data() + (r * k + i) * 4, &v, sizeof(float));
            std::memcpy(indices.data_.data() + (r * k + i) * 4, &idx, sizeof(float));
        }
    }
    return {values, indices};
}

Tensor row_gather_values(const Tensor& vals2d, const Tensor& idx2d_int64) {
    int64_t R = idx2d_int64.shape.d[0];
    int64_t C = idx2d_int64.shape.d[1];
    int64_t N = vals2d.shape.d[1];

    Shape result_shape;
    result_shape.rank = 2;
    result_shape.d[0] = R;
    result_shape.d[1] = C;

    Tensor result;
    result.dtype = DType::FP32;
    result.shape = result_shape;
    result.data_.resize(static_cast<size_t>(R * C) * 4);

    for (int64_t r = 0; r < R; ++r) {
        for (int64_t c = 0; c < C; ++c) {
            int64_t idx = static_cast<int64_t>(idx2d_int64.at_flat(r * C + c));
            double val = vals2d.at_flat(r * N + idx);
            float fval = static_cast<float>(val);
            std::memcpy(result.data_.data() + (r * C + c) * 4, &fval, sizeof(float));
        }
    }
    return result;
}

} // namespace mt
