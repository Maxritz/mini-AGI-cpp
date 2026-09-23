#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <utility>

namespace mt {

enum class DType : uint8_t { FP32, BF16 };

struct Shape {
    size_t rank = 0;
    std::array<int64_t, 4> d{};
    int64_t numel() const {
        int64_t n = 1;
        for (size_t i = 0; i < rank; ++i) {
            n *= d[i];
        }
        return n;
    }
    bool operator==(const Shape& other) const {
        if (rank != other.rank) return false;
        for (size_t i = 0; i < rank; ++i) {
            if (d[i] != other.d[i]) return false;
        }
        return true;
    }
};

struct bfloat16 {
    uint16_t bits = 0;
    static bfloat16 from_float(float f);
    float to_float() const;
};

struct Tensor {
    DType dtype = DType::FP32;
    Shape shape{};
    std::vector<uint8_t> data_;
    bool is_view_ = false;
    std::array<int64_t, 4> stride_{};

    size_t elem_bytes() const {
        return (dtype == DType::FP32) ? 4 : 2;
    }

    double at_flat(int64_t i) const;
    float atf(int64_t i) const;
    void set_flat(int64_t i, double v);

    template <class T>
    T* ptr();

    template <class T>
    const T* ptr() const;
};

Tensor make(const Shape& s, DType dt = DType::FP32, float fill = 0.0f);
Tensor make_like(const Tensor& t, float fill = 0.0f);
Tensor make_zeros(const Shape& s, DType dt);

Tensor to_bf16(const Tensor& t);
Tensor to_fp32(const Tensor& t);

Tensor add(const Tensor& a, const Tensor& b);
Tensor sub(const Tensor& a, const Tensor& b);
Tensor mul(const Tensor& a, const Tensor& b);
Tensor div(const Tensor& a, const Tensor& b);

Tensor add_s(const Tensor& a, float s);
Tensor mul_s(const Tensor& a, float s);

Tensor exp_(const Tensor& a);
Tensor sqrt_(const Tensor& a);
Tensor rsqrt_(const Tensor& a);
Tensor log_(const Tensor& a);

Tensor sum_all(const Tensor& a);
Tensor mean_all(const Tensor& a);
Tensor sum_axis(const Tensor& a, size_t axis);

Tensor softmax(const Tensor& a, size_t axis);
Tensor rms_norm(const Tensor& x, const Tensor& weight, float eps);
Tensor matmul(const Tensor& A, const Tensor& B);
Tensor matmul_bf16(const Tensor& A, const Tensor& B);
Tensor gelu(const Tensor& a);
Tensor rope(const Tensor& x, int64_t head_dim, const Tensor& freqs_cos, const Tensor& freqs_sin, int64_t pos_offset);

Tensor argtopk_flatten(const Tensor& scores, int k);
std::pair<Tensor, Tensor> topk(const Tensor& scores, int k);
Tensor row_gather_values(const Tensor& vals2d, const Tensor& idx2d_int64);

} // namespace mt
