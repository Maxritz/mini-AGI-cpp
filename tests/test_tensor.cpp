#include "tensor.hpp"

#include <cmath>
#include <cstdio>
#include <limits>

#ifndef M_PI
#  define M_PI 3.14159265358979323846
#endif

using namespace mt;

static bool approx_equal(double a, double b, double tol = 1e-4) {
    return std::fabs(a - b) <= tol * (1.0 + std::fabs(b));
}

int main() {
    int checks = 0;
    int passed = 0;

    auto check = [&](const char* name, bool cond) {
        checks++;
        if (cond) {
            passed++;
            std::printf("check %d PASS\n", checks);
        } else {
            std::printf("check %d FAIL\n", checks);
        }
    };

    // Check 1: bfloat16 conversion 1.0f -> 0x3f80
    {
        bfloat16 bf = bfloat16::from_float(1.0f);
        check("bf16 1.0f -> 0x3f80", bf.bits == 0x3f80);
    }

    // Check 2: bfloat16 conversion -2.0f -> 0xc000
    {
        bfloat16 bf = bfloat16::from_float(-2.0f);
        check("bf16 -2.0f -> 0xc000", bf.bits == 0xc000);
    }

    // Check 3: bfloat16 conversion 3.140625f -> 0x4049
    {
        bfloat16 bf = bfloat16::from_float(3.140625f);
        check("bf16 3.140625f -> 0x4049", bf.bits == 0x4049);
    }

    // Check 4: bfloat16 round-trip for 1e-3f
    {
        bfloat16 bf = bfloat16::from_float(1e-3f);
        float back = bf.to_float();
        check("bf16 1e-3f round-trip", approx_equal(back, 1e-3f, 0.01));
    }

    // Check 5: bfloat16 round-trip for 65504.0f
    {
        bfloat16 bf = bfloat16::from_float(65504.0f);
        float back = bf.to_float();
        check("bf16 65504.0f round-trip", approx_equal(back, 65504.0f, 0.005));
    }

    // Check 6: bfloat16 round-trip for 1e-10f
    {
        bfloat16 bf = bfloat16::from_float(1e-10f);
        float back = bf.to_float();
        check("bf16 1e-10f round-trip", approx_equal(back, 1e-10f, 0.005));
    }

    // Check 7: bfloat16 Inf handling
    {
        bfloat16 bf = bfloat16::from_float(std::numeric_limits<float>::infinity());
        check("bf16 Inf -> 0x7f80", bf.bits == 0x7f80);
    }

    // Check 8: bfloat16 NaN handling
    {
        bfloat16 bf = bfloat16::from_float(std::numeric_limits<float>::quiet_NaN());
        check("bf16 NaN -> 0x7fc0", bf.bits == 0x7fc0);
    }

    // Check 9: make + elementwise add
    {
        Shape s;
        s.rank = 1;
        s.d[0] = 3;
        Tensor a = make(s, DType::FP32, 1.0f);
        Tensor b = make(s, DType::FP32, 2.0f);
        Tensor c = add(a, b);
        check("make + add", approx_equal(c.at_flat(0), 3.0));
    }

    // Check 10: elementwise mul
    {
        Shape s;
        s.rank = 1;
        s.d[0] = 3;
        Tensor a = make(s, DType::FP32, 2.0f);
        Tensor b = make(s, DType::FP32, 3.0f);
        Tensor c = mul(a, b);
        check("mul", approx_equal(c.at_flat(0), 6.0));
    }

    // Check 11: softmax sums to 1.0 per row
    {
        Shape s;
        s.rank = 2;
        s.d[0] = 2;
        s.d[1] = 5;
        Tensor a = make(s, DType::FP32, 0.0f);
        float vals[] = {1, 2, 3, 4, 5, 5, 4, 3, 2, 1};
        for (int i = 0; i < 10; ++i) {
            a.set_flat(i, vals[i]);
        }
        Tensor sm = softmax(a, 1);
        double sum1 = 0, sum2 = 0;
        for (int i = 0; i < 5; ++i) {
            sum1 += sm.at_flat(i);
            sum2 += sm.at_flat(5 + i);
        }
        check("softmax row 1 sums to 1", approx_equal(sum1, 1.0, 1e-5));
        check("softmax row 2 sums to 1", approx_equal(sum2, 1.0, 1e-5));
    }

    // Check 12: rms_norm vs hand-computed
    {
        Shape s;
        s.rank = 2;
        s.d[0] = 1;
        s.d[1] = 3;
        Tensor x = make(s, DType::FP32, 0.0f);
        x.set_flat(0, 3.0);
        x.set_flat(1, 4.0);
        x.set_flat(2, 5.0);
        Tensor w = make(s, DType::FP32, 1.0f);
        Tensor y = rms_norm(x, w, 1e-6f);
        double mean_sq = (9.0 + 16.0 + 25.0) / 3.0;
        double scale = 1.0 / std::sqrt(mean_sq + 1e-6);
        check("rms_norm[0]", approx_equal(y.at_flat(0), 3.0 * scale, 1e-5));
        check("rms_norm[1]", approx_equal(y.at_flat(1), 4.0 * scale, 1e-5));
        check("rms_norm[2]", approx_equal(y.at_flat(2), 5.0 * scale, 1e-5));
    }

    // Check 13: gelu at 0 -> 0
    {
        Shape s;
        s.rank = 1;
        s.d[0] = 1;
        Tensor a = make(s, DType::FP32, 0.0f);
        Tensor g = gelu(a);
        check("gelu(0) ~ 0", approx_equal(g.at_flat(0), 0.0, 1e-6));
    }

    // Check 14: gelu(1) ~ 0.84119
    {
        Shape s;
        s.rank = 1;
        s.d[0] = 1;
        Tensor a = make(s, DType::FP32, 0.0f);
        a.set_flat(0, 1.0);
        Tensor g = gelu(a);
        check("gelu(1) ~ 0.84119", approx_equal(g.at_flat(0), 0.84119, 1e-4));
    }

    // Check 15: matmul identity
    {
        Shape s;
        s.rank = 2;
        s.d[0] = 3;
        s.d[1] = 3;
        Tensor I = make(s, DType::FP32, 0.0f);
        I.set_flat(0, 1.0);
        I.set_flat(4, 1.0);
        I.set_flat(8, 1.0);
        Shape sv;
        sv.rank = 2;
        sv.d[0] = 3;
        sv.d[1] = 1;
        Tensor v = make(sv, DType::FP32, 0.0f);
        v.set_flat(0, 1.0);
        v.set_flat(1, 2.0);
        v.set_flat(2, 3.0);
        Tensor r = matmul(I, v);
        check("matmul identity", approx_equal(r.at_flat(0), 1.0) &&
               approx_equal(r.at_flat(1), 2.0) &&
               approx_equal(r.at_flat(2), 3.0));
    }

    // Check 16: matmul 2x3 by 3x2 hand check
    {
        Shape sa;
        sa.rank = 2;
        sa.d[0] = 2;
        sa.d[1] = 3;
        Tensor A = make(sa, DType::FP32, 0.0f);
        float a_vals[] = {1, 2, 3, 4, 5, 6};
        for (int i = 0; i < 6; ++i) A.set_flat(i, a_vals[i]);

        Shape sb;
        sb.rank = 2;
        sb.d[0] = 3;
        sb.d[1] = 2;
        Tensor B = make(sb, DType::FP32, 0.0f);
        float b_vals[] = {7, 8, 9, 10, 11, 12};
        for (int i = 0; i < 6; ++i) B.set_flat(i, b_vals[i]);

        Tensor C = matmul(A, B);
        // Expected: [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
        //           [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
        check("matmul 2x3*3x2", approx_equal(C.at_flat(0), 58.0) &&
               approx_equal(C.at_flat(1), 64.0) &&
               approx_equal(C.at_flat(2), 139.0) &&
               approx_equal(C.at_flat(3), 154.0));
    }

    // Check 17: sum_all/mean_all of 1..6
    {
        Shape s;
        s.rank = 1;
        s.d[0] = 6;
        Tensor a = make(s, DType::FP32, 0.0f);
        for (int i = 0; i < 6; ++i) a.set_flat(i, static_cast<float>(i + 1));
        Tensor sa = sum_all(a);
        Tensor ma = mean_all(a);
        check("sum_all 1..6 = 21", approx_equal(sa.at_flat(0), 21.0));
        check("mean_all 1..6 = 3.5", approx_equal(ma.at_flat(0), 3.5));
    }

    // Check 18: topk per-row indices correctness
    {
        Shape s;
        s.rank = 2;
        s.d[0] = 2;
        s.d[1] = 4;
        Tensor scores = make(s, DType::FP32, 0.0f);
        float vals[] = {1, 3, 2, 4, 5, 6, 7, 8};
        for (int i = 0; i < 8; ++i) scores.set_flat(i, vals[i]);
        auto [values, indices] = topk(scores, 2);
        // Row 0: top-2 are 4 (idx 3) and 3 (idx 1)
        // Row 1: top-2 are 8 (idx 3) and 7 (idx 2)
        check("topk values row 0", approx_equal(values.at_flat(0), 4.0) &&
               approx_equal(values.at_flat(1), 3.0));
        check("topk indices row 0", approx_equal(indices.at_flat(0), 3.0) &&
               approx_equal(indices.at_flat(1), 1.0));
        check("topk values row 1", approx_equal(values.at_flat(2), 8.0) &&
               approx_equal(values.at_flat(3), 7.0));
        check("topk indices row 1", approx_equal(indices.at_flat(2), 3.0) &&
               approx_equal(indices.at_flat(3), 2.0));
    }

    // Check 19: argtopk returns ties-by-smaller-index
    {
        Shape s;
        s.rank = 2;
        s.d[0] = 1;
        s.d[1] = 4;
        Tensor scores = make(s, DType::FP32, 0.0f);
        float vals[] = {5, 5, 3, 3};
        for (int i = 0; i < 4; ++i) scores.set_flat(i, vals[i]);
        Tensor idx = argtopk_flatten(scores, 2);
        // Top-2 are both 5, ties broken by smaller index: 0, 1
        check("argtopk ties", approx_equal(idx.at_flat(0), 0.0) &&
               approx_equal(idx.at_flat(1), 1.0));
    }

    // Check 20: rope with theta=pi/2
    {
        Shape sx;
        sx.rank = 2;
        sx.d[0] = 1;
        sx.d[1] = 4;
        Tensor x = make(sx, DType::FP32, 0.0f);
        x.set_flat(0, 1.0); // d0 of head 0
        x.set_flat(1, 0.0); // d1 of head 0
        x.set_flat(2, 0.0); // d0 of head 1
        x.set_flat(3, 1.0); // d1 of head 1

        Shape sf;
        sf.rank = 1;
        sf.d[0] = 1;
        Tensor freqs_cos = make(sf, DType::FP32, 0.0f);
        Tensor freqs_sin = make(sf, DType::FP32, 0.0f);
        freqs_cos.set_flat(0, std::cos(M_PI / 2.0)); // ~0
        freqs_sin.set_flat(0, std::sin(M_PI / 2.0)); // ~1

        Tensor y = rope(x, 2, freqs_cos, freqs_sin, 0);
        // Pair 0: (1,0) -> (1*0 - 0*1, 0*0 + 1*1) = (0, 1)
        // Pair 1: (0,1) -> (0*0 - 1*1, 1*0 + 0*1) = (-1, 0)
        check("rope pair 0", approx_equal(y.at_flat(0), 0.0, 1e-5) &&
               approx_equal(y.at_flat(1), 1.0, 1e-5));
        check("rope pair 1", approx_equal(y.at_flat(2), -1.0, 1e-5) &&
               approx_equal(y.at_flat(3), 0.0, 1e-5));
    }

    // Check 21: matmul_bf16 close to fp32
    {
        Shape sa;
        sa.rank = 2;
        sa.d[0] = 2;
        sa.d[1] = 3;
        Tensor A_fp32 = make(sa, DType::FP32, 0.0f);
        float a_vals[] = {1.5f, 2.5f, 3.5f, 4.5f, 5.5f, 6.5f};
        for (int i = 0; i < 6; ++i) A_fp32.set_flat(i, a_vals[i]);

        Shape sb;
        sb.rank = 2;
        sb.d[0] = 3;
        sb.d[1] = 2;
        Tensor B_fp32 = make(sb, DType::FP32, 0.0f);
        float b_vals[] = {0.5f, 1.5f, 2.5f, 3.5f, 4.5f, 5.5f};
        for (int i = 0; i < 6; ++i) B_fp32.set_flat(i, b_vals[i]);

        Tensor C_fp32 = matmul(A_fp32, B_fp32);

        Tensor A_bf16 = to_bf16(A_fp32);
        Tensor B_bf16 = to_bf16(B_fp32);
        Tensor C_bf16 = matmul_bf16(A_bf16, B_bf16);

        bool close = true;
        for (int i = 0; i < 4; ++i) {
            double ref = C_fp32.at_flat(i);
            double bf = C_bf16.at_flat(i);
            if (std::fabs(ref) > 1e-6) {
                if (std::fabs(ref - bf) / std::fabs(ref) > 0.01) close = false;
            } else {
                if (std::fabs(ref - bf) > 0.01) close = false;
            }
        }
        check("matmul_bf16 close to fp32", close);
    }

    std::printf("\n%d/%d tests passed\n", passed, checks);
    if (passed == checks) {
        std::printf("tests PASSED\n");
        return 0;
    } else {
        std::printf("tests FAILED\n");
        return 1;
    }
}
