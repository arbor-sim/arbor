#pragma once

// AVX512F SIMD intrinsics implementation.

#ifdef __AVX512F__

#include <cmath>
#include <cstdint>
#include <immintrin.h>

#include <util/simd/approx.hpp>

namespace arb {
namespace simd_detail {

struct avx512_double8;
struct avx512_mask8;

template <>
struct simd_traits<avx512_mask8> {
    static constexpr unsigned width = 8;
    using scalar_type = bool;
    using vector_type = __mmask8;
    using mask_impl = avx512_mask8;
};

template <>
struct simd_traits<avx512_double8> {
    static constexpr unsigned width = 8;
    using scalar_type = double;
    using vector_type = __m512d;
    using mask_impl = avx512_mask8;
};

struct avx512_mask8: implbase<avx512_mask8> {
    static __mmask8 broadcast(bool b) {
        return _mm512_int2mask(-b);
    }

    static void copy_to(const __mmask8& k, bool* b) {
        __m256i a = _mm256_setzero_si256();
        a = _mm512_castsi512_si256(_mm512_mask_set1_epi32(_mm512_castsi256_si512(a), k, 1));

        __m256i s = _mm256_set1_epi32(0x0c080400);
        a = _mm256_shuffle_epi8(a, s);

        s = _mm256_setr_epi32(0, 4, 0, 0, 0, 0, 0, 0);
        a = _mm256_permutevar8x32_epi32(a, s);

        std::memcpy(b, &a, 8);
    }

    static __mmask8 copy_from(const bool* p) {
        __m256i a = _mm256_setzero_si256();
        std::memcpy(&a, p, 8);
        a = _mm256_sub_epi8(_mm256_setzero_si256(), a);
        return _mm512_int2mask(_mm256_movemask_epi8(a));
    }

    static __mmask8 logical_not(const __mmask8& k) {
        return _mm512_knot(k);
    }

    static __mmask8 logical_and(const __mmask8& a, const __mmask8& b) {
        return _mm512_kand(a, b);
    }

    static __mmask8 logical_or(const __mmask8& a, const __mmask8& b) {
        return _mm512_kor(a, b);
    }

    // Arithmetic operations not necessarily appropriate for
    // packed bit mask, but implemented for completeness/testing,
    // with Z modulo 2 semantics:
    //     a + b   is equivalent to   a ^ b
    //     a * b                      a & b
    //     a / b                      a
    //     a - b                      a ^ b
    //     -a                         a
    //     max(a, b)                  a | b
    //     min(a, b)                  a & b

    static __mmask8 negate(const __mmask8& a) {
        return a;
    }

    static __mmask8 add(const __mmask8& a, const __mmask8& b) {
        return _mm512_kxor(a, b);
    }

    static __mmask8 sub(const __mmask8& a, const __mmask8& b) {
        return _mm512_kxor(a, b);
    }

    static __mmask8 mul(const __mmask8& a, const __mmask8& b) {
        return _mm512_kand(a, b);
    }

    static __mmask8 div(const __mmask8& a, const __mmask8& b) {
        return a;
    }

    static __mmask8 fma(const __mmask8& a, const __mmask8& b, const __mmask8& c) {
        return add(mul(a, b), c);
    }

    static __mmask8 max(const __mmask8& a, const __mmask8& b) {
        return _mm512_kor(a, b);
    }

    static __mmask8 min(const __mmask8& a, const __mmask8& b) {
        return _mm512_kand(a, b);
    }

    // Comparison operators are also taken as operating on Z modulo 2,
    // with 1 > 0:
    //
    //     a > b    is equivalent to  a & ~b
    //     a >= b                     a | ~b,  ~(~a & b)
    //     a < b                      ~a & b
    //     a <= b                     ~a | b,  ~(a & ~b)
    //     a == b                     ~(a ^ b)
    //     a != b                     a ^ b

    static __mmask8 cmp_eq(const __mmask8& a, const __mmask8& b) {
        return _mm512_kxnor(a, b);
    }

    static __mmask8 cmp_neq(const __mmask8& a, const __mmask8& b) {
        return _mm512_kxor(a, b);
    }

    static __mmask8 cmp_lt(const __mmask8& a, const __mmask8& b) {
        return _mm512_kandn(a, b);
    }

    static __mmask8 cmp_gt(const __mmask8& a, const __mmask8& b) {
        return cmp_lt(b, a);
    }

    static __mmask8 cmp_geq(const __mmask8& a, const __mmask8& b) {
        return logical_not(cmp_lt(a, b));
    }

    static __mmask8 cmp_leq(const __mmask8& a, const __mmask8& b) {
        return logical_not(cmp_gt(a, b));
    }

    static __mmask8 ifelse(const __mmask8& m, const __mmask8& u, const __mmask8& v) {
        return _mm512_kor(_mm512_kandn(m, u), _mm512_kand(m, v));
    }

    static bool element(const __mmask8& k, int i) {
        return _mm512_mask2int(k)&(1<<i);
    }

    static void set_element(__mmask8& k, int i, bool b) {
        int n = _mm512_mask2int(k);
        k = _mm512_int2mask((n&~(1<<i))|(b<<i));
    }

    static __mmask8 mask_broadcast(bool b) {
        return broadcast(b);
    }

    static bool mask_element(const __mmask8& u, int i) {
        return element(u, i);
    }

    static void mask_set_element(__mmask8& u, int i, bool b) {
        set_element(u, i, b);
    }

    static void mask_copy_to(const __mmask8& m, bool* y) {
        copy_to(m, y);
    }

    static __mmask8 mask_copy_from(const bool* y) {
        return copy_from(y);
    }
};

struct avx512_double8: implbase<avx512_double8> {
    // Use default implementations for:
    //     element, set_element, fma.

    using int64 = std::int64_t;

    // CMPPD predicates:
    static constexpr int cmp_eq_oq =    0;
    static constexpr int cmp_unord_q =  3;
    static constexpr int cmp_neq_uq =   4;
    static constexpr int cmp_true_uq = 15;
    static constexpr int cmp_lt_oq =   17;
    static constexpr int cmp_le_oq =   18;
    static constexpr int cmp_nge_uq =  25;
    static constexpr int cmp_ge_oq =   29;
    static constexpr int cmp_gt_oq =   30;

    static __m512d broadcast(double v) {
        return _mm512_set1_pd(v);
    }

    static void copy_to(const __m512d& v, double* p) {
        _mm512_storeu_pd(p, v);
    }

    static __m512d copy_from(const double* p) {
        return _mm512_loadu_pd(p);
    }

    static __m512d negate(const __m512d& a) {
        return _mm512_sub_pd(_mm512_setzero_pd(), a);
    }

    static __m512d add(const __m512d& a, const __m512d& b) {
        return _mm512_add_pd(a, b);
    }

    static __m512d sub(const __m512d& a, const __m512d& b) {
        return _mm512_sub_pd(a, b);
    }

    static __m512d mul(const __m512d& a, const __m512d& b) {
        return _mm512_mul_pd(a, b);
    }

    static __m512d div(const __m512d& a, const __m512d& b) {
        return _mm512_div_pd(a, b);
    }

    static __mmask8 cmp_eq(const __m512d& a, const __m512d& b) {
        return _mm512_cmp_pd_mask(a, b, cmp_eq_oq);
    }

    static __mmask8 cmp_neq(const __m512d& a, const __m512d& b) {
        return _mm512_cmp_pd_mask(a, b, cmp_neq_uq);
    }

    static __mmask8 cmp_gt(const __m512d& a, const __m512d& b) {
        return _mm512_cmp_pd_mask(a, b, cmp_gt_oq);
    }

    static __mmask8 cmp_geq(const __m512d& a, const __m512d& b) {
        return _mm512_cmp_pd_mask(a, b, cmp_ge_oq);
    }

    static __mmask8 cmp_lt(const __m512d& a, const __m512d& b) {
        return _mm512_cmp_pd_mask(a, b, cmp_lt_oq);
    }

    static __mmask8 cmp_leq(const __m512d& a, const __m512d& b) {
        return _mm512_cmp_pd_mask(a, b, cmp_le_oq);
    }

    static __m512d ifelse(const __mmask8& m, const __m512d& u, const __m512d& v) {
        return _mm512_mask_blend_pd(m, v, u);
    }

    static __m512d max(const __m512d& a, const __m512d& b) {
        return _mm512_max_pd(a, b);
    }

    static __m512d min(const __m512d& a, const __m512d& b) {
        return _mm512_min_pd(a, b);
    }

    static __m512d abs(const __m512d& x) {
        // (NB: _mm512_abs_pd() wrapper buggy on gcc, so use AND directly.)
        __m512i m = _mm512_set1_epi64(0x7fffffffffffffff);
        return _mm512_castsi512_pd(_mm512_and_epi64(_mm512_castpd_si512(x), m));
    }

    // Use SVML for exp and log if compiling with icpc, else use ratpoly
    // approximations.

#if defined(__INTEL_COMPILER)
    static  __m512d exp(const __m512d& x) {
        return _mm512_exp_pd(x);
    }

    static  __m512d expm1(const __m512d& x) {
        return _mm512_expm1_pd(x);
    }

    static __m512d log(const __m512d& x) {
        return _mm512_log_pd(x);
    }

    static __m512d pow(const __m512d& x, const __m512d& y) {
        return _mm512_pow_pd(x, y);
    }
#else

    // Refer to avx/avx2 code for details of the exponential and log
    // implementations.

    static  __m512d exp(const __m512d& x) {
        // Masks for exceptional cases.

        auto is_large = cmp_gt(x, broadcast(exp_maxarg));
        auto is_small = cmp_lt(x, broadcast(exp_minarg));

        // Compute n and g.

        auto n = _mm512_floor_pd(add(mul(broadcast(ln2inv), x), broadcast(0.5)));

        auto g = fma(n, broadcast(-ln2C1), x);
        g = fma(n, broadcast(-ln2C2), g);

        auto gg = mul(g, g);

        // Compute the g*P(g^2) and Q(g^2).

        auto odd = mul(g, horner(gg, P0exp, P1exp, P2exp));
        auto even = horner(gg, Q0exp, Q1exp, Q2exp, Q3exp);

        // Compute R(g)/R(-g) = 1 + 2*g*P(g^2) / (Q(g^2)-g*P(g^2))

        auto expg = fma(broadcast(2), div(odd, sub(even, odd)), broadcast(1));

        // Scale by 2^n, propogating NANs.

        auto result = _mm512_scalef_pd(expg, n);

        return
            ifelse(is_large, broadcast(HUGE_VAL),
            ifelse(is_small, broadcast(0),
                   result));
    }

    static  __m512d expm1(const __m512d& x) {
        auto is_large = cmp_gt(x, broadcast(exp_maxarg));
        auto is_small = cmp_lt(x, broadcast(exp_minarg));

        auto half = broadcast(0.5);
        auto one = broadcast(1.);

        auto n = _mm512_maskz_roundscale_round_pd(
                    cmp_gt(abs(x), half),
                    mul(broadcast(ln2inv), x),
                    0,
                    _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC);

        auto g = fma(n, broadcast(-ln2C1), x);
        g = fma(n, broadcast(-ln2C2), g);

        auto gg = mul(g, g);

        auto odd = mul(g, horner(gg, P0exp, P1exp, P2exp));
        auto even = horner(gg, Q0exp, Q1exp, Q2exp, Q3exp);

        // Compute R(g)/R(-g) -1 = 2*g*P(g^2) / (Q(g^2)-g*P(g^2))

        auto expgm1 = mul(broadcast(2), div(odd, sub(even, odd)));

        // Scale by 2^n, propogating NANs.

        auto result = add(sub(_mm512_scalef_pd(one, n), one), _mm512_scalef_pd(expgm1, n));

        return
            ifelse(is_large, broadcast(HUGE_VAL),
            ifelse(is_small, broadcast(-1),
                   result));
    }

    static __m512d log(const __m512d& x) {
        // Masks for exceptional cases.

        auto is_large = cmp_geq(x, broadcast(HUGE_VAL));
        auto is_small = cmp_lt(x, broadcast(log_minarg));
        is_small = avx512_mask8::logical_and(is_small, cmp_geq(x, broadcast(0)));

        __m512d g = _mm512_getexp_pd(x);
        __m512d u = _mm512_getmant_pd(x, _MM_MANT_NORM_1_2, _MM_MANT_SIGN_nan);

        __m512d one = broadcast(1.);
        __m512d half = broadcast(0.5);
        auto gtsqrt2 = cmp_geq(u, broadcast(sqrt2));
        g = ifelse(gtsqrt2, add(g, one), g);
        u = ifelse(gtsqrt2, mul(u, half), u);

        auto z = sub(u, one);
        auto pz = horner(z, P0log, P1log, P2log, P3log, P4log, P5log);
        auto qz = horner1(z, Q0log, Q1log, Q2log, Q3log, Q4log);

        auto z2 = mul(z, z);
        auto z3 = mul(z2, z);

        auto r = div(mul(z3, pz), qz);
        r = fma(g,  broadcast(ln2C4), r);
        r = fms(z2, half, r);
        r = sub(z, r);
        r = fma(g,  broadcast(ln2C3), r);

        // r is alrady NaN if x is NaN or negative, otherwise
        // return  +inf if x is +inf, or -inf if zero or (positive) denormal.

        return
            ifelse(is_large, broadcast(HUGE_VAL),
            ifelse(is_small, broadcast(-HUGE_VAL),
                r));
    }
#endif

protected:
    static inline __m512d horner1(__m512d x, double a0) {
        return add(x, broadcast(a0));
    }

    static inline __m512d horner(__m512d x, double a0) {
        return broadcast(a0);
    }

    template <typename... T>
    static __m512d horner(__m512d x, double a0, T... tail) {
        return fma(x, horner(x, tail...), broadcast(a0));
    }

    template <typename... T>
    static __m512d horner1(__m512d x, double a0, T... tail) {
        return fma(x, horner1(x, tail...), broadcast(a0));
    }

    static __m512d fms(const __m512d& a, const __m512d& b, const __m512d& c) {
        return _mm512_fmsub_pd(a, b, c);
    }
};

} // namespace simd_detail

namespace simd_abi {
    template <typename T, unsigned N> struct avx512;
    template <> struct avx512<double, 8> { using type = simd_detail::avx512_double8; };
} // namespace simd_abi;

} // namespace arb

#endif // def __AVX512F__
