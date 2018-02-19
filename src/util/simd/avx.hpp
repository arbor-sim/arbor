#pragma once

// AVX/AVX2 SIMD intrinsics implementation.

#ifdef __AVX__

#include <cstdint>
#include <immintrin.h>

namespace arb {
namespace simd_detail {

struct avx_double4 {
    using scalar_type = double;
    using vector_type = __m256d;

    // Masks use the same representation:
    using mask_impl = avx_double4;
    using mask_type = typename mask_impl::vector_type;

    using int64 = std::int64_t;

    static constexpr int cmp_gt_oq = 30; // _CMP_GT_OQ
    static constexpr int cmp_lt_oq = 17; // _CMP_LT_OQ
    static constexpr int cmp_eq_oq =  0; // _CMP_EQ_OQ
    static constexpr int cmp_neq_uq = 4; // _CMP_NEQ_UQ

    constexpr static unsigned width = 4;

    static vector_type broadcast(double v) {
        return _mm256_set1_pd(v);
    }

    static vector_type broadcast(bool b) {
        return _mm256_castsi256_pd(_mm256_set1_epi64x((int64)-b));
    }

    static vector_type immediate(double v0, double v1, double v2, double v3) {
        return _mm256_setr_pd(v0, v1, v2, v3);
    }

    static vector_type immediate(bool b0, bool b1, bool b2, bool b3) {
        return _mm256_castsi256_pd(_mm256_setr_epi64x(-b0, -b1, -b2, -b3));
    }

    static void copy_to(const vector_type& v, scalar_type* p) {
        _mm256_storeu_pd(p, v);
    }

    static vector_type copy_from(const scalar_type* p) {
        return _mm256_loadu_pd(p);
    }

    static vector_type add(const vector_type& a, const vector_type& b) {
        return _mm256_add_pd(a, b);
    }

    static vector_type sub(const vector_type& a, const vector_type& b) {
        return _mm256_sub_pd(a, b);
    }

    static vector_type mul(const vector_type& a, const vector_type& b) {
        return _mm256_mul_pd(a, b);
    }

    static vector_type div(const vector_type& a, const vector_type& b) {
        return _mm256_div_pd(a, b);
    }

    static vector_type fma(const vector_type& a, const vector_type& b, const vector_type& c) {
        return _mm256_add_pd(_mm256_mul_pd(a, b), c);
    }

    static vector_type logical_not(const vector_type& a) {
        return _mm256_castsi256_pd(~_mm256_castpd_si256(a));
    }

    static vector_type logical_and(const vector_type& a, const vector_type& b) {
        return _mm256_and_pd(a, b);
    }

    static vector_type logical_or(const vector_type& a, const vector_type& b) {
        return _mm256_or_pd(a, b);
    }

    static mask_type cmp_eq(const vector_type& a, const vector_type& b) {
        return _mm256_cmp_pd(a, b, cmp_eq_oq);
    }

    static mask_type cmp_not_eq(const vector_type& a, const vector_type& b) {
        return _mm256_cmp_pd(a, b, cmp_neq_uq);
    }

    static vector_type select(const mask_type& m, const vector_type& u, const vector_type& v) {
        return _mm256_blendv_pd(u, v, m);
    }

    static scalar_type element(const vector_type& u, int i) {
        return u[i];
    }

    static void set_element(vector_type& u, int i, scalar_type x) {
        u[i] = x;
    }

    static scalar_type bool_element(const vector_type& u, int i) {
        return u[i];
    }

    static void set_element(vector_type& u, int i, bool b) {
        __m256i ui = _mm256_castpd_si256(u);
        ui[i] = -b;
        u = _mm256_castsi256_pd(ui);
    }
};

} // namespace simd_detail

namespace simd_abi {

    template <typename T, unsigned N> struct avx;
    template <> struct avx<double, 4> { using type = simd_detail::avx_double4; };

} // namespace simd_abi;


// AVX2 extends AVX operations, with the same data representation.

#ifdef __AVX2__

namespace simd_detail {

struct avx2_double4: avx_double4 {
    // Masks use the same representation:
    using mask_impl = avx2_double4;
    using mask_type = typename mask_impl::vector_type;

    static vector_type fma(const vector_type& a, const vector_type& b, const vector_type& c) {
        return _mm256_fmadd_pd(a, b, c);
    }
};

} // namespace simd_detail

namespace simd_abi {

    template <typename T, unsigned N> struct avx2;
    template <> struct avx2<double, 4> { using type = simd_detail::avx2_double4; };

} // namespace simd_abi;

#endif // def __AVX2__

} // namespace arb

#endif // def __AVX__
