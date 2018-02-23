#pragma once

// AVX/AVX2 SIMD intrinsics implementation.

#ifdef __AVX__

#include <cstdint>
#include <immintrin.h>

namespace arb {
namespace simd_detail {

struct avx_int4;
struct avx_double4;

template <>
struct simd_traits<avx_int4> {
    static constexpr unsigned width = 4;
    using scalar_type = std::int32_t;
    using vector_type = __m128i;
    using mask_impl = avx_int4;
};

template <>
struct simd_traits<avx_double4> {
    static constexpr unsigned width = 4;
    using scalar_type = double;
    using vector_type = __m256d;
    using mask_impl = avx_double4;
};

struct avx_int4: implbase<avx_int4> {
    // Use default implementations for:
    //     element, set_element, fma, div.

    using int32 = std::int32_t;

    static __m128i broadcast(double v) {
        return _mm_set1_epi32(v);
    }

    static void copy_to(const __m128i& v, int32* p) {
        _mm_storeu_si128((__m128i*)p, v);
    }

    static __m128i copy_from(const int32* p) {
        return _mm_loadu_si128((const __m128i*)p);
    }

    static __m128i add(const __m128i& a, const __m128i& b) {
        return _mm_add_epi32(a, b);
    }

    static __m128i sub(const __m128i& a, const __m128i& b) {
        return _mm_sub_epi32(a, b);
    }

    static __m128i mul(const __m128i& a, const __m128i& b) {
        return _mm_mullo_epi32(a, b);
    }

    static __m128i fma(const __m128i& a, const __m128i& b, const __m128i& c) {
        return _mm_add_epi32(_mm_mullo_epi32(a, b), c);
    }

    static __m128i logical_not(const __m128i& a) {
        __m128i ones = {};
        return _mm_xor_si128(a, _mm_cmpeq_epi32(ones, ones));
    }

    static __m128i logical_and(const __m128i& a, const __m128i& b) {
        return _mm_and_si128(a, b);
    }

    static __m128i logical_or(const __m128i& a, const __m128i& b) {
        return _mm_or_si128(a, b);
    }

    static __m128i cmp_eq(const __m128i& a, const __m128i& b) {
        return _mm_cmpeq_epi32(a, b);
    }

    static __m128i cmp_neq(const __m128i& a, const __m128i& b) {
        return logical_not(cmp_eq(a, b));
    }

    static __m128i cmp_gt(const __m128i& a, const __m128i& b) {
        return _mm_cmpgt_epi32(a, b);
    }

    static __m128i cmp_geq(const __m128i& a, const __m128i& b) {
        return logical_not(cmp_gt(b, a));
    }

    static __m128i cmp_lt(const __m128i& a, const __m128i& b) {
        return cmp_gt(b, a);
    }

    static __m128i cmp_leq(const __m128i& a, const __m128i& b) {
        return logical_not(cmp_gt(a, b));
    }

    static __m128i mask_broadcast(bool b) {
        return _mm_set1_epi32(-(int32)b);
    }

    static bool mask_element(const __m128i& u, int i) {
        return static_cast<bool>(element(u, i));
    }

    static void mask_set_element(__m128i& u, int i, bool b) {
        set_element(u, i, -(int32)b);
    }

    static void mask_copy_to(const __m128i& m, bool* y) {
        __m128i s = _mm_setr_epi32(0x0c080400ul,0,0,0);
        __m128i p = _mm_shuffle_epi8(m, s);
        std::memcpy(y, &p, 4);
    }

    static __m128i mask_copy_from(const bool* w) {
        __m128i r;
        std::memcpy(&r, w, 4);

        __m128i s = _mm_setr_epi32(0x80808000ul, 0x80808001ul, 0x80808002ul, 0x80808003ul);
        return _mm_shuffle_epi8(r, s);
    }
};

struct avx_double4: implbase<avx_double4> {
    // Use default implementations for:
    //     element, set_element, fma.

    using int64 = std::int64_t;

    static constexpr int cmp_gt_oq = 30;  // _CMP_GT_OQ
    static constexpr int cmp_ge_oq = 29;  // _CMP_GE_OQ
    static constexpr int cmp_le_oq = 18;  // _CMP_LE_OQ
    static constexpr int cmp_lt_oq = 17;  // _CMP_LT_OQ
    static constexpr int cmp_eq_oq =  0;  // _CMP_EQ_OQ
    static constexpr int cmp_neq_uq = 4;  // _CMP_NEQ_UQ
    static constexpr int cmp_true_uq =15; // _CMP_TRUE_UQ

    static __m256d broadcast(double v) {
        return _mm256_set1_pd(v);
    }

    static void copy_to(const __m256d& v, double* p) {
        _mm256_storeu_pd(p, v);
    }

    static __m256d copy_from(const double* p) {
        return _mm256_loadu_pd(p);
    }

    static __m256d add(const __m256d& a, const __m256d& b) {
        return _mm256_add_pd(a, b);
    }

    static __m256d sub(const __m256d& a, const __m256d& b) {
        return _mm256_sub_pd(a, b);
    }

    static __m256d mul(const __m256d& a, const __m256d& b) {
        return _mm256_mul_pd(a, b);
    }

    static __m256d div(const __m256d& a, const __m256d& b) {
        return _mm256_div_pd(a, b);
    }

    static __m256d logical_not(const __m256d& a) {
        __m256d ones = {};
        return _mm256_xor_pd(a, _mm256_cmp_pd(ones, ones, 15));
    }

    static __m256d logical_and(const __m256d& a, const __m256d& b) {
        return _mm256_and_pd(a, b);
    }

    static __m256d logical_or(const __m256d& a, const __m256d& b) {
        return _mm256_or_pd(a, b);
    }

    static __m256d cmp_eq(const __m256d& a, const __m256d& b) {
        return _mm256_cmp_pd(a, b, cmp_eq_oq);
    }

    static __m256d cmp_neq(const __m256d& a, const __m256d& b) {
        return _mm256_cmp_pd(a, b, cmp_neq_uq);
    }

    static __m256d cmp_gt(const __m256d& a, const __m256d& b) {
        return _mm256_cmp_pd(a, b, cmp_gt_oq);
    }

    static __m256d cmp_geq(const __m256d& a, const __m256d& b) {
        return _mm256_cmp_pd(a, b, cmp_ge_oq);
    }

    static __m256d cmp_lt(const __m256d& a, const __m256d& b) {
        return _mm256_cmp_pd(a, b, cmp_lt_oq);
    }

    static __m256d cmp_leq(const __m256d& a, const __m256d& b) {
        return _mm256_cmp_pd(a, b, cmp_le_oq);
    }

    static __m256d select(const __m256d& m, const __m256d& u, const __m256d& v) {
        return _mm256_blendv_pd(u, v, m);
    }

    static __m256d mask_broadcast(bool b) {
        return _mm256_castsi256_pd(_mm256_set1_epi64x(-(int64)b));
    }

    static bool mask_element(const __m256d& u, int i) {
        return static_cast<bool>(element(u, i));
    }

    static void mask_set_element(__m256d& u, int i, bool b) {
        char data[256];
        _mm256_storeu_pd((double*)data, u);
        ((int64*)data)[i] = -(int64)b;
        u = _mm256_loadu_pd((double*)data);
    }

    static void mask_copy_to(const __m256d& m, bool* y) {
        __m128i zero = _mm_setzero_si128();

        // Split into upper and lower 128-bits (two mask values
        // in each), translate 0xffffffffffffffff to 0x0000000000000001.

        __m128i ml = _mm_castpd_si128(_mm256_castpd256_pd128(m));
        ml = _mm_sub_epi64(zero, ml);

        __m128i mu = _mm_castpd_si128(_mm256_castpd256_pd128(_mm256_permute2f128_pd(m, m, 1)));
        mu = _mm_sub_epi64(zero, mu);

        // Move bytes with bool value to bytes 0 and 1 in lower half,
        // bytes 2 and 3 in upper half, and merge with bitwise-or.

        __m128i sl = _mm_setr_epi32(0x80800800ul,0,0,0);
        ml = _mm_shuffle_epi8(ml, sl);

        __m128i su = _mm_setr_epi32(0x08008080ul,0,0,0);
        mu = _mm_shuffle_epi8(mu, su);

        __m128i r = _mm_or_si128(mu, ml);
        std::memcpy(y, &r, 4);
    }

    static __m256d mask_copy_from(const bool* w) {
        __m128i zero = _mm_setzero_si128();

        __m128i r;
        std::memcpy(&r, w, 4);

        // Move bytes:
        //   rl: byte 0 to byte 0, byte 1 to byte 8, zero elsewhere.
        //   ru: byte 2 to byte 0, byte 3 to byte 8, zero elsewhere.
        //
        // Subtract from zero to translate
        // 0x0000000000000001 to 0xffffffffffffffff.

        __m128i sl = _mm_setr_epi32(0x80808000ul, 0x80808080ul, 0x80808001ul, 0x80808080ul);
        __m128i rl = _mm_sub_epi64(zero, _mm_shuffle_epi8(r, sl));

        __m128i su = _mm_setr_epi32(0x80808002ul, 0x80808080ul, 0x80808003ul, 0x80808080ul);
        __m128i ru = _mm_sub_epi64(zero, _mm_shuffle_epi8(r, su));

        return _mm256_castsi256_pd(combine_m128i(ru, rl));
    }

protected:
    static __m256i combine_m128i(__m128i hi, __m128i lo) {
        return _mm256_insertf128_si256(_mm256_castsi128_si256(lo), hi, 1);
    }
};


#ifdef __AVX2__
using avx2_int4 = avx_int4;

struct avx2_double4;

template <>
struct simd_traits<avx2_double4> {
    static constexpr unsigned width = 4;
    using scalar_type = double;
    using vector_type = __m256d;
    using mask_impl = avx2_double4;
};

struct avx2_double4: avx_double4 {
    static __m256d fma(const __m256d& a, const __m256d& b, const __m256d& c) {
        return _mm256_fmadd_pd(a, b, c);
    }

    static vector_type logical_not(const vector_type& a) {
        __m256i ones = {};
        return _mm256_xor_pd(a, _mm256_castsi256_pd(_mm256_cmpeq_epi32(ones, ones)));
    }

    static void mask_copy_to(const __m256d& m, bool* y) {
        __m256i zero = _mm256_setzero_si256();

        // Translate 0xffffffffffffffff scalars to 0x0000000000000001.

        __m256i x = _mm256_castpd_si256(m);
        x = _mm256_sub_epi64(zero, x);

        // Move lower 32-bits of each field to lower 128-bit half of x.

        __m256i s1 = _mm256_setr_epi32(0,2,4,8,0,0,0,0);
        x = _mm256_permutevar8x32_epi32(x, s1);

        // Move the lowest byte from each 32-bit field to bottom bytes.

        __m128i s2 = _mm_setr_epi32(0x0c080400ul,0,0,0);
        __m128i r = _mm_shuffle_epi8(_mm256_castsi256_si128(x), s2);
        std::memcpy(y, &r, 4);
    }

    static __m256d mask_copy_from(const bool* w) {
        __m256i zero = _mm256_setzero_si256();

        __m128i r;
        std::memcpy(&r, w, 4);
        return _mm256_castsi256_pd(_mm256_sub_epi64(zero, _mm256_cvtepi8_epi64(r)));
    }

    static __m256d gather(avx2_int4, const double* p, const __m128i& index) {
        return _mm256_i32gather_pd(p, index, 8);
    }

    static __m256d gather(avx2_int4, __m256d a, const double* p, const __m128i& index, const __m256d& mask) {
        return  _mm256_mask_i32gather_pd(a, p, index, mask, 8);
    };
};
#endif // def __AVX2__

} // namespace simd_detail

namespace simd_abi {
    template <typename T, unsigned N> struct avx;

    template <> struct avx<int, 4> { using type = simd_detail::avx_int4; };
    template <> struct avx<double, 4> { using type = simd_detail::avx_double4; };

#ifdef __AVX2__
    template <typename T, unsigned N> struct avx2;

    template <> struct avx2<int, 4> { using type = simd_detail::avx2_int4; };
    template <> struct avx2<double, 4> { using type = simd_detail::avx2_double4; };
#endif
} // namespace simd_abi;

} // namespace arb

#endif // def __AVX__
