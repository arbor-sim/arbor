#pragma once

// SVE SIMD intrinsics implementation with fixed sizes (VLS_SVE)

#ifdef __ARM_FEATURE_SVE
#include <cmath>
#include <cstdint>

#include <arbor/simd/vls_sve_traits.hpp>

namespace arb {
namespace simd {
namespace detail {

struct vls_sve_mask : public implbase<vls_sve_mask> {

    using base = implbase<vls_sve_mask>;

    static fvbool_t broadcast(bool b) noexcept {
        return svdup_b64(b);
    }

    // TODO: find efficient intrinsics for getting setting an element
    //static void set_element(fvbool_t& k, int i, bool b) noexcept;
    //static fvbool_t element(const fvbool_t& k, int i) noexcept;

    static void mask_set_element(fvbool_t& k, int i, bool b) noexcept {
        set_element(k, i, b);
    }

    static bool mask_element(const fvbool_t& k, int i) noexcept {
        return element(k, i);
    }

    static void copy_to(const fvbool_t& k, bool* b) noexcept {
        fvuint64_t a = svdup_u64_z(k, 1);
        svst1b_u64(svptrue_b64(), reinterpret_cast<uint8_t*>(b), a);
    }

    static void copy_to_masked(const fvbool_t& k, bool* b, const fvbool_t& mask) noexcept {
        fvuint64_t a = svdup_u64_z(k, 1);
        svst1b_u64(mask, reinterpret_cast<uint8_t*>(b), a);
    }

    static fvbool_t copy_from(const bool* p) noexcept {
        fvuint64_t a = svld1ub_u64(svptrue_b64(), reinterpret_cast<const uint8_t*>(p));
        fvuint64_t ones = svdup_n_u64(1);
        return svcmpeq_u64(svptrue_b64(), a, ones);
    }

    static fvbool_t copy_from_masked(const bool* p, const fvbool_t& mask) noexcept {
        fvuint64_t a = svld1ub_u64(mask, reinterpret_cast<const uint8_t*>(p));
        fvuint64_t ones = svdup_n_u64(1);
        return svcmpeq_u64(mask, a, ones);
    }

    static fvbool_t mask_broadcast(bool b) noexcept {
        return broadcast(b);
    }

    static void mask_copy_to(const fvbool_t& m, bool* y) noexcept {
        copy_to(m, y);
    }

    static fvbool_t mask_copy_from(const bool* y) noexcept {
        return copy_from(y);
    }

    static fvbool_t logical_not(const fvbool_t& k) noexcept {
        return svnot_b_z(svptrue_b64(), k);
    }

    static fvbool_t logical_and(const fvbool_t& a, const fvbool_t& b) noexcept {
        return svand_b_z(svptrue_b64(), a, b);
    }

    static fvbool_t logical_or(const fvbool_t& a, const fvbool_t& b) noexcept {
        return svorr_b_z(svptrue_b64(), a, b);
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

    static fvbool_t neg(const fvbool_t& a) noexcept {
        return a;
    }

    static fvbool_t add(const fvbool_t& a, const fvbool_t& b) noexcept {
        return sveor_b_z(svptrue_b64(), a, b);
    }

    static fvbool_t sub(const fvbool_t& a, const fvbool_t& b) noexcept {
        return sveor_b_z(svptrue_b64(), a, b);
    }

    static fvbool_t mul(const fvbool_t& a, const fvbool_t& b) noexcept {
        return svand_b_z(svptrue_b64(), a, b);
    }

    static fvbool_t div(const fvbool_t& a, const fvbool_t& b) noexcept {
        return a;
    }

    static fvbool_t fma(const fvbool_t& a, const fvbool_t& b, const fvbool_t& c) noexcept {
        return add(mul(a, b), c);
    }

    static fvbool_t max(const fvbool_t& a, const fvbool_t& b) noexcept {
        return svorr_b_z(svptrue_b64(), a, b);
    }

    static fvbool_t min(const fvbool_t& a, const fvbool_t& b) noexcept {
        return svand_b_z(svptrue_b64(), a, b);
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

    static fvbool_t cmp_eq(const fvbool_t& a, const fvbool_t& b) noexcept {
        return svnot_b_z(svptrue_b64(), sveor_b_z(svptrue_b64(), a, b));
    }

    static fvbool_t cmp_neq(const fvbool_t& a, const fvbool_t& b) noexcept {
        return sveor_b_z(svptrue_b64(), a, b);
    }

    static fvbool_t cmp_lt(const fvbool_t& a, const fvbool_t& b) noexcept {
        return svbic_b_z(svptrue_b64(), b, a);
    }

    static fvbool_t cmp_gt(const fvbool_t& a, const fvbool_t& b) noexcept {
        return cmp_lt(b, a);
    }

    static fvbool_t cmp_geq(const fvbool_t& a, const fvbool_t& b) noexcept {
        return logical_not(cmp_lt(a, b));
    }

    static fvbool_t cmp_leq(const fvbool_t& a, const fvbool_t& b) noexcept {
        return logical_not(cmp_gt(a, b));
    }

    static fvbool_t ifelse(const fvbool_t& m, const fvbool_t& u, const fvbool_t& v) noexcept {
        return svsel_b(m, u, v);
    }
};

struct vls_sve_int : public implbase<vls_sve_int> {

    using base = implbase<vls_sve_int>;
    using int32 = typename base::scalar_type;
    using store = typename base::store;

    using base::scatter;

    static fvint64_t broadcast(int32 v) noexcept {
        return svreinterpret_s64_s32(svdup_n_s32(v));
    }

    static void copy_to(const fvint64_t& v, int32* p) noexcept {
        svst1w_s64(svptrue_b64(), p, v);
    }

    static void copy_to_masked(const fvint64_t& v, int32* p, const fvbool_t& mask) noexcept {
        svst1w_s64(mask, p, v);
    }

    static fvint64_t copy_from(const int32* p) noexcept {
        return svld1sw_s64(svptrue_b64(), p);
    }

    static fvint64_t copy_from_masked(const int32* p, const fvbool_t& mask) noexcept {
        return svld1sw_s64(mask, p);
    }

    static fvint64_t copy_from_masked(const fvint64_t& v, const int32* p, const fvbool_t& mask) noexcept {
        return svsel_s64(mask, svld1sw_s64(mask, p), v);
    }

    static int element0(const fvint64_t& a) noexcept {
        return svlasta_s64(svptrue_b64(), a);
    }

    static fvint64_t neg(const fvint64_t& a) noexcept {
        return svneg_s64_z(svptrue_b64(), a);
    }

    static fvint64_t add(const fvint64_t& a, const fvint64_t& b) noexcept {
        return svadd_s64_z(svptrue_b64(), a, b);
    }

    static fvint64_t sub(const fvint64_t& a, const fvint64_t& b) noexcept {
        return svsub_s64_m(svptrue_b64(), a, b);
    }

    static fvint64_t mul(const fvint64_t& a, const fvint64_t& b) noexcept {
        //May overflow
        return svmul_s64_z(svptrue_b64(), a, b);
    }

    static fvint64_t div(const fvint64_t& a, const fvint64_t& b) noexcept {
        return svdiv_s64_z(svptrue_b64(), a, b);
    }

    static fvint64_t fma(const fvint64_t& a, const fvint64_t& b, const fvint64_t& c) noexcept {
        return add(mul(a, b), c);
    }

    static fvbool_t cmp_eq(const fvint64_t& a, const fvint64_t& b) noexcept {
        return svcmpeq_s64(svptrue_b64(), a, b);
    }

    static fvbool_t cmp_neq(const fvint64_t& a, const fvint64_t& b) noexcept {
        return svcmpne_s64(svptrue_b64(), a, b);
    }

    static fvbool_t cmp_gt(const fvint64_t& a, const fvint64_t& b) noexcept {
        return svcmpgt_s64(svptrue_b64(), a, b);
    }

    static fvbool_t cmp_geq(const fvint64_t& a, const fvint64_t& b) noexcept {
        return svcmpge_s64(svptrue_b64(), a, b);
    }

    static fvbool_t cmp_lt(const fvint64_t& a, const fvint64_t& b) noexcept {
        return svcmplt_s64(svptrue_b64(), a, b);
    }

    static fvbool_t cmp_leq(const fvint64_t& a, const fvint64_t& b) noexcept {
        return svcmple_s64(svptrue_b64(), a, b);
    }

    static fvint64_t ifelse(const fvbool_t& m, const fvint64_t& u, const fvint64_t& v) noexcept {
        return svsel_s64(m, u, v);
    }

    static fvint64_t max(const fvint64_t& a, const fvint64_t& b) noexcept {
        return svmax_s64_x(svptrue_b64(), a, b);
    }

    static fvint64_t min(const fvint64_t& a, const fvint64_t& b) noexcept {
        return svmin_s64_x(svptrue_b64(), a, b);
    }

    static fvint64_t abs(const fvint64_t& a) noexcept {
        return svabs_s64_z(svptrue_b64(), a);
    }

    static int reduce_add(const fvint64_t& a) noexcept {
        return svaddv_s64(svptrue_b64(), a);
    }

    // pow implemented in implbase

    static fvint64_t gather(tag<vls_sve_int>, const int32* p, const fvint64_t& index) noexcept {
        return svld1sw_gather_s64index_s64(svptrue_b64(), p, index);
    }

    static fvint64_t gather(tag<vls_sve_int>, fvint64_t a, const int32* p, const fvint64_t& index, const fvbool_t& mask) noexcept {
        return svsel_s64(mask, svld1sw_gather_s64index_s64(mask, p, index), a);
    }

    static void scatter(tag<vls_sve_int>, const fvint64_t& s, int32* p, const fvint64_t& index) noexcept {
        svst1w_scatter_s64index_s64(svptrue_b64(), p, index, s);
    }
};

struct vls_sve_double : public implbase<vls_sve_double> {

    using base = implbase<vls_sve_double>;
    using store = typename base::store;

    using base::scatter;

    static fvfloat64_t broadcast(double v) noexcept {
        return svdup_n_f64(v);
    }

    static void copy_to(const fvfloat64_t& v, double* p) noexcept {
        svst1_f64(svptrue_b64(), p, v);
    }

    static void copy_to_masked(const fvfloat64_t& v, double* p, const fvbool_t& mask) noexcept {
        svst1_f64(mask, p, v);
    }

    static fvfloat64_t copy_from(const double* p) noexcept {
        return svld1_f64(svptrue_b64(), p);
    }

    static fvfloat64_t copy_from_masked(const double* p, const fvbool_t& mask) noexcept {
        return svld1_f64(mask, p);
    }

    static fvfloat64_t copy_from_masked(const fvfloat64_t& v, const double* p, const fvbool_t& mask) noexcept {
        return svsel_f64(mask, svld1_f64(mask, p), v);
    }

    static double element0(const fvfloat64_t& a) noexcept {
        return svlasta_f64(svptrue_b64(), a);
    }

    static fvfloat64_t neg(const fvfloat64_t& a) noexcept {
        return svneg_f64_z(svptrue_b64(), a);
    }

    static fvfloat64_t add(const fvfloat64_t& a, const fvfloat64_t& b) noexcept {
        return svadd_f64_z(svptrue_b64(), a, b);
    }

    static fvfloat64_t sub(const fvfloat64_t& a, const fvfloat64_t& b) noexcept {
        return svsub_f64_z(svptrue_b64(), a, b);
    }

    static fvfloat64_t mul(const fvfloat64_t& a, const fvfloat64_t& b) noexcept {
        return svmul_f64_z(svptrue_b64(), a, b);
    }

    static fvfloat64_t div(const fvfloat64_t& a, const fvfloat64_t& b) noexcept {
        return svdiv_f64_z(svptrue_b64(), a, b);
    }

    static fvfloat64_t fma(const fvfloat64_t& a, const fvfloat64_t& b, const fvfloat64_t& c) noexcept {
        return svmad_f64_z(svptrue_b64(), a, b, c);
    }

    static fvbool_t cmp_eq(const fvfloat64_t& a, const fvfloat64_t& b) noexcept {
        return svcmpeq_f64(svptrue_b64(), a, b);
    }

    static fvbool_t cmp_neq(const fvfloat64_t& a, const fvfloat64_t& b) noexcept {
        return svcmpne_f64(svptrue_b64(), a, b);
    }

    static fvbool_t cmp_gt(const fvfloat64_t& a, const fvfloat64_t& b) noexcept {
        return svcmpgt_f64(svptrue_b64(), a, b);
    }

    static fvbool_t cmp_geq(const fvfloat64_t& a, const fvfloat64_t& b) noexcept {
        return svcmpge_f64(svptrue_b64(), a, b);
    }

    static fvbool_t cmp_lt(const fvfloat64_t& a, const fvfloat64_t& b) noexcept {
        return svcmplt_f64(svptrue_b64(), a, b);
    }

    static fvbool_t cmp_leq(const fvfloat64_t& a, const fvfloat64_t& b) noexcept {
        return svcmple_f64(svptrue_b64(), a, b);
    }

    static fvfloat64_t ifelse(const fvbool_t& m, const fvfloat64_t& u, const fvfloat64_t& v) noexcept {
        return svsel_f64(m, u, v);
    }

    static fvfloat64_t max(const fvfloat64_t& a, const fvfloat64_t& b) noexcept {
        return svmax_f64_x(svptrue_b64(), a, b);
    }

    static fvfloat64_t min(const fvfloat64_t& a, const fvfloat64_t& b) noexcept {
        return svmin_f64_x(svptrue_b64(), a, b);
    }

    static fvfloat64_t abs(const fvfloat64_t& x) noexcept {
        return svabs_f64_x(svptrue_b64(), x);
    }

    static double reduce_add(const fvfloat64_t& a) noexcept {
        return svaddv_f64(svptrue_b64(), a);
    }

    static fvfloat64_t gather(tag<vls_sve_int>, const double* p, const fvint64_t& index) noexcept {
        return svld1_gather_s64index_f64(svptrue_b64(), p, index);
    }

    static fvfloat64_t gather(tag<vls_sve_int>, fvfloat64_t a, const double* p, const fvint64_t& index, const fvbool_t& mask) noexcept {
        return svsel_f64(mask, svld1_gather_s64index_f64(mask, p, index), a);
    }

    static void scatter(tag<vls_sve_int>, const fvfloat64_t& s, double* p, const fvint64_t& index) noexcept {
        svst1_scatter_s64index_f64(svptrue_b64(), p, index, s);
    }

    // Refer to avx/avx2 code for details of the exponential and log
    // implementations.

    static fvfloat64_t exp(const fvfloat64_t& x) noexcept {
        // Masks for exceptional cases.

        auto is_large = cmp_gt(x, broadcast(exp_maxarg));
        auto is_small = cmp_lt(x, broadcast(exp_minarg));

        // Compute n and g.

        fvfloat64_t n = svrintm_f64_z(svptrue_b64(), add(mul(broadcast(ln2inv), x), broadcast(0.5)));

        auto g = fma(n, broadcast(-ln2C1), x);
        g = fma(n, broadcast(-ln2C2), g);

        auto gg = mul(g, g);

        // Compute the g*P(g^2) and Q(g^2).

        auto odd = mul(g, horner(gg, P0exp, P1exp, P2exp));
        auto even = horner(gg, Q0exp, Q1exp, Q2exp, Q3exp);

        // Compute R(g)/R(-g) = 1 + 2*g*P(g^2) / (Q(g^2)-g*P(g^2))

        auto expg = fma(broadcast(2), div(odd, sub(even, odd)), broadcast(1));

        // Scale by 2^n, propogating NANs.

        fvfloat64_t result = svscale_f64_z(svptrue_b64(), expg, svcvt_s64_f64_z(svptrue_b64(), n));

        return
            ifelse(is_large, broadcast(HUGE_VAL),
            ifelse(is_small, broadcast(0),
                   result));
    }

    static fvfloat64_t expm1(const fvfloat64_t& x) noexcept {
        fvbool_t is_large = cmp_gt(x, broadcast(exp_maxarg));
        fvbool_t is_small = cmp_lt(x, broadcast(expm1_minarg));

        fvfloat64_t half = broadcast(0.5);
        fvfloat64_t one = broadcast(1.);

        auto nnz = cmp_gt(abs(x), half);
        fvfloat64_t n = svrinta_f64_z(nnz, mul(broadcast(ln2inv), x));

        auto g = fma(n, broadcast(-ln2C1), x);
        g = fma(n, broadcast(-ln2C2), g);

        auto gg = mul(g, g);

        auto odd = mul(g, horner(gg, P0exp, P1exp, P2exp));
        auto even = horner(gg, Q0exp, Q1exp, Q2exp, Q3exp);

        // Compute R(g)/R(-g) -1 = 2*g*P(g^2) / (Q(g^2)-g*P(g^2))

        auto expgm1 = div(mul(broadcast(2), odd), sub(even, odd));

        // For small x (n zero), bypass scaling step to avoid underflow.
        // Otherwise, compute result 2^n * expgm1 + (2^n-1) by:
        //     result = 2 * ( 2^(n-1)*expgm1 + (2^(n-1)+0.5) )
        // to avoid overflow when n=1024.

        fvint64_t nm1 = svcvt_s64_f64_z(svptrue_b64(), sub(n, one));

        auto result =
            svscale_f64_z(svptrue_b64(),
                add(sub(svscale_f64_z(svptrue_b64(),one, nm1), half),
                    svscale_f64_z(svptrue_b64(),expgm1, nm1)),
                svcvt_s64_f64_z(svptrue_b64(), one));

        return
            ifelse(is_large, broadcast(HUGE_VAL),
            ifelse(is_small, broadcast(-1),
            ifelse(nnz, result, expgm1)));
    }

    static fvfloat64_t log(const fvfloat64_t& x) noexcept {
        // Masks for exceptional cases.

        auto is_large = cmp_geq(x, broadcast(HUGE_VAL));
        auto is_small = cmp_lt(x, broadcast(log_minarg));
        auto is_domainerr = cmp_lt(x, broadcast(0));

        fvbool_t is_nan = svnot_b_z(svptrue_b64(), cmp_eq(x, x));
        is_domainerr = svorr_b_z(svptrue_b64(), is_nan, is_domainerr);

        fvfloat64_t g = svcvt_f64_s32_z(svptrue_b64(), logb_normal(x));
        fvfloat64_t u = fraction_normal(x);

        fvfloat64_t one = broadcast(1.);
        fvfloat64_t half = broadcast(0.5);
        auto gtsqrt2 = cmp_geq(u, broadcast(sqrt2));
        g = ifelse(gtsqrt2, add(g, one), g);
        u = ifelse(gtsqrt2, mul(u, half), u);

        auto z = sub(u, one);
        auto pz = horner(z, P0log, P1log, P2log, P3log, P4log, P5log);
        auto qz = horner1(z, Q0log, Q1log, Q2log, Q3log, Q4log);

        auto z2 = mul(z, z);
        auto z3 = mul(z2, z);

        auto r = div(mul(z3, pz), qz);
        r = add(r, mul(g,  broadcast(ln2C4)));
        r = sub(r, mul(z2, half));
        r = add(r, z);
        r = add(r, mul(g,  broadcast(ln2C3)));

        // r is alrady NaN if x is NaN or negative, otherwise
        // return  +inf if x is +inf, or -inf if zero or (positive) denormal.

        return ifelse(is_domainerr, broadcast(NAN),
                      ifelse(is_large, broadcast(HUGE_VAL),
                             ifelse(is_small, broadcast(-HUGE_VAL), r)));
    }

protected:
    // Compute n and f such that x = 2^n·f, with |f| ∈ [1,2), given x is finite and normal.
    static inline fvint32_t logb_normal(const fvfloat64_t& x) noexcept {
        fvuint32_t xw    = svtrn2_u32(svreinterpret_u32_f64(x), svreinterpret_u32_f64(x));
        fvuint64_t lmask = svdup_n_u64(0x00000000ffffffff);
        fvuint64_t xt = svand_u64_z(svptrue_b64(), svreinterpret_u64_u32(xw), lmask);
        fvuint32_t xhi = svreinterpret_u32_u64(xt);
        fvuint32_t emask = svdup_n_u32(0x7ff00000);
        fvuint32_t ebiased = svlsr_n_u32_z(svptrue_b64(), svand_u32_z(svptrue_b64(), xhi, emask), 20);

        return svsub_s32_z(svptrue_b64(), svreinterpret_s32_u32(ebiased), svdup_n_s32(1023));
    }

    static inline fvfloat64_t fraction_normal(const fvfloat64_t& x) noexcept {
        fvuint64_t emask = svdup_n_u64(-0x7ff0000000000001);
        fvuint64_t bias =  svdup_n_u64(0x3ff0000000000000);
        return svreinterpret_f64_u64(
            svorr_u64_z(svptrue_b64(), bias, svand_u64_z(svptrue_b64(), emask, svreinterpret_u64_f64(x))));
    }

    static inline fvfloat64_t horner1(fvfloat64_t x, double a0) noexcept {
        return add(x, broadcast(a0));
    }

    static inline fvfloat64_t horner(fvfloat64_t x, double a0) noexcept {
        return broadcast(a0);
    }

    template <typename... T>
    static fvfloat64_t horner(fvfloat64_t x, double a0, T... tail) noexcept {
        return fma(x, horner(x, tail...), broadcast(a0));
    }

    template <typename... T>
    static fvfloat64_t horner1(fvfloat64_t x, double a0, T... tail) noexcept {
        return fma(x, horner1(x, tail...), broadcast(a0));
    }

    static fvfloat64_t fms(const fvfloat64_t& a, const fvfloat64_t& b, const fvfloat64_t& c) noexcept {
        return svnmsb_f64_z(svptrue_b64(), a, b, c);
    }
};

} // namespace detail

namespace simd_abi {
    template <typename T, unsigned N> struct vls_sve;
    template <> struct vls_sve<int, detail::vls_sve_width> { using type = detail::vls_sve_int; };
    template <> struct vls_sve<double, detail::vls_sve_width> { using type = detail::vls_sve_double; };
} // namespace simd_abi

} // namespace simd
} // namespace arb

#endif  // def __ARM_FEATURE_SVE
