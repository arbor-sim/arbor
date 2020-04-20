#pragma once

// VSX SIMD intrinsics implementation.
#if defined(__VEC__) || defined(__ALTIVEC__) || defined(__VSX__)
#include <altivec.h>
#include <cmath>
#include <cstdint>

#include <arbor/simd/approx.hpp>
#include <arbor/simd/implbase.hpp>

namespace arb {
namespace simd {
namespace detail {

struct vsx_double4;
struct vsx_int4;
struct vsx_bool2x2;
struct vsx_bool4;

template <> struct simd_traits<vsx_double4> {
    static constexpr unsigned width = 4;
    using scalar_type = double;
    using vector_type = std::array<vector double, 2>;
    using mask_impl = vsx_bool2x2;
};

template <> struct simd_traits<vsx_int4> {
    static constexpr unsigned width = 4;
    using scalar_type = int;
    using vector_type = vector int;
    using mask_impl = vsx_bool4;
};

template <> struct simd_traits<vsx_bool2x2> {
    static constexpr unsigned width = 4;
    using scalar_type = bool;
    using vector_type = std::array<vector bool long long, 2>;
    using mask_impl = vsx_bool2x2;
};

template <> struct simd_traits<vsx_bool4> {
    static constexpr unsigned width = 4;
    using scalar_type = bool;
    using vector_type = vector bool int;
    using mask_impl = vsx_bool4;
};

struct vsx_int4 : implbase<vsx_int4> {
    using array = vector int;
    using bools = vector bool int;

    static void copy_to(const array& v, int* p) {
        p[0] = v[0];
        p[1] = v[1];
        p[2] = v[2];
        p[3] = v[3];
    }

    static array copy_from(const int* p) {
        array result;
        result[0] = p[0];
        result[1] = p[1];
        result[2] = p[2];
        result[3] = p[3];
        return result;
    }

    static bools cmp_eq(const array& a, const array& b) {
        bools result;
        result = vec_cmpeq(a, b);
        return result;
    }

    static bools cmp_neq(const array& a, const array& b) {
        return !cmp_eq(a, b);
    }

    static bools cmp_lt(const array& a, const array& b) {
        bools result;
        result = vec_cmplt(a, b);
        return result;
    }

    static bools cmp_ge(const array& a, const array& b) {
        bools result;
        result = vec_cmplt(a, b);
        return !result;
    }

    static bools cmp_gt(const array& a, const array& b) {
        bools result;
        result = vec_cmpgt(a, b);
        return result;
    }

    static bools cmp_le(const array& a, const array& b) {
        bools result;
        result = vec_cmpgt(a, b);
        return !result;
    }

    static array broadcast(int v) {
        array result;
        result = vec_splats(v);
        return result;
    }

    static array add(const array& a, const array& b) {
        array result;
        result = a + b;
        return result;
    }

    static array sub(const array& a, const array& b) {
        array result;
        result = a - b;
        return result;
    }

    static array mul(const array& a, const array& b) {
        array result;
        result = a * b;
        return result;
    }

    static array div(const array& a, const array& b) {
        array result;
        result = a / b;
        return result;
    }

    static array min(const array& a, const array& b) {
        array result;
        result = vec_min(a, b);
        return result;
    }

    static array max(const array& a, const array& b) {
        array result;
        result = vec_max(a, b);
        return result;
    }

    static array abs(const array& a) {
        array result;
        result = vec_abs(a);
        return result;
    }
};

struct vsx_bool4 : implbase<vsx_bool4> {
    using array = vector bool int;

    static void copy_to(const array& v, bool* p) {
        p[0] = v[0];
        p[1] = v[1];
        p[2] = v[2];
        p[3] = v[3];
    }

    static array copy_from(const bool* p) {
        array result;
        result[0] = p[0];
        result[1] = p[1];
        result[2] = p[2];
        result[3] = p[3];
        return result;
    }

    static void mask_copy_to(const array& v, bool* p) {
        p[0] = v[0];
        p[1] = v[1];
        p[2] = v[2];
        p[3] = v[3];
    }

    static array mask_copy_from(const bool* p) {
        array result;
        result[0] = p[0] ? (~0) : 0;
        result[1] = p[1] ? (~0) : 0;
        result[2] = p[2] ? (~0) : 0;
        result[3] = p[3] ? (~0) : 0;
        return result;
    }

    static bool mask_element(const array& v, int i) {
        return static_cast<bool>(v[i]);
    }

    static void mask_set_element(array& v, int i, bool x) {
        v[i] = x ? (~0) : 0;
    }

    static array logical_not(const array& v) {
        array result;
        result = !v;
        return result;
    }

    static array logical_and(const array& v, const array& w) {
        array result;
        result = v & w;
        return result;
    }

    static array logical_or(const array& v, const array& w) {
        array result;
        result = v | w;
        return result;
    }
};

struct vsx_bool2x2 : implbase<vsx_bool2x2> {
    using array = std::array<vector bool long long, 2>;

    static void copy_to(const array& v, bool* p) {
        p[0] = v[0][0];
        p[1] = v[0][1];
        p[2] = v[1][0];
        p[3] = v[1][1];
    }

    static array copy_from(const bool* p) {
        array result;
        result[0][0] = p[0];
        result[0][1] = p[1];
        result[1][0] = p[2];
        result[1][1] = p[3];
        return result;
    }

    static void mask_copy_to(const array& v, bool* p) {
        p[0] = v[0][0];
        p[1] = v[0][1];
        p[2] = v[1][0];
        p[3] = v[1][1];
    }

    static array mask_copy_from(const bool* p) {
        array result;
        result[0][0] = p[0] ? (~0) : 0;
        result[0][1] = p[1] ? (~0) : 0;
        result[1][0] = p[2] ? (~0) : 0;
        result[1][1] = p[3] ? (~0) : 0;
        return result;
    }

    static bool mask_element(const array& v, int i) {
        auto s = i & 1;
        auto k = (i >> 1) & 1;
        return static_cast<bool>(v[k][s]);
    }

    static void mask_set_element(array& v, int i, bool x) {
        auto s = i & 1;
        auto k = (i >> 1) & 1;
        v[k][s] = x ? (~0) : 0;
    }
};

struct vsx_double4 : implbase<vsx_double4> {
    using array = std::array<vector double, 2>;
    using bools = std::array<vector bool long long, 2>;
    using ints = vector int;

    static void copy_to(const array& v, double* p) {
        p[0] = v[0][0];
        p[1] = v[0][1];
        p[2] = v[1][0];
        p[3] = v[1][1];
    }

    static array copy_from(const double* p) {
        array result;
        result[0][0] = p[0];
        result[0][1] = p[1];
        result[1][0] = p[2];
        result[1][1] = p[3];
        return result;
    }

    static array add(const array& a, const array& b) {
        array result;
        result[0] = a[0] + b[0];
        result[1] = a[1] + b[1];
        return result;
    }

    static array fma(const array& a, const array& b, const array& c) {
        array result;
        result[0] = vec_madd(a[0], b[0], c[0]);
        result[1] = vec_madd(a[1], b[1], c[1]);
        return result;
    }

    static array sub(const array& a, const array& b) {
        array result;
        result[0] = a[0] - b[0];
        result[1] = a[1] - b[1];
        return result;
    }

    static array fms(const array& a, const array& b, const array& c) {
        array result;
        result[0] = vec_msub(a[0], b[0], c[0]);
        result[1] = vec_msub(a[1], b[1], c[1]);
        return result;
    }

    static array mul(const array& a, const array& b) {
        array result;
        result[0] = a[0] * b[0];
        result[1] = a[1] * b[1];
        return result;
    }

    static array div(const array& a, const array& b) {
        array result;
        result[0] = a[0] / b[0];
        result[1] = a[1] / b[1];
        return result;
    }

    static array min(const array& a, const array& b) {
        array result;
        result[0] = vec_min(a[0], b[0]);
        result[1] = vec_min(a[1], b[1]);
        return result;
    }

    static array max(const array& a, const array& b) {
        array result;
        result[0] = vec_max(a[0], b[0]);
        result[1] = vec_max(a[1], b[1]);
        return result;
    }

    static array abs(const array& a) {
        array result;
        result[0] = vec_abs(a[0]);
        result[1] = vec_abs(a[1]);
        return result;
    }

    static bools cmp_eq(const array& a, const array& b) {
        bools result;
        result[0] = vec_cmpeq(a[0], b[0]);
        result[1] = vec_cmpeq(a[1], b[1]);
        return result;
    }

    static bools cmp_lt(const array& a, const array& b) {
        bools result;
        result[0] = vec_cmplt(a[0], b[0]);
        result[1] = vec_cmplt(a[1], b[1]);
        return result;
    }

    static bools cmp_gt(const array& a, const array& b) {
        bools result;
        result[0] = vec_cmpgt(a[0], b[0]);
        result[1] = vec_cmpgt(a[1], b[1]);
        return result;
    }

    static array broadcast(double v) {
        array result;
        result[0] = vec_splats(v);
        result[1] = vec_splats(v);
        return result;
    }

    static array ifelse(const bools& m, const array& u, const array& v) {
        array result;
        result[0] = vec_sel(v[0], u[0], m[0]);
        result[1] = vec_sel(v[1], u[1], m[1]);
        return result;
    }

    static array floor(const array& a) {
        array result;
        result[0] = vec_floor(a[0]);
        result[1] = vec_floor(a[1]);
        return result;
    }

    // Stolen from avx2
    static array exp(const array& x) {
        const auto is_large = cmp_gt(x, broadcast(exp_maxarg));
        const auto is_small = cmp_lt(x, broadcast(exp_minarg));
        const auto is_not_nan = cmp_eq(x, x);

        // Compute n and g.
        const auto n = floor(fma(broadcast(ln2inv), x, broadcast(0.5)));
        auto g = fma(n, broadcast(-ln2C1), x);
        g = fma(n, broadcast(-ln2C2), g);
        const auto gg = mul(g, g);

        // Compute the g*P(g^2) and Q(g^2).
        const auto odd = mul(g, horner(gg, P0exp, P1exp, P2exp));
        const auto even = horner(gg, Q0exp, Q1exp, Q2exp, Q3exp);

        // Compute R(g)/R(-g) = 1 + 2*g*P(g^2) / (Q(g^2)-g*P(g^2))
        const auto expg =
            fma(broadcast(2.0), div(odd, sub(even, odd)), broadcast(1.0));

        // Finally, compute product with 2^n.
        // Note: can only achieve full range using the ldexp implementation,
        // rather than multiplying by 2^n directly.
        const auto result = ldexp_positive(expg, cfti(n));
        return ifelse(is_large, broadcast(HUGE_VAL),
                      ifelse(is_small, broadcast(0.0),
                             ifelse(is_not_nan, result, broadcast(NAN))));
    }

    static array expm1(const array& x) {
        const auto is_large = cmp_gt(x, broadcast(exp_maxarg));
        const auto is_small = cmp_lt(x, broadcast(expm1_minarg));
        const auto is_not_nan = cmp_eq(x, x);

        const auto zero = broadcast(0.0);
        const auto half = broadcast(0.5);
        const auto one = broadcast(1.0);
        const auto two = add(one, one);

        const auto smallx = cmp_leq(abs(x), half);
        auto n = floor(fma(broadcast(ln2inv), x, half));
        n = ifelse(smallx, zero, n);

        auto g = fma(n, broadcast(-ln2C1), x);
        g = fma(n, broadcast(-ln2C2), g);

        const auto gg = mul(g, g);

        const auto odd = mul(g, horner(gg, P0exp, P1exp, P2exp));
        const auto even = horner(gg, Q0exp, Q1exp, Q2exp, Q3exp);

        const auto expgm1 = div(mul(broadcast(2.0), odd), sub(even, odd));

        const auto nm1 = cfti(sub(n, one));
        const auto scaled =
            mul(add(sub(exp2int(nm1), half), ldexp_normal(expgm1, nm1)), two);

        return ifelse(is_large, broadcast(HUGE_VAL),
                      ifelse(is_small, broadcast(-1),
                             ifelse(is_not_nan, ifelse(smallx, expgm1, scaled),
                                    broadcast(NAN))));
    }

    static array log(const array& x) {
        // Masks for exceptional cases.
        const auto is_not_large = cmp_lt(x, broadcast(HUGE_VAL));
        const auto is_small = cmp_lt(x, broadcast(log_minarg));
        const auto is_not_nan = cmp_eq(x, x);
        // error if x < 0 or n == nan
        // nb. x == 0 is handled by is_small
        auto is_domainerr = cmp_lt(x, broadcast(0.0));
        is_domainerr[0] |= ~is_not_nan[0];
        is_domainerr[1] |= ~is_not_nan[1];

        const array one = broadcast(1.0);
        const array half = broadcast(0.5);

        array g = logb_normal(x);
        array u = fraction_normal(x);
        const auto gtsqrt2 = cmp_geq(u, broadcast(sqrt2));
        g = ifelse(gtsqrt2, add(g, one), g);
        u = ifelse(gtsqrt2, mul(u, half), u);

        const auto z = sub(u, one);
        const auto pz = horner(z, P0log, P1log, P2log, P3log, P4log, P5log);
        const auto qz = horner1(z, Q0log, Q1log, Q2log, Q3log, Q4log);

        const auto z2 = mul(z, z);
        const auto z3 = mul(z2, z);

        auto r = div(mul(z3, pz), qz);
        r = fma(g, broadcast(ln2C4), r);
        r = fms(z2, half, r);
        r = sub(z, r);
        r = fma(g, broadcast(ln2C3), r);

        // Return NaN if x is NaN or negative, +inf if x is +inf,
        // or -inf if zero or (positive) denormal.
        return ifelse(is_domainerr, broadcast(NAN),
                      ifelse(is_not_large,
                             ifelse(is_small, broadcast(-HUGE_VAL), r),
                             broadcast(HUGE_VAL)));
    }

  protected:
    static inline array horner(const array& x, const double a0) {
        return broadcast(a0);
    }

    template <typename... T>
    static array horner(const array& x, const double a0, T... tail) {
        return fma(x, horner(x, tail...), broadcast(a0));
    }

    static inline array horner1(const array& x, const double a0) {
        return add(x, broadcast(a0));
    }

    template <typename... T>
    static array horner1(const array& x, const double a0, T... tail) {
        return fma(x, horner1(x, tail...), broadcast(a0));
    }

    // Compute 2^n·x
    static array ldexp_positive(const array& x, const ints n) {
        const auto nshift_l = vec_vupkhsw(n)
                              << 52; // the unpack *high* unpacks indices 0
                                     // and 1. BigEndian, I guess
        const auto nshift_h = vec_vupklsw(n) << 52;
        const auto x_l = reinterpret_cast<vector long long>(x[0]);
        const auto x_h = reinterpret_cast<vector long long>(x[1]);
        const auto s_l = x_l + nshift_l;
        const auto s_h = x_h + nshift_h;
        array res;
        res[0] = reinterpret_cast<vector double>(s_l);
        res[1] = reinterpret_cast<vector double>(s_h);
        return res;
    }

    // Compute n and f such that x = 2^n·f, with |f| ∈ [1,2), given x is finite
    static array logb_normal(const array& x) {
        auto xw =
            vec_perm(reinterpret_cast<vector unsigned>(x[0]),
                     reinterpret_cast<vector unsigned>(x[1]),
                     (vector unsigned char){4, 5, 6, 7, 12, 13, 14, 15, 20, 21,
                                            22, 23, 28, 29, 30, 31});
        auto mask =
            (vector unsigned){0x7ff00000, 0x7ff00000, 0x7ff00000, 0x7ff00000};
        auto c1023 = (vector int){1023, 1023, 1023, 1023};
        auto res = reinterpret_cast<vector int>((xw & mask) >> 20) - c1023;
        array r;
        r[0][0] = res[0];
        r[0][1] = res[1];
        r[1][0] = res[2];
        r[1][1] = res[3];
        return r;
    }

    static array fraction_normal(const array& x) {
        const auto mask =
            (vector long long){-0x7ff0000000000001ll, -0x7ff0000000000001ll};
        const auto bias =
            (vector long long){0x3ff0000000000000ll, 0x3ff0000000000000ll};
        const auto x_l = reinterpret_cast<vector long long>(x[0]);
        const auto x_h = reinterpret_cast<vector long long>(x[1]);
        array result;
        result[0] = reinterpret_cast<vector double>(bias | (mask & x_l));
        result[1] = reinterpret_cast<vector double>(bias | (mask & x_h));
        return result;
    }

    // Compute 2^n*x when both x and 2^n*x are normal and finite.
    static array ldexp_normal(const array& x, const ints n) {
        const auto x_l = reinterpret_cast<vector long long>(x[0]);
        const auto x_h = reinterpret_cast<vector long long>(x[1]);
        const auto smask =
            (vector long long){0x7fffffffffffffffll, 0x7fffffffffffffffll};
        const auto sbits_l = (~smask) & x_l;
        const auto sbits_h = (~smask) & x_h;
        const auto nshift_l = vec_vupkhsw(n)
                              << 52; // the unpack *high* unpacks indices 0
                                     // and 1. BigEndian, I guess
        const auto nshift_h = vec_vupklsw(n) << 52;
        const auto s_l = x_l + nshift_l;
        const auto s_h = x_h + nshift_h;
        const auto nzans_l = (s_l & smask) | sbits_l;
        const auto nzans_h = (s_h & smask) | sbits_h;
        array result;
        result[0] = reinterpret_cast<vector double>(nzans_l);
        result[1] = reinterpret_cast<vector double>(nzans_h);
        return ifelse(cmp_eq(x, broadcast(0.0)), broadcast(0.0), result);
    }

    static array exp2int(const ints n) {
        return ldexp_positive(broadcast(1.0), n);
    }

    static ints cfti(const array& v) {
        return (ints){static_cast<int>(v[0][0]), static_cast<int>(v[0][1]),
                      static_cast<int>(v[1][0]), static_cast<int>(v[1][1])};
    }
};
} // namespace detail

namespace simd_abi {
template <typename T, unsigned N> struct vsx;

template <> struct vsx<double, 4> { using type = detail::vsx_double4; };

template <> struct vsx<int, 4> { using type = detail::vsx_int4; };
} // namespace simd_abi
} // namespace simd
} // namespace arb

#endif // def __VSX
