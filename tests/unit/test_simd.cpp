#include <cmath>
#include <random>

#include <util/simd.hpp>
#include <util/simd/avx.hpp>

#include "common.hpp"

// For quicker compiling while debugging:
#define DEFAULT_ABI_ONLY

using namespace arb;

namespace {
    template <typename V, typename = typename std::enable_if<std::is_floating_point<V>::value>::type>
    std::uniform_real_distribution<V> make_udist(V lb = -1., V ub = 1.) {
        return std::uniform_real_distribution<V>(lb, ub);
    }

    template <typename V, typename = typename std::enable_if<std::is_integral<V>::value && !std::is_same<V, bool>::value>::type>
    std::uniform_int_distribution<V> make_udist(
            V lb = std::numeric_limits<V>::lowest() / (2 << std::numeric_limits<V>::digits/2),
            V ub = std::numeric_limits<V>::max() >> (1+std::numeric_limits<V>::digits/2))
    {
        return std::uniform_int_distribution<V>(lb, ub);
    }

    template <typename V, typename = typename std::enable_if<std::is_same<V, bool>::value>::type>
    std::uniform_int_distribution<> make_udist(V lb = 0, V ub = 1) {
        return std::uniform_int_distribution<>(0, 1);
    }

    template <typename Seq, typename Rng>
    void fill_random(Seq&& seq, Rng& rng) {
        using V = typename std::decay<decltype(*std::begin(seq))>::type;

        auto u = make_udist<V>();
        for (auto& x: seq) { x = u(rng); }
    }

    template <typename Seq, typename Rng, typename B1, typename B2>
    void fill_random(Seq&& seq, Rng& rng, B1 lb, B2 ub) {
        using V = typename std::decay<decltype(*std::begin(seq))>::type;

        auto u = make_udist<V>(lb, ub);
        for (auto& x: seq) { x = u(rng); }
    }

    template <typename Simd, typename Rng, typename B1, typename B2, typename = typename std::enable_if<is_simd<Simd>::value>::type>
    void fill_random(Simd& s, Rng& rng, B1 lb, B2 ub) {
        using V = typename Simd::scalar_type;
        constexpr unsigned N = Simd::width;

        V v[N];
        fill_random(v, rng, lb, ub);
        s.copy_from(v);
    }

    template <typename Simd, typename Rng, typename = typename std::enable_if<is_simd<Simd>::value>::type>
    void fill_random(Simd& s, Rng& rng) {
        using V = typename Simd::scalar_type;
        constexpr unsigned N = Simd::width;

        V v[N];
        fill_random(v, rng);
        s.copy_from(v);
    }

    template <typename Simd>
    ::testing::AssertionResult simd_eq(Simd a, Simd b) {
        constexpr unsigned N = Simd::width;
        using V = typename Simd::scalar_type;

        V as[N], bs[N];
        a.copy_to(as);
        b.copy_to(bs);

        return ::testing::seq_eq(as, bs);
    }

    constexpr unsigned nrounds = 20u;
}

template <typename S>
struct simd_value: public ::testing::Test {};

TYPED_TEST_CASE_P(simd_value);

// Initialization and element access.
TYPED_TEST_P(simd_value, elements) {
    using simd = TypeParam;
    using scalar = typename simd::scalar_type;
    constexpr unsigned N = simd::width;

    std::minstd_rand rng(1001);

    // broadcast:
    simd a(2);
    for (unsigned i = 0; i<N; ++i) {
        EXPECT_EQ(2., a[i]);
    }

    // scalar assignment:
    a = 3;
    for (unsigned i = 0; i<N; ++i) {
        EXPECT_EQ(3, a[i]);
    }

    scalar bv[N], cv[N], dv[N];

    fill_random(bv, rng);
    fill_random(cv, rng);
    fill_random(dv, rng);

    // array initialization:
    simd b(bv);
    EXPECT_TRUE(testing::indexed_eq_n(N, bv, b));

    // array rvalue initialization:
    simd c(std::move(cv));
    EXPECT_TRUE(testing::indexed_eq_n(N, cv, c));

    // pointer initialization:
    simd d(&dv[0]);
    EXPECT_TRUE(testing::indexed_eq_n(N, dv, d));

    // copy construction:
    simd e(d);
    EXPECT_TRUE(testing::indexed_eq_n(N, dv, e));

    // copy assignment:
    b = d;
    EXPECT_TRUE(testing::indexed_eq_n(N, dv, b));
}

TYPED_TEST_P(simd_value, element_lvalue) {
    using simd = TypeParam;
    constexpr unsigned N = simd::width;

    simd a(3);
    ASSERT_GT(N, 1u);
    a[N-2] = 5;

    for (unsigned i = 0; i<N; ++i) {
        EXPECT_EQ(i==N-2? 5: 3, a[i]);
    }
}

TYPED_TEST_P(simd_value, copy_to_from) {
    using simd = TypeParam;
    using scalar = typename simd::scalar_type;
    constexpr unsigned N = simd::width;

    std::minstd_rand rng(1010);

    scalar buf1[N], buf2[N];
    fill_random(buf1, rng);
    fill_random(buf2, rng);

    simd s;
    s.copy_from(buf1);
    s.copy_to(buf2);

    EXPECT_TRUE(testing::indexed_eq_n(N, buf1, s));
    EXPECT_TRUE(testing::seq_eq(buf1, buf2));
}

TYPED_TEST_P(simd_value, arithmetic) {
    using simd = TypeParam;
    using scalar = typename simd::scalar_type;
    constexpr unsigned N = simd::width;

    std::minstd_rand rng(1002);
    scalar u[N], v[N], w[N], r[N];

    for (unsigned i = 0; i<nrounds; ++i) {
        fill_random(u, rng);
        fill_random(v, rng);
        fill_random(w, rng);

        scalar neg_u[N];
        for (unsigned i = 0; i<N; ++i) neg_u[i] = -u[i];

        scalar u_plus_v[N];
        for (unsigned i = 0; i<N; ++i) u_plus_v[i] = u[i]+v[i];

        scalar u_minus_v[N];
        for (unsigned i = 0; i<N; ++i) u_minus_v[i] = u[i]-v[i];

        scalar u_times_v[N];
        for (unsigned i = 0; i<N; ++i) u_times_v[i] = u[i]*v[i];

        scalar u_divide_v[N];
        for (unsigned i = 0; i<N; ++i) u_divide_v[i] = u[i]/v[i];

        scalar fma_u_v_w[N];
        for (unsigned i = 0; i<N; ++i) fma_u_v_w[i] = std::fma(u[i],v[i],w[i]);

        simd us(u), vs(v), ws(w);

        (-us).copy_to(r);
        EXPECT_TRUE(testing::seq_eq(neg_u, r));

        (us+vs).copy_to(r);
        EXPECT_TRUE(testing::seq_eq(u_plus_v, r));

        (us-vs).copy_to(r);
        EXPECT_TRUE(testing::seq_eq(u_minus_v, r));

        (us*vs).copy_to(r);
        EXPECT_TRUE(testing::seq_eq(u_times_v, r));

        (us/vs).copy_to(r);
#if defined(__INTEL_COMPILER)
        // icpc will by default use an approximation for scalar division,
        // and a different one for vectorized scalar division; the latter,
        // in particular, is often out by 1 ulp for normal quotients.
        //
        // Unfortunately, we can't check at compile time the floating
        // point dodginess quotient.

        if (std::is_floating_point<scalar>::value) {
            EXPECT_TRUE(testing::seq_almost_eq<scalar>(u_divide_v, r));
        }
        else {
            EXPECT_TRUE(testing::seq_eq(u_divide_v, r));
        }
#else
        EXPECT_TRUE(testing::seq_eq(u_divide_v, r));
#endif

        (fma(us, vs, ws)).copy_to(r);
        EXPECT_TRUE(testing::seq_eq(fma_u_v_w, r));
    }
}

TYPED_TEST_P(simd_value, compound_assignment) {
    using simd = TypeParam;

    simd a, b, r;

    std::minstd_rand rng(1003);
    fill_random(a, rng);
    fill_random(b, rng);

    EXPECT_TRUE(simd_eq(a+b, (r = a)+=b));
    EXPECT_TRUE(simd_eq(a-b, (r = a)-=b));
    EXPECT_TRUE(simd_eq(a*b, (r = a)*=b));
    EXPECT_TRUE(simd_eq(a/b, (r = a)/=b));
}

TYPED_TEST_P(simd_value, comparison) {
    using simd = TypeParam;
    using mask = typename simd::simd_mask;
    constexpr unsigned N = simd::width;

    std::minstd_rand rng(1004);
    std::uniform_int_distribution<> sgn(-1, 1); // -1, 0 or 1.

    for (unsigned i = 0; i<nrounds; ++i) {
        int cmp[N];
        bool test[N];
        simd a, b;

        fill_random(b, rng);

        for (unsigned j = 0; j<N; ++j) {
            cmp[j] = sgn(rng);
            a[j] = b[j]+17*cmp[j];
        }

        mask gt = a>b;
        for (unsigned j = 0; j<N; ++j) { test[j] = cmp[j]>0; }
        EXPECT_TRUE(testing::indexed_eq_n(N, test, gt));

        mask geq = a>=b;
        for (unsigned j = 0; j<N; ++j) { test[j] = cmp[j]>=0; }
        EXPECT_TRUE(testing::indexed_eq_n(N, test, geq));

        mask lt = a<b;
        for (unsigned j = 0; j<N; ++j) { test[j] = cmp[j]<0; }
        EXPECT_TRUE(testing::indexed_eq_n(N, test, lt));

        mask leq = a<=b;
        for (unsigned j = 0; j<N; ++j) { test[j] = cmp[j]<=0; }
        EXPECT_TRUE(testing::indexed_eq_n(N, test, leq));

        mask eq = a==b;
        for (unsigned j = 0; j<N; ++j) { test[j] = cmp[j]==0; }
        EXPECT_TRUE(testing::indexed_eq_n(N, test, eq));

        mask ne = a!=b;
        for (unsigned j = 0; j<N; ++j) { test[j] = cmp[j]!=0; }
        EXPECT_TRUE(testing::indexed_eq_n(N, test, ne));
    }
}

TYPED_TEST_P(simd_value, mask_elements) {
    using simd = TypeParam;
    using mask = typename simd::simd_mask;
    constexpr unsigned N = simd::width;

    std::minstd_rand rng(1005);

    // bool broadcast:
    mask a(true);
    for (unsigned i = 0; i<N; ++i) {
        EXPECT_EQ(true, a[i]);
    }

    // scalar assignment:
    mask d;
    d = false;
    for (unsigned i = 0; i<N; ++i) {
        EXPECT_EQ(false, d[i]);
    }
    d = true;
    for (unsigned i = 0; i<N; ++i) {
        EXPECT_EQ(true, d[i]);
    }

    for (unsigned i = 0; i<nrounds; ++i) {
        bool bv[N], cv[N], dv[N];

        fill_random(bv, rng);
        fill_random(cv, rng);
        fill_random(dv, rng);

        // array initialization:
        mask b(bv);
        EXPECT_TRUE(testing::indexed_eq_n(N, bv, b));

        // array rvalue initialization:
        mask c(std::move(cv));
        EXPECT_TRUE(testing::indexed_eq_n(N, cv, c));

        // pointer initialization:
        mask d(&dv[0]);
        EXPECT_TRUE(testing::indexed_eq_n(N, dv, d));

        // copy construction:
        mask e(d);
        EXPECT_TRUE(testing::indexed_eq_n(N, dv, e));

        // copy assignment:
        b = d;
        EXPECT_TRUE(testing::indexed_eq_n(N, dv, b));
    }
}

TYPED_TEST_P(simd_value, mask_element_lvalue) {
    using simd = TypeParam;
    using mask = typename simd::simd_mask;
    constexpr unsigned N = simd::width;

    std::minstd_rand rng(1006);

    for (unsigned i = 0; i<nrounds; ++i) {
        bool v[N];
        fill_random(v, rng);

        mask m(v);
        for (unsigned j = 0; j<N; ++j) {
            bool b = v[j];
            m[j] = !b;
            v[j] = !b;

            EXPECT_EQ(m[j], !b);
            EXPECT_TRUE(testing::indexed_eq_n(N, v, m));

            m[j] = b;
            v[j] = b;
            EXPECT_EQ(m[j], b);
            EXPECT_TRUE(testing::indexed_eq_n(N, v, m));
        }
    }
}

TYPED_TEST_P(simd_value, mask_copy_to_from) {
    using simd = TypeParam;
    using simd_mask = typename simd::simd_mask;
    constexpr unsigned N = simd::width;

    std::minstd_rand rng(1012);

    for (unsigned i = 0; i<nrounds; ++i) {
        bool buf1[N], buf2[N];
        fill_random(buf1, rng);
        fill_random(buf2, rng);

        simd_mask m;
        m.copy_from(buf1);
        m.copy_to(buf2);

        EXPECT_TRUE(testing::indexed_eq_n(N, buf1, m));
        EXPECT_TRUE(testing::seq_eq(buf1, buf2));
    }
}

TYPED_TEST_P(simd_value, maths) {
    // min, max, abs tests valid for both fp and int types.

    using simd = TypeParam;
    using scalar = typename simd::scalar_type;
    constexpr unsigned N = simd::width;

    std::minstd_rand rng(1013);

    for (unsigned i = 0; i<nrounds; ++i) {
        scalar a[N], b[N], test[N];
        fill_random(a, rng);
        fill_random(b, rng);

        simd as(a), bs(b);

        for (unsigned j = 0; j<N; ++j) { test[j] = std::abs(a[j]); }
        EXPECT_TRUE(testing::indexed_eq_n(N, test, abs(as)));

        for (unsigned j = 0; j<N; ++j) { test[j] = std::min(a[j], b[j]); }
        EXPECT_TRUE(testing::indexed_eq_n(N, test, min(as, bs)));

        for (unsigned j = 0; j<N; ++j) { test[j] = std::max(a[j], b[j]); }
        EXPECT_TRUE(testing::indexed_eq_n(N, test, max(as, bs)));
    }
}

REGISTER_TYPED_TEST_CASE_P(simd_value, elements, element_lvalue, copy_to_from, arithmetic, compound_assignment, comparison, mask_elements, mask_element_lvalue, mask_copy_to_from, maths);

typedef ::testing::Types<
#ifndef DEFAULT_ABI_ONLY

#ifdef __AVX__
    simd<int, 4, simd_abi::avx>,
    simd<double, 4, simd_abi::avx>,
#endif
#ifdef __AVX2__
    simd<int, 4, simd_abi::avx2>,
    simd<double, 4, simd_abi::avx2>,
#endif
#ifdef __AVX512F__
    simd<double, 8, simd_abi::avx512>,
#endif

    simd<int, 4, simd_abi::generic>,
    simd<float, 2, simd_abi::generic>,
    simd<double, 4, simd_abi::generic>,
    simd<float, 8, simd_abi::generic>,

#endif // ndef DEFAULT_ABI_ONLY

    simd<int, 4, simd_abi::default_abi>,
    simd<double, 4, simd_abi::default_abi>,
    simd<int, 8, simd_abi::default_abi>,
    simd<double, 8, simd_abi::default_abi>
> simd_test_types;

INSTANTIATE_TYPED_TEST_CASE_P(S, simd_value, simd_test_types);

// FP-only SIMD value tests (maths).

template <typename S>
struct simd_fp_value: public ::testing::Test {};

TYPED_TEST_CASE_P(simd_fp_value);

TYPED_TEST_P(simd_fp_value, fp_maths) {
    using simd = TypeParam;
    using fp = typename simd::scalar_type;
    constexpr unsigned N = simd::width;

    std::minstd_rand rng(1014);

    for (unsigned i = 0; i<nrounds; ++i) {
        fp epsilon = std::numeric_limits<fp>::epsilon();
        int min_exponent = std::numeric_limits<fp>::min_exponent;
        int max_exponent = std::numeric_limits<fp>::max_exponent;

        fp u[N], v[N], r[N];

        // Trigonometric functions (sin, cos):
        fill_random(u, rng);

        fp sin_u[N];
        for (unsigned i = 0; i<N; ++i) sin_u[i] = std::sin(u[i]);
        sin(simd(u)).copy_to(r);
        EXPECT_TRUE(testing::seq_almost_eq<fp>(sin_u, r));

        fp cos_u[N];
        for (unsigned i = 0; i<N; ++i) cos_u[i] = std::cos(u[i]);
        cos(simd(u)).copy_to(r);
        EXPECT_TRUE(testing::seq_almost_eq<fp>(cos_u, r));

        // Logarithms (natural log):
        fill_random(u, rng, -max_exponent*std::log(2.), max_exponent*std::log(2.));
        for (auto& x: u) {
            x = std::exp(x);
            // SIMD log implementation may treat subnormal as zero
            if (std::fpclassify(x)==FP_SUBNORMAL) x = 0;
        }

        fp log_u[N];
        for (unsigned i = 0; i<N; ++i) log_u[i] = std::log(u[i]);
        log(simd(u)).copy_to(r);
        EXPECT_TRUE(testing::seq_almost_eq<fp>(log_u, r));

        // Exponential functions (exp, expm1, exprelr):

        // Use max_exponent to get coverage over finite domain.
        fp exp_min_arg = min_exponent*std::log(2.);
        fp exp_max_arg = max_exponent*std::log(2.);
        fill_random(u, rng, exp_min_arg, exp_max_arg);

        fp exp_u[N];
        for (unsigned i = 0; i<N; ++i) exp_u[i] = std::exp(u[i]);
        exp(simd(u)).copy_to(r);
        EXPECT_TRUE(testing::seq_almost_eq<fp>(exp_u, r));

        fp expm1_u[N];
        for (unsigned i = 0; i<N; ++i) expm1_u[i] = std::expm1(u[i]);
        expm1(simd(u)).copy_to(r);
        EXPECT_TRUE(testing::seq_almost_eq<fp>(expm1_u, r));

        fp exprelr_u[N];
        for (unsigned i = 0; i<N; ++i) {
            exprelr_u[i] = u[i]+fp(1)==fp(1)? fp(1): u[i]/(std::exp(u[i])-fp(1));
        }
        exprelr(simd(u)).copy_to(r);
        EXPECT_TRUE(testing::seq_almost_eq<fp>(exprelr_u, r));

        // Test expm1 and exprelr with small (magnitude < fp epsilon) values.
        fill_random(u, rng, -epsilon, epsilon);
        fp expm1_u_small[N];
        for (unsigned i = 0; i<N; ++i) {
            expm1_u_small[i] = std::expm1(u[i]);
            EXPECT_NEAR(u[i], expm1_u_small[i], std::abs(4*u[i]*epsilon)); // just to confirm!
        }
        expm1(simd(u)).copy_to(r);
        EXPECT_TRUE(testing::seq_almost_eq<fp>(expm1_u_small, r));

        fp exprelr_u_small[N];
        for (unsigned i = 0; i<N; ++i) exprelr_u_small[i] = 1;
        exprelr(simd(u)).copy_to(r);
        EXPECT_TRUE(testing::seq_almost_eq<fp>(exprelr_u_small, r));

        // Test zero result for highly negative exponents.
        fill_random(u, rng, 4*exp_min_arg, 2*exp_min_arg);
        fp exp_u_very_negative[N];
        for (unsigned i = 0; i<N; ++i) exp_u_very_negative[i] = std::exp(u[i]);
        exp(simd(u)).copy_to(r);
        EXPECT_TRUE(testing::seq_almost_eq<fp>(exp_u_very_negative, r));

        // Power function:

        // Non-negative base, arbitrary exponent.
        fill_random(u, rng, 0., std::exp(1));
        fill_random(v, rng, exp_min_arg, exp_max_arg);
        fp pow_u_pos_v[N];
        for (unsigned i = 0; i<N; ++i) pow_u_pos_v[i] = std::pow(u[i], v[i]);
        pow(simd(u), simd(v)).copy_to(r);
        EXPECT_TRUE(testing::seq_almost_eq<fp>(pow_u_pos_v, r));

        // Arbitrary base, small magnitude integer exponent.
        fill_random(u, rng);
        int int_exponent[N];
        fill_random(int_exponent, rng, -2, 2);
        for (unsigned i = 0; i<N; ++i) v[i] = int_exponent[i];
        fp pow_u_v_int[N];
        for (unsigned i = 0; i<N; ++i) pow_u_v_int[i] = std::pow(u[i], v[i]);
        pow(simd(u), simd(v)).copy_to(r);
        EXPECT_TRUE(testing::seq_almost_eq<fp>(pow_u_v_int, r));
    }
}

// Check special function behaviour for specific values including
// qNAN, infinity etc.

TYPED_TEST_P(simd_fp_value, exp_special_values) {
    using simd = TypeParam;
    using fp = typename simd::scalar_type;
    constexpr unsigned N = simd::width;

    using limits = std::numeric_limits<fp>;

    constexpr fp inf = limits::infinity();
    constexpr fp eps = limits::epsilon();
    constexpr fp largest = limits::max();
    constexpr fp normal_least = limits::min();
    constexpr fp denorm_least = limits::denorm_min();
    constexpr fp qnan = limits::quiet_NaN();

    const fp exp_minarg = std::log(normal_least);
    const fp exp_maxarg = std::log(largest);

    fp values[] = { inf, -inf, eps, -eps,
                    eps/2, -eps/2, 0., -0.,
                    1., -1., 2., -2.,
                    normal_least, denorm_least, -normal_least, -denorm_least,
                    exp_minarg, exp_maxarg, qnan, -qnan };

    constexpr unsigned n_values = sizeof(values)/sizeof(fp);
    constexpr unsigned n_packed = (n_values+N-1)/N;
    fp data[n_packed][N];

    std::fill((fp *)data, (fp *)data+N*n_packed, fp(0));
    std::copy(std::begin(values), std::end(values), (fp *)data);

    for (unsigned i = 0; i<n_packed; ++i) {
        fp expected[N], result[N];
        for (unsigned j = 0; j<N; ++j) {
            expected[j] = std::exp(data[i][j]);
        }

        simd s(data[i]);
        s = exp(s);
        s.copy_to(result);

        EXPECT_TRUE(testing::seq_almost_eq<fp>(expected, result));
    }
}

TYPED_TEST_P(simd_fp_value, log_special_values) {
    using simd = TypeParam;
    using fp = typename simd::scalar_type;
    constexpr unsigned N = simd::width;

    using limits = std::numeric_limits<fp>;

    // NOTE: simd log implementations may treat subnormal
    // numbers as zero, so omit the denorm_least tests...

    constexpr fp inf = limits::infinity();
    constexpr fp eps = limits::epsilon();
    constexpr fp largest = limits::max();
    constexpr fp normal_least = limits::min();
    //constexpr fp denorm_least = limits::denorm_min();
    constexpr fp qnan = limits::quiet_NaN();

    fp values[] = { inf, -inf, eps, -eps,
                    eps/2, -eps/2, 0., -0.,
                    1., -1., 2., -2.,
                    //normal_least, denorm_least, -normal_least, -denorm_least,
                    normal_least, -normal_least,
                    qnan, -qnan, largest };

    constexpr unsigned n_values = sizeof(values)/sizeof(fp);
    constexpr unsigned n_packed = (n_values+N-1)/N;
    fp data[n_packed][N];

    std::fill((fp *)data, (fp *)data+N*n_packed, fp(0));
    std::copy(std::begin(values), std::end(values), (fp *)data);

    for (unsigned i = 0; i<n_packed; ++i) {
        fp expected[N], result[N];
        for (unsigned j = 0; j<N; ++j) {
            expected[j] = std::log(data[i][j]);
        }

        simd s(data[i]);
        s = log(s);
        s.copy_to(result);

        EXPECT_TRUE(testing::seq_almost_eq<fp>(expected, result));
    }
}

REGISTER_TYPED_TEST_CASE_P(simd_fp_value, fp_maths, exp_special_values, log_special_values);

typedef ::testing::Types<
#ifndef DEFAULT_ABI_ONLY

#ifdef __AVX__
    simd<double, 4, simd_abi::avx>,
#endif
#ifdef __AVX2__
    simd<double, 4, simd_abi::avx2>,
#endif
#ifdef __AVX512F__
    simd<double, 8, simd_abi::avx512>,
#endif

    simd<float, 2, simd_abi::generic>,
    simd<double, 4, simd_abi::generic>,
    simd<float, 8, simd_abi::generic>,

#endif // ndef DEFAULT_ABI_ONLY

    simd<double, 4, simd_abi::default_abi>,
    simd<double, 8, simd_abi::default_abi>
> simd_fp_test_types;

INSTANTIATE_TYPED_TEST_CASE_P(S, simd_fp_value, simd_fp_test_types);

// Gather/scatter tests.

template <typename A, typename B>
struct simd_and_index {
    using simd = A;
    using simd_index = B;
};

template <typename SI>
struct simd_indirect: public ::testing::Test {};

TYPED_TEST_CASE_P(simd_indirect);

TYPED_TEST_P(simd_indirect, gather) {
    using simd = typename TypeParam::simd;
    using simd_index = typename TypeParam::simd_index;

    constexpr unsigned N = simd::width;
    using scalar = typename simd::scalar_type;
    using index = typename simd_index::scalar_type;

    std::minstd_rand rng(1011);

    constexpr std::size_t buflen = 1000;

    for (unsigned i = 0; i<nrounds; ++i) {
        scalar array[buflen];
        index indirect[N];

        fill_random(array, rng);
        fill_random(indirect, rng, 0, (int)(buflen-1));

        simd s;
        s.gather(array, simd_index(indirect));

        scalar test[N];
        for (unsigned j = 0; j<N; ++j) {
            test[j] = array[indirect[j]];
        }

        EXPECT_TRUE(::testing::indexed_eq_n(N, test, s));
    }
}

TYPED_TEST_P(simd_indirect, masked_gather) {
    using simd = typename TypeParam::simd;
    using simd_index = typename TypeParam::simd_index;
    using simd_mask = typename simd::simd_mask;

    constexpr unsigned N = simd::width;
    using scalar = typename simd::scalar_type;
    using index = typename simd_index::scalar_type;

    std::minstd_rand rng(1011);

    constexpr std::size_t buflen = 1000;

    for (unsigned i = 0; i<nrounds; ++i) {
        scalar array[buflen], original[N], test[N];
        index indirect[N];
        bool mask[N];

        fill_random(array, rng);
        fill_random(original, rng);
        fill_random(indirect, rng, 0, (int)(buflen-1));
        fill_random(mask, rng);

        for (unsigned j = 0; j<N; ++j) {
            test[j] = mask[j]? array[indirect[j]]: original[j];
        }

        simd s(original);
        s.gather(array, simd_index(indirect), simd_mask(mask));

        EXPECT_TRUE(::testing::indexed_eq_n(N, test, s));
    }
}

TYPED_TEST_P(simd_indirect, scatter) {
    using simd = typename TypeParam::simd;
    using simd_index = typename TypeParam::simd_index;

    constexpr unsigned N = simd::width;
    using scalar = typename simd::scalar_type;
    using index = typename simd_index::scalar_type;

    std::minstd_rand rng(1011);

    constexpr std::size_t buflen = 1000;

    for (unsigned i = 0; i<nrounds; ++i) {
        scalar array[buflen], test[buflen], values[N];
        index indirect[N];

        fill_random(array, rng);
        fill_random(values, rng);
        fill_random(indirect, rng, 0, (int)(buflen-1));

        for (unsigned j = 0; j<buflen; ++j) {
            test[j] = array[j];
        }
        for (unsigned j = 0; j<N; ++j) {
            test[indirect[j]] = values[j];
        }

        simd s(values);
        s.scatter(array, simd_index(indirect));

        EXPECT_TRUE(::testing::indexed_eq_n(N, test, array));
    }
}

TYPED_TEST_P(simd_indirect, masked_scatter) {
    using simd = typename TypeParam::simd;
    using simd_index = typename TypeParam::simd_index;
    using simd_mask = typename simd::simd_mask;

    constexpr unsigned N = simd::width;
    using scalar = typename simd::scalar_type;
    using index = typename simd_index::scalar_type;

    std::minstd_rand rng(1011);

    constexpr std::size_t buflen = 1000;

    for (unsigned i = 0; i<nrounds; ++i) {
        scalar array[buflen], test[buflen], values[N];
        index indirect[N];
        bool mask[N];

        fill_random(array, rng);
        fill_random(values, rng);
        fill_random(indirect, rng, 0, (int)(buflen-1));
        fill_random(mask, rng);

        for (unsigned j = 0; j<buflen; ++j) {
            test[j] = array[j];
        }
        for (unsigned j = 0; j<N; ++j) {
            if (mask[j]) { test[indirect[j]] = values[j]; }
        }

        simd s(values);
        s.scatter(array, simd_index(indirect), simd_mask(mask));

        EXPECT_TRUE(::testing::indexed_eq_n(N, test, array));
    }
}


REGISTER_TYPED_TEST_CASE_P(simd_indirect, gather, masked_gather, scatter, masked_scatter);

typedef ::testing::Types<
#ifndef DEFAULT_ABI_ONLY

#ifdef __AVX__
    simd_and_index<simd<double, 4, simd_abi::avx>,
                   simd<int, 4, simd_abi::avx>>,
#endif

#ifdef __AVX2__
    simd_and_index<simd<double, 4, simd_abi::avx2>,
                   simd<int, 4, simd_abi::avx2>>,
#endif

#ifdef __AVX512F__
    simd_and_index<simd<double, 8, simd_abi::avx512>,
                   simd<int, 8, simd_abi::avx512>>,
#endif

    simd_and_index<simd<float, 4, simd_abi::generic>,
                   simd<std::int64_t, 4, simd_abi::generic>>,

    simd_and_index<simd<double, 8, simd_abi::generic>,
                   simd<unsigned, 8, simd_abi::generic>>,

#endif // ndef DEFAULT_ABI_ONLY

    simd_and_index<simd<double, 4, simd_abi::default_abi>,
                   simd<int, 4, simd_abi::default_abi>>,

    simd_and_index<simd<double, 8, simd_abi::default_abi>,
                   simd<int, 8, simd_abi::default_abi>>
> simd_indirect_test_types;

INSTANTIATE_TYPED_TEST_CASE_P(S, simd_indirect, simd_indirect_test_types);
