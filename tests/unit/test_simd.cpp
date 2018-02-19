#include <random>

#include <util/simd.hpp>
#include <util/simd/avx.hpp>

#include "common.hpp"

using namespace arb;

template <typename S>
struct simdfp: public ::testing::Test {};

TYPED_TEST_CASE_P(simdfp);

// Initialization and element access.

TYPED_TEST_P(simdfp, elements) {
    using simdfp = TypeParam;
    using fp = typename simdfp::scalar_type;
    constexpr unsigned N = simdfp::width;

    // broadcast:
    simdfp a(2);
    for (unsigned i = 0; i<N; ++i) {
        EXPECT_EQ(2., a[i]);
    }

    // array initialization:
    fp bv[N];
    for (unsigned i = 0; i<N; ++i) {
        bv[i] = 1.5+i;
    }
    simdfp b(bv);
    for (unsigned i = 0; i<N; ++i) {
        EXPECT_EQ(1.5+i, b[i]);
    }

    // array rvalue initialization:
    fp cv_rvalue[N];
    for (unsigned i = 0; i<N; ++i) {
        cv_rvalue[i] = 3.5-i;
    }
    simdfp c(std::move(cv_rvalue));
    for (unsigned i = 0; i<N; ++i) {
        EXPECT_EQ(3.5-i, c[i]);
    }

    // scalar assignment:
    simdfp d(bv);
    d = 3.;
    for (unsigned i = 0; i<N; ++i) {
        EXPECT_EQ(3., d[i]);
    }

    // pointer initialization:
    fp ev[N];
    for (unsigned i = 0; i<N; ++i) {
        ev[i] = 7.5-i;
    }
    simdfp e(&ev[0]);
    for (unsigned i = 0; i<N; ++i) {
        EXPECT_EQ(7.5-i, e[i]);
    }

    // copy construction:
    simdfp f(e);
    for (unsigned i = 0; i<N; ++i) {
        EXPECT_EQ(7.5-i, f[i]);
    }

    // copy assignment:
    f = c;
    for (unsigned i = 0; i<N; ++i) {
        EXPECT_EQ(3.5-i, f[i]);
    }

}

TYPED_TEST_P(simdfp, element_lvalue) {
    using simdfp = TypeParam;
    constexpr unsigned N = simdfp::width;

    simdfp a(3);
    ASSERT_GT(N, 1u);
    a[N-2] = 0.25;

    for (unsigned i = 0; i<N; ++i) {
        EXPECT_EQ(i==N-2? 0.25: 3., a[i]);
    }
}

TYPED_TEST_P(simdfp, copy_to_from) {
    using simdfp = TypeParam;
    using fp = typename simdfp::scalar_type;
    constexpr unsigned N = simdfp::width;

    fp buf1[N], buf2[N];
    for (unsigned i = 0; i<N; ++i) {
        buf1[i] = i*0.25f+23.f;
        buf2[i] = -1;
    }

    simdfp s(2.);
    s.copy_from(buf1);
    s.copy_to(buf2);

    for (unsigned i = 0; i<N; ++i) {
        fp v = i*0.25f+23.f;
        EXPECT_EQ(v, s[i]);
        EXPECT_EQ(v, buf2[i]);
    }
}

// TODO: gather scatter test

namespace {
    template <typename FP, unsigned N>
    void fill_random(FP (&a)[N]) {
        static std::minstd_rand R(12345);
        static std::uniform_real_distribution<FP> U(-1., 1.);

        for (unsigned i = 0; i<N; ++i) {
            FP v = U(R);
            while (!v) v = U(R);
            a[i] = v;
        }
    }

    template <typename Simd>
    ::testing::AssertionResult simd_eq(Simd a, Simd b) {
        constexpr unsigned N = Simd::width;
        using fp = typename Simd::scalar_type;

        fp as[N], bs[N];
        a.copy_to(as);
        b.copy_to(bs);

        return ::testing::seq_eq(as, bs);
    }
}

TYPED_TEST_P(simdfp, arithmetic) {
    using simdfp = TypeParam;
    using fp = typename simdfp::scalar_type;
    constexpr unsigned N = simdfp::width;

    fp u[N], v[N], w[N], r[N];

    for (unsigned i = 0; i<20u; ++i) {
        fill_random(u);
        fill_random(v);
        fill_random(w);

        fp u_plus_v[N];
        for (unsigned i = 0; i<N; ++i) u_plus_v[i] = u[i]+v[i];

        fp u_minus_v[N];
        for (unsigned i = 0; i<N; ++i) u_minus_v[i] = u[i]-v[i];

        fp u_times_v[N];
        for (unsigned i = 0; i<N; ++i) u_times_v[i] = u[i]*v[i];

        fp u_divide_v[N];
        for (unsigned i = 0; i<N; ++i) u_divide_v[i] = u[i]/v[i];

        fp fma_u_v_w[N];
        for (unsigned i = 0; i<N; ++i) fma_u_v_w[i] = std::fma(u[i],v[i],w[i]);

        simdfp us(u), vs(v), ws(w);

        (us+vs).copy_to(r);
        EXPECT_TRUE(testing::seq_eq(u_plus_v, r));

        (us-vs).copy_to(r);
        EXPECT_TRUE(testing::seq_eq(u_minus_v, r));

        (us*vs).copy_to(r);
        EXPECT_TRUE(testing::seq_eq(u_times_v, r));

        (us/vs).copy_to(r);
        EXPECT_TRUE(testing::seq_eq(u_divide_v, r));

        (fma(us, vs, ws)).copy_to(r);
        EXPECT_TRUE(testing::seq_eq(fma_u_v_w, r));
    }
}

TYPED_TEST_P(simdfp, compound_assignment) {
    using simdfp = TypeParam;
    using fp = typename simdfp::scalar_type;
    constexpr unsigned N = simdfp::width;

    fp u[N], v[N], w[N];

    fill_random(u);
    fill_random(v);
    fill_random(w);

    simdfp a(u), b(v), c(w), r;

    EXPECT_TRUE(simd_eq(a+b, (r = a)+=b));
    EXPECT_TRUE(simd_eq(a-b, (r = a)-=b));
    EXPECT_TRUE(simd_eq(a*b, (r = a)*=b));
    EXPECT_TRUE(simd_eq(a/b, (r = a)/=b));
}

REGISTER_TYPED_TEST_CASE_P(simdfp, elements, element_lvalue, copy_to_from, arithmetic, compound_assignment);

typedef ::testing::Types<
    simd<float, 2, simd_abi::generic>,
    simd<double, 4, simd_abi::generic>,
#ifdef __AVX__
    simd<double, 4, simd_abi::avx>,
#endif
#ifdef __AVX2__
    simd<double, 4, simd_abi::avx2>,
#endif
    simd<double, 4, simd_abi::default_abi>
> simdfp_test_types;

INSTANTIATE_TYPED_TEST_CASE_P(simdfp_tests, simdfp, simdfp_test_types);

