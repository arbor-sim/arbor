#include <random>

#include <util/simd.hpp>
#include <util/simd/avx.hpp>

#include "common.hpp"

using namespace arb;

namespace {
    template <typename FP, unsigned N, typename Rng>
    void fill_random_nz(FP (&a)[N], Rng& rng) {
        static std::uniform_real_distribution<FP> U(-1., 1.);

        for (auto& x: a) {
            do { x = U(rng); } while (!x);
        }
    }

    template <typename Simd, typename Rng, typename = typename std::enable_if<is_simd<Simd>::value>::type>
    void fill_random_nz(Simd& s, Rng& rng) {
        using fp = typename Simd::scalar_type;
        constexpr unsigned N = Simd::width;

        fp v[N];
        fill_random_nz(v, rng);
        s.copy_from(v);
    }

    template <unsigned N, typename Rng>
    void fill_random_bool(bool (&a)[N], Rng& rng) {
        static std::uniform_int_distribution<> U(0, 1);

        for (auto& x: a) x = U(rng);
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
}

template <typename S>
struct simdfp: public ::testing::Test {};

TYPED_TEST_CASE_P(simdfp);

// Initialization and element access.

TYPED_TEST_P(simdfp, elements) {
    using simdfp = TypeParam;
    using fp = typename simdfp::scalar_type;
    constexpr unsigned N = simdfp::width;

    std::minstd_rand rng(12345);

    // broadcast:
    simdfp a(2);
    for (unsigned i = 0; i<N; ++i) {
        EXPECT_EQ(2., a[i]);
    }

    // scalar assignment:
    a = 3.;
    for (unsigned i = 0; i<N; ++i) {
        EXPECT_EQ(3., a[i]);
    }

    fp bv[N], cv[N], dv[N];

    fill_random_nz(bv, rng);
    fill_random_nz(cv, rng);
    fill_random_nz(dv, rng);

    // array initialization:
    simdfp b(bv);
    EXPECT_TRUE(testing::indexed_eq_n(N, bv, b));

    // array rvalue initialization:
    simdfp c(std::move(cv));
    EXPECT_TRUE(testing::indexed_eq_n(N, cv, c));

    // pointer initialization:
    simdfp d(&dv[0]);
    EXPECT_TRUE(testing::indexed_eq_n(N, dv, d));

    // copy construction:
    simdfp e(d);
    EXPECT_TRUE(testing::indexed_eq_n(N, dv, e));

    // copy assignment:
    b = d;
    EXPECT_TRUE(testing::indexed_eq_n(N, dv, b));
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


TYPED_TEST_P(simdfp, arithmetic) {
    using simdfp = TypeParam;
    using fp = typename simdfp::scalar_type;
    constexpr unsigned N = simdfp::width;

    std::minstd_rand rng(12345);
    fp u[N], v[N], w[N], r[N];

    for (unsigned i = 0; i<20u; ++i) {
        fill_random_nz(u, rng);
        fill_random_nz(v, rng);
        fill_random_nz(w, rng);

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

    simdfp a, b, r;

    std::minstd_rand rng(23456);
    fill_random_nz(a, rng);
    fill_random_nz(b, rng);

    EXPECT_TRUE(simd_eq(a+b, (r = a)+=b));
    EXPECT_TRUE(simd_eq(a-b, (r = a)-=b));
    EXPECT_TRUE(simd_eq(a*b, (r = a)*=b));
    EXPECT_TRUE(simd_eq(a/b, (r = a)/=b));
}

TYPED_TEST_P(simdfp, comparison) {
    using simdfp = TypeParam;
    using mask = typename simdfp::simd_mask;
    constexpr unsigned N = simdfp::width;

    std::minstd_rand rng(34567);
    std::uniform_int_distribution<> sgn(-1, 1); // -1, 0 or 1.

    for (unsigned i = 0; i<20u; ++i) {
        int cmp[N];
        bool test[N];
        simdfp a, b;

        fill_random_nz(b, rng);

        for (unsigned j = 0; j<N; ++j) {
            cmp[j] = sgn(rng);
            a[j] = b[j]+0.1*cmp[j];
        }

        mask gt = a>b;
        for (unsigned j = 0; j<N; ++j) {
            test[j] = cmp[j]>0;
            EXPECT_EQ(test[j], gt[j]);
        }

        mask geq = a>=b;
        for (unsigned j = 0; j<N; ++j) {
            test[j] = cmp[j]>=0;
            EXPECT_EQ(test[j], geq[j]);
        }

        mask lt = a<b;
        for (unsigned j = 0; j<N; ++j) {
            test[j] = cmp[j]<0;
            EXPECT_EQ(test[j], lt[j]);
        }

        mask leq = a<=b;
        for (unsigned j = 0; j<N; ++j) {
            test[j] = cmp[j]<=0;
            EXPECT_EQ(test[j], leq[j]);
        }

        mask eq = a==b;
        for (unsigned j = 0; j<N; ++j) {
            test[j] = cmp[j]==0;
            EXPECT_EQ(test[j], eq[j]);
        }

        mask ne = a!=b;
        for (unsigned j = 0; j<N; ++j) {
            test[j] = cmp[j]!=0;
            EXPECT_EQ(test[j], ne[j]);
        }
    }
}

TYPED_TEST_P(simdfp, mask_elements) {
    using simdfp = TypeParam;
    using mask = typename simdfp::simd_mask;
    constexpr unsigned N = simdfp::width;

    std::minstd_rand rng(12345);

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

    for (unsigned i = 0; i<20u; ++i) {
        bool bv[N], cv[N], dv[N];

        fill_random_bool(bv, rng);
        fill_random_bool(cv, rng);
        fill_random_bool(dv, rng);

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

REGISTER_TYPED_TEST_CASE_P(simdfp, elements, element_lvalue, copy_to_from, arithmetic, compound_assignment, comparison, mask_elements);

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

