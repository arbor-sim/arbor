#include <random>

#include <util/simd.hpp>
#include <util/simd/avx.hpp>

#include "common.hpp"

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

        static auto u = make_udist<V>();
        for (auto& x: seq) { x = u(rng); }
    }

    template <typename Seq, typename Rng, typename B>
    void fill_random(Seq&& seq, Rng& rng, B lb, B ub) {
        using V = typename std::decay<decltype(*std::begin(seq))>::type;

        static auto u = make_udist<V>(lb, ub);
        for (auto& x: seq) { x = u(rng); }
    }

    template <typename Simd, typename Rng, typename B, typename = typename std::enable_if<is_simd<Simd>::value>::type>
    void fill_random(Simd& s, Rng& rng, B lb, B ub) {
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

    for (unsigned i = 0; i<1u; ++i) {
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


REGISTER_TYPED_TEST_CASE_P(simd_value, elements, element_lvalue, copy_to_from, arithmetic, compound_assignment, comparison, mask_elements, mask_element_lvalue, mask_copy_to_from);

typedef ::testing::Types<
#ifdef __AVX__
    simd<int, 4, simd_abi::avx>,
    simd<double, 4, simd_abi::avx>,
#endif
#ifdef __AVX2__
    simd<int, 4, simd_abi::avx2>,
    simd<double, 4, simd_abi::avx2>,
#endif

    simd<int, 4, simd_abi::generic>,
    simd<float, 2, simd_abi::generic>,
    simd<double, 4, simd_abi::generic>,
    simd<float, 8, simd_abi::generic>,

    simd<int, 4, simd_abi::default_abi>,
    simd<double, 4, simd_abi::default_abi>
> simd_test_types;

INSTANTIATE_TYPED_TEST_CASE_P(S, simd_value, simd_test_types);

// TODO: scatter, cast tests

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
#ifdef __AVX__
    simd_and_index<simd<double, 4, simd_abi::avx>,
                   simd<int, 4, simd_abi::avx>>,
#endif

#ifdef __AVX2__
    simd_and_index<simd<double, 4, simd_abi::avx2>,
                   simd<int, 4, simd_abi::avx2>>,
#endif

    simd_and_index<simd<float, 4, simd_abi::generic>,
                   simd<std::int64_t, 4, simd_abi::generic>>,

    simd_and_index<simd<double, 8, simd_abi::generic>,
                   simd<unsigned, 8, simd_abi::generic>>,

    simd_and_index<simd<double, 4, simd_abi::default_abi>,
                   simd<int, 4, simd_abi::default_abi>>

> simd_indirect_test_types;

INSTANTIATE_TYPED_TEST_CASE_P(S, simd_indirect, simd_indirect_test_types);
