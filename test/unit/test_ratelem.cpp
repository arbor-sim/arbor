#include <gtest/gtest.h>

#include <cmath>

#include "util/ratelem.hpp"
#include "util/rangeutil.hpp"

using namespace arb;
using util::rat_element;

TEST(ratelem, direct_ctor) {
    rat_element<0, 0> x00(3.5);
    EXPECT_EQ(1u, x00.size());
    EXPECT_EQ(3.5, x00[0]);

    rat_element<1, 3> x13(1.1, 2.2, 3.3, 4.4, 5.5);
    EXPECT_EQ(5u, x13.size());
    EXPECT_EQ(1.1, x13[0]);
    EXPECT_EQ(2.2, x13[1]);
    EXPECT_EQ(3.3, x13[2]);
    EXPECT_EQ(4.4, x13[3]);
    EXPECT_EQ(5.5, x13[4]);

    std::array<float, 4> x21_arr{1.25f, 1.5f, 0.5f, 2.25f};
    rat_element<2, 1> x21(x21_arr);
    EXPECT_EQ(4u, x21.size());
    EXPECT_EQ(1.25, x21[0]);
    EXPECT_EQ(1.5,  x21[1]);
    EXPECT_EQ(0.5,  x21[2]);
    EXPECT_EQ(2.25, x21[3]);

    int x20_arr[3] = {3, 2, 4};
    rat_element<2, 0> x20(x20_arr);
    EXPECT_EQ(3u, x20.size());
    EXPECT_EQ(3., x20[0]);
    EXPECT_EQ(2., x20[1]);
    EXPECT_EQ(4., x20[2]);
}

TEST(ratelem, fn_ctor) {
    auto f = [](double x) { return 1+x*x; };

    rat_element<0, 0> x00(f);
    EXPECT_EQ(1., x00[0]);

    rat_element<1, 3> x12(f);
    EXPECT_EQ(f(0.00), x12[0]);
    EXPECT_EQ(f(0.25), x12[1]);
    EXPECT_EQ(f(0.50), x12[2]);
    EXPECT_EQ(f(0.75), x12[3]);
    EXPECT_EQ(f(1.00), x12[4]);
}


TEST(ratelem, constants) {
    // (Only expect this to work for polynomial interpolators).
    auto k = [](double c) { return [c](double x) { return c; }; };

    rat_element<0, 0> k00(k(2.));
    rat_element<1, 0> k10(k(3.));
    rat_element<2, 0> k20(k(4.));
    rat_element<3, 0> k30(k(5.));

    double xs[] = {0., 0.1, 0.3, 0.5, 0.7, 0.9, 1.};
    for (auto x: xs) {
        EXPECT_DOUBLE_EQ(2., k00(x));
        EXPECT_DOUBLE_EQ(3., k10(x));
        EXPECT_DOUBLE_EQ(4., k20(x));
        EXPECT_DOUBLE_EQ(5., k30(x));
    }
}

template <unsigned p_, unsigned q_>
struct wrap_pq {
    static constexpr unsigned p = p_;
    static constexpr unsigned q = q_;
};

template <typename PQ>
struct ratelem_pq: public testing::Test {};

using pq_types = ::testing::Types<
    wrap_pq<0, 0>, wrap_pq<1, 0>, wrap_pq<2, 0>, wrap_pq<3, 0>,
    wrap_pq<0, 1>, wrap_pq<1, 1>, wrap_pq<2, 1>, wrap_pq<3, 1>,
    wrap_pq<0, 2>, wrap_pq<1, 2>, wrap_pq<2, 2>, wrap_pq<3, 2>
>;

// Compute 1+x+x^2+...+x^n
template <unsigned n>
constexpr double upoly(double x) { return x*upoly<n-1>(x)+1.; }

template <>
constexpr double upoly<0>(double x) { return 1.; }

TYPED_TEST_SUITE(ratelem_pq, pq_types);

TYPED_TEST(ratelem_pq, interpolate_monotonic) {
    constexpr unsigned p = TypeParam::p;
    constexpr unsigned q = TypeParam::q;

    // Pick f to have irreducible order p, q.
    auto f = [](double x) { return upoly<p>(x)/upoly<q>(2*x); };
    rat_element<p, q> fpq(f);

    // Check values both on and off nodes.
    for (unsigned i = 0; i<=1+p+q; ++i) {
        double x = (double)i/(1+p+q);
        EXPECT_DOUBLE_EQ(f(x), fpq(x));
    }

    for (unsigned i = 1; i<p+q; ++i) {
        if constexpr (p+q!=0) { // avoid a spurious gcc 10 divide by zero warning.
            double x = (double)i/(p+q);
            EXPECT_DOUBLE_EQ(f(x), fpq(x));
        }
    }
}

TEST(ratelem, p1q1singular) {
    // Check special case behaviour for p==1, q==1 when interpolants
    // are strictly monotonic but possibly infinite.

    auto f1 = [](double x) { return (1-x)/x; };
    rat_element<1, 1> r1(f1);

    for (unsigned i = 0; i<=4; ++i) {
        double x = (double)i/4.0;
        EXPECT_DOUBLE_EQ(f1(x), r1(x));
    }

    auto f2 = [](double x) { return x/(1-x); };
    rat_element<1, 1> r2(f2);

    for (unsigned i = 0; i<=4; ++i) {
        double x = (double)i/4.0;
        EXPECT_DOUBLE_EQ(f2(x), r2(x));
    }

    // With p==1, q==1, all infinite node values should
    // give us NaN when we try to evaluate.

    rat_element<1, 1> nope(INFINITY, INFINITY, INFINITY);
    for (unsigned i = 0; i<=4; ++i) {
        double x = (double)i/4.0;
        EXPECT_TRUE(std::isnan(nope(x)));
    }
}
