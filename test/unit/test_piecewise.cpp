#include "../gtest.h"

#include <list>

#include "util/piecewise.hpp"
#include "util/rangeutil.hpp"

using namespace arb;
using util::pw_elements;
using util::pw_element;
using util::pw_npos;

TEST(piecewise, eq) {
    pw_elements<int> p1((double[3]){1., 1.5, 2.}, (int[2]){3, 4});
    pw_elements<int> p2(p1);
    EXPECT_EQ(p2, p1);

    pw_elements<int> p3((double[3]){1., 1.7, 2.}, (int[2]){3, 4});
    EXPECT_NE(p3, p1);

    pw_elements<int> p4((double[3]){1., 1.5, 2.}, (int[2]){3, 5});
    EXPECT_NE(p4, p1);

    pw_elements<int> p5(p1);
    p5.push_back(2., 7);
    EXPECT_NE(p5, p1);

    pw_elements<> v1((double[3]){1., 1.5, 2.});
    pw_elements<> v2(v1);
    EXPECT_EQ(v2, v1);

    pw_elements<> v3((double[3]){1., 1.7, 2.});
    EXPECT_NE(v3, v1);

    pw_elements<> v5(v1);
    v5.push_back(2.);
    EXPECT_NE(v5, v1);
}

TEST(piecewise, generic_ctors) {
    pw_elements<int> p1({1., 1.5, 2.}, {3, 4});
    pw_elements<int> p2(std::list<float>{1.f, 1.5f, 2.f}, std::list<long>{3, 4});

    EXPECT_EQ(p1, p2);

    pw_elements<> v1(p1);
    EXPECT_EQ((std::vector<double>{1., 1.5, 2.}), v1.vertices());

    pw_elements<> v3(std::list<float>{1.f, 1.5f, 2.f});
    EXPECT_EQ(v1, v3);

    EXPECT_THROW(pw_elements<>({2.}), std::runtime_error);
    EXPECT_THROW(pw_elements<int>({2.}, {1}), std::runtime_error);
    EXPECT_THROW(pw_elements<int>({1., 2.}, {}), std::runtime_error);
    EXPECT_THROW(pw_elements<int>({1., 2.}, {1, 3}), std::runtime_error);
}

TEST(piecewise, size) {
    pw_elements<int> p1;
    EXPECT_EQ(0u, p1.size());
    EXPECT_TRUE(p1.empty());

    pw_elements<int> p2({1., 2.}, {1});
    EXPECT_EQ(1u, p2.size());
    EXPECT_FALSE(p2.empty());

    pw_elements<> v1;
    EXPECT_EQ(0u, v1.size());
    EXPECT_TRUE(v1.empty());

    pw_elements<> v2({1., 2.});
    EXPECT_EQ(1u, v2.size());
    EXPECT_FALSE(v2.empty());
}

TEST(piecewise, assign) {
    pw_elements<int> p;

    double v[5] = {1., 1.5, 2., 2.5, 3.};
    int x[4] = {10, 8, 9, 4};
    p.assign(v, x);

    ASSERT_EQ(4u, p.size());

    EXPECT_EQ(10, p[0].second);
    EXPECT_EQ( 8, p[1].second);
    EXPECT_EQ( 9, p[2].second);
    EXPECT_EQ( 4, p[3].second);

    using dp = std::pair<double, double>;
    EXPECT_EQ(dp(1.0, 1.5), p.interval(0));
    EXPECT_EQ(dp(1.5, 2.0), p.interval(1));
    EXPECT_EQ(dp(2.0, 2.5), p.interval(2));
    EXPECT_EQ(dp(2.5, 3.0), p.interval(3));

    pw_elements<int> q1(p);
    pw_elements<int> q2;
    q2 = p;
    pw_elements<int> q3(p);
    q3.assign(p.vertices(), p.elements());

    EXPECT_EQ((std::vector<double>{1.0, 1.5, 2.0, 2.5, 3.0}), p.vertices());
    EXPECT_EQ((std::vector<int>{10, 8, 9, 4}), p.elements());

    EXPECT_EQ(q1.vertices(), p.vertices());
    EXPECT_EQ(q2.vertices(), p.vertices());
    EXPECT_EQ(q3.vertices(), p.vertices());

    EXPECT_EQ(q1.elements(), p.elements());
    EXPECT_EQ(q2.elements(), p.elements());
    EXPECT_EQ(q3.elements(), p.elements());

    q3.assign({}, {});
    EXPECT_TRUE(q3.empty());

    q3.assign({1., 2.}, {3});
    pw_elements<int> q4(q3);
    ASSERT_EQ(q3, q4);

    // bad assign should throw but preserve value
    ASSERT_THROW(q4.assign({3.}, {}), std::runtime_error);
    EXPECT_EQ(q3, q4);
}

TEST(piecewise, assign_void) {
    pw_elements<> p;

    double v[5] = {1., 1.5, 2., 2.5, 3.};
    p.assign(v);

    ASSERT_EQ(4u, p.size());

    using dp = std::pair<double, double>;
    EXPECT_EQ(dp(1.0, 1.5), p.interval(0));
    EXPECT_EQ(dp(1.5, 2.0), p.interval(1));
    EXPECT_EQ(dp(2.0, 2.5), p.interval(2));
    EXPECT_EQ(dp(2.5, 3.0), p.interval(3));

    pw_elements<> q1(p);
    pw_elements<> q2;
    q2 = p;
    pw_elements<> q3(p);
    q3.assign(p.vertices());

    EXPECT_EQ((std::vector<double>{1.0, 1.5, 2.0, 2.5, 3.0}), p.vertices());

    EXPECT_EQ(q1.vertices(), p.vertices());
    EXPECT_EQ(q2.vertices(), p.vertices());
    EXPECT_EQ(q3.vertices(), p.vertices());

    q3.assign({});
    EXPECT_TRUE(q3.empty());

    q3.assign({1., 2.});
    pw_elements<> q4(q3);
    ASSERT_EQ(q3, q4);

    // bad assign should throw but preserve value
    ASSERT_THROW(q4.assign({3.}), std::runtime_error);
    EXPECT_EQ(q3, q4);
}

TEST(piecewise, access) {
    pw_elements<int> p;

    double v[5] = {1., 1.5, 2., 2.5, 3.};
    int x[4] = {10, 8, 9, 4};
    p.assign(v, x);

    for (unsigned i = 0; i<4; ++i) {
        EXPECT_EQ(v[i], p[i].first.first);
        EXPECT_EQ(v[i+1], p[i].first.second);

        EXPECT_EQ(v[i], p.interval(i).first);
        EXPECT_EQ(v[i+1], p.interval(i).second);

        EXPECT_EQ(x[i], p[i].second);
        EXPECT_EQ(x[i], p.element(i));
    }

    EXPECT_EQ(p[0], p.front());
    EXPECT_EQ(p[3], p.back());

    unsigned j = 0;
    for (auto entry: p) {
        EXPECT_EQ(p[j++], entry);
    }
}

TEST(piecewise, bounds) {
    pw_elements<int> p{{1., 1.5, 2., 2.5, 3.}, {10, 8, 9, 4}};

    EXPECT_EQ(1., p.bounds().first);
    EXPECT_EQ(3., p.bounds().second);

    pw_elements<> v{{1., 1.5, 2., 2.5, 3.}};

    EXPECT_EQ(1., v.bounds().first);
    EXPECT_EQ(3., v.bounds().second);
}

TEST(piecewise, index_of) {
    pw_elements<int> p{{1., 1.5, 2., 2.5, 3.}, {10, 8, 9, 4}};

    EXPECT_EQ(pw_npos, p.index_of(0.3));
    EXPECT_EQ(0u, p.index_of(1.));
    EXPECT_EQ(0u, p.index_of(1.1));
    EXPECT_EQ(1u, p.index_of(1.5));
    EXPECT_EQ(1u, p.index_of(1.6));
    EXPECT_EQ(2u, p.index_of(2));
    EXPECT_EQ(3u, p.index_of(2.9));
    EXPECT_EQ(3u, p.index_of(3));
    EXPECT_EQ(pw_npos, p.index_of(3.1));

    pw_elements<> v(p);

    EXPECT_EQ(pw_npos, v.index_of(0.3));
    EXPECT_EQ(0u, v.index_of(1.));
    EXPECT_EQ(0u, v.index_of(1.1));
    EXPECT_EQ(1u, v.index_of(1.5));
    EXPECT_EQ(1u, v.index_of(1.6));
    EXPECT_EQ(2u, v.index_of(2));
    EXPECT_EQ(3u, v.index_of(2.9));
    EXPECT_EQ(3u, v.index_of(3));
    EXPECT_EQ(pw_npos, v.index_of(3.1));

    pw_elements<int> p0;
    pw_elements<> v0;

    EXPECT_EQ(pw_npos, p0.index_of(0.));
    EXPECT_EQ(pw_npos, v0.index_of(0.));
}

TEST(piecewise, push) {
    pw_elements<int> q;
    using dp = std::pair<double, double>;

    // Need left hand side!
    EXPECT_THROW(q.push_back(3.1, 4), std::runtime_error);

    q.clear();
    q.push_back(1.1, 3.1, 4);
    q.push_back(3.1, 4.3, 5);
    EXPECT_EQ(dp(1.1, 3.1), q.interval(0));
    EXPECT_EQ(dp(3.1, 4.3), q.interval(1));
    EXPECT_EQ(4, q[0].second);
    EXPECT_EQ(5, q[1].second);

    q.push_back(7.2, 6);
    EXPECT_EQ(dp(4.3, 7.2), q.interval(2));
    EXPECT_EQ(6, q[2].second);

    // Supplied left side doesn't match current right.
    EXPECT_THROW(q.push_back(7.4, 9.1, 7), std::runtime_error);
}

TEST(piecewise, push_void) {
    pw_elements<> p;
    using dp = std::pair<double, double>;

    // Need left hand side!
    EXPECT_THROW(p.push_back(3.1), std::runtime_error);

    p.clear();
    p.push_back(0.1, 0.2);
    p.push_back(0.2, 0.4);
    p.push_back(0.5);

    EXPECT_EQ(3u, p.size());
    EXPECT_EQ((std::vector<double>{0.1, 0.2, 0.4, 0.5}), p.vertices());
    EXPECT_EQ(dp(0.2,0.4), p.interval(1));

    // Supplied left side doesn't match current right.
    EXPECT_THROW(p.push_back(0.7, 0.9), std::runtime_error);
}

TEST(piecewise, zip) {
    pw_elements<int> p03;
    p03.assign((double [3]){0., 1.5, 3.}, (int [2]){10, 11});

    pw_elements<int> p14;
    p14.assign((double [5]){1, 2.25, 3., 3.5, 4.}, (int [4]){3, 4, 5, 6});

    using ii = std::pair<int, int>;
    pw_elements<ii> p03_14 = zip(p03, p14);
    EXPECT_EQ(1., p03_14.bounds().first);
    EXPECT_EQ(3., p03_14.bounds().second);

    EXPECT_EQ((std::vector<double>{1., 1.5, 2.25, 3.}), p03_14.vertices());
    EXPECT_EQ((std::vector<ii>{ii(10, 3), ii(11, 3), ii(11, 4)}), p03_14.elements());

    pw_elements<ii> p14_03 = zip(p14, p03);
    EXPECT_EQ(p03_14.vertices(), p14_03.vertices());

    std::vector<ii> flipped = util::assign_from(util::transform_view(p14_03.elements(),
        [](ii p) { return ii{p.second, p.first}; }));
    EXPECT_EQ(p03_14.elements(), flipped);

    pw_elements<> v03;
    v03.assign((double [3]){0., 1.5, 3.});

    EXPECT_EQ((std::vector<int>{3, 3, 4}), zip(v03, p14).elements());
    EXPECT_EQ((std::vector<int>{3, 3, 4}), zip(p14, v03).elements());

    EXPECT_EQ((std::vector<double>{1., 1.5, 2.25, 3.}), zip(v03, p14).vertices());
    EXPECT_EQ((std::vector<double>{1., 1.5, 2.25, 3.}), zip(p14, v03).vertices());

    auto project = [](double l, double r, pw_element<void>, const pw_element<int>& b) -> double {
        double b_width = b.first.second-b.first.first;
        return b.second*(r-l)/b_width;
    };

    pw_elements<void> vxx; // elements cover bounds of p14
    vxx.assign((double [6]){0.2, 1.7, 1.95, 2.325, 2.45, 4.9});

    pw_elements<double> pxx = zip(vxx, p14, project);
    double p14_sum = util::sum(util::transform_view(p14, [](auto v) { return v.second; }));
    double pxx_sum = util::sum(util::transform_view(pxx, [](auto v) { return v.second; }));
    EXPECT_DOUBLE_EQ(p14_sum, pxx_sum);

}

TEST(piecewise, zip_void) {
    pw_elements<> p03;
    p03.assign((double [3]){0., 1.5, 3.});

    pw_elements<> p14;
    p14.assign((double [5]){1, 2.25, 3., 3.5, 4.});

    EXPECT_EQ((std::vector<double>{1., 1.5, 2.25, 3.}), zip(p03, p14).vertices());
    EXPECT_EQ((std::vector<double>{1., 1.5, 2.25, 3.}), zip(p14, p03).vertices());
}
