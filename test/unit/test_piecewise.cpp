#include <gtest/gtest.h>

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

    EXPECT_EQ(10, p[0].value);
    EXPECT_EQ( 8, p[1].value);
    EXPECT_EQ( 9, p[2].value);
    EXPECT_EQ( 4, p[3].value);

    using dp = std::pair<double, double>;
    EXPECT_EQ(dp(1.0, 1.5), p.extent(0));
    EXPECT_EQ(dp(1.5, 2.0), p.extent(1));
    EXPECT_EQ(dp(2.0, 2.5), p.extent(2));
    EXPECT_EQ(dp(2.5, 3.0), p.extent(3));

    pw_elements<int> q1(p);
    pw_elements<int> q2;
    q2 = p;
    pw_elements<int> q3(p);
    q3.assign(p.vertices(), p.values());

    EXPECT_EQ((std::vector<double>{1.0, 1.5, 2.0, 2.5, 3.0}), p.vertices());
    EXPECT_EQ((std::vector<int>{10, 8, 9, 4}), p.values());

    EXPECT_EQ(q1.vertices(), p.vertices());
    EXPECT_EQ(q2.vertices(), p.vertices());
    EXPECT_EQ(q3.vertices(), p.vertices());

    EXPECT_EQ(q1.values(), p.values());
    EXPECT_EQ(q2.values(), p.values());
    EXPECT_EQ(q3.values(), p.values());

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
    EXPECT_EQ(dp(1.0, 1.5), p.extent(0));
    EXPECT_EQ(dp(1.5, 2.0), p.extent(1));
    EXPECT_EQ(dp(2.0, 2.5), p.extent(2));
    EXPECT_EQ(dp(2.5, 3.0), p.extent(3));

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
        EXPECT_EQ(v[i], p[i].extent.first);
        EXPECT_EQ(v[i+1], p[i].extent.second);

        EXPECT_EQ(v[i], p.extent(i).first);
        EXPECT_EQ(v[i+1], p.extent(i).second);

        EXPECT_EQ(x[i], p[i].value);
        EXPECT_EQ(x[i], p.value(i));
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
    EXPECT_EQ(1., p.lower_bound());
    EXPECT_EQ(3., p.bounds().second);
    EXPECT_EQ(3., p.upper_bound());

    pw_elements<> v{{1., 1.5, 2., 2.5, 3.}};

    EXPECT_EQ(1., v.bounds().first);
    EXPECT_EQ(1., p.lower_bound());
    EXPECT_EQ(3., v.bounds().second);
    EXPECT_EQ(3., p.upper_bound());
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

TEST(piecewise, equal_range) {
    {
        pw_elements<int> p{{1, 2, 3, 4}, {10, 9, 8}};

        auto er0 = p.equal_range(0.0);
        ASSERT_EQ(er0.first, er0.second);

        auto er1 = p.equal_range(1.0);
        ASSERT_EQ(1u, er1.second-er1.first);
        EXPECT_EQ(10, er1.first->value);

        auto er2 = p.equal_range(2.0);
        ASSERT_EQ(2u, er2.second-er2.first);
        auto iter = er2.first;
        EXPECT_EQ(10, iter++->value);
        EXPECT_EQ(9, iter->value);

        auto er3_5 = p.equal_range(3.5);
        ASSERT_EQ(1u, er3_5.second-er3_5.first);
        EXPECT_EQ(8, er3_5.first->value);

        auto er4 = p.equal_range(4.0);
        ASSERT_EQ(1u, er4.second-er4.first);
        EXPECT_EQ(8, er4.first->value);

        auto er5 = p.equal_range(5.0);
        ASSERT_EQ(er5.first, er5.second);
    }

    {
        pw_elements<int> p{{1, 1, 2, 2, 2, 3, 3}, {10, 11, 12, 13, 14, 15}};

        auto er0 = p.equal_range(0.0);
        ASSERT_EQ(er0.first, er0.second);

        auto er1 = p.equal_range(1.0);
        ASSERT_EQ(2u, er1.second-er1.first);
        auto iter = er1.first;
        EXPECT_EQ(10, iter++->value);
        EXPECT_EQ(11, iter++->value);

        auto er2 = p.equal_range(2.0);
        ASSERT_EQ(4u, er2.second-er2.first);
        iter = er2.first;
        EXPECT_EQ(11, iter++->value);
        EXPECT_EQ(12, iter++->value);
        EXPECT_EQ(13, iter++->value);
        EXPECT_EQ(14, iter++->value);

        auto er3 = p.equal_range(3.0);
        ASSERT_EQ(2u, er3.second-er3.first);
        iter = er3.first;
        EXPECT_EQ(14, iter++->value);
        EXPECT_EQ(15, iter++->value);

        auto er5 = p.equal_range(5.0);
        ASSERT_EQ(er5.first, er5.second);
    }
}

TEST(piecewise, push) {
    pw_elements<int> q;
    using dp = std::pair<double, double>;

    // Need left hand side!
    EXPECT_THROW(q.push_back(3.1, 4), std::runtime_error);

    q.clear();
    q.push_back(1.1, 3.1, 4);
    q.push_back(3.1, 4.3, 5);
    EXPECT_EQ(dp(1.1, 3.1), q.extent(0));
    EXPECT_EQ(dp(3.1, 4.3), q.extent(1));
    EXPECT_EQ(4, q[0].value);
    EXPECT_EQ(5, q[1].value);

    q.push_back(7.2, 6);
    EXPECT_EQ(dp(4.3, 7.2), q.extent(2));
    EXPECT_EQ(6, q[2].value);

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
    EXPECT_EQ(dp(0.2,0.4), p.extent(1));

    // Supplied left side doesn't match current right.
    EXPECT_THROW(p.push_back(0.7, 0.9), std::runtime_error);
}

TEST(piecewise, mutate) {
    pw_elements<int> p({1., 2., 3., 3., 4., 5.}, {10, 11, 12, 13, 14});
    ASSERT_EQ(10, p.value(0));
    ASSERT_EQ(11, p.value(1));

    p.value(0) = 20;
    EXPECT_EQ((std::vector<int>{20, 11, 12, 13, 14}), p.values());

    p.front() = 30;
    EXPECT_EQ((std::vector<int>{30, 11, 12, 13, 14}), p.values());

    p[0] = 40;
    EXPECT_EQ((std::vector<int>{40, 11, 12, 13, 14}), p.values());

    for (auto&& elem: util::make_range(p.equal_range(3.))) {
        elem = 7;
    }
    EXPECT_EQ((std::vector<int>{40, 7, 7, 7, 14}), p.values());

    p(3.) = 50; // right most element intersecting 3.
    EXPECT_EQ((std::vector<int>{40, 7, 7, 50, 14}), p.values());

    p(3.).value = 60;
    EXPECT_EQ((std::vector<int>{40, 7, 7, 60, 14}), p.values());

    ASSERT_TRUE(std::is_const_v<decltype(p[0].extent)>);
}

TEST(piecewise, map) {
    double xx[5] = {1, 2.25, 3.25, 3.5, 4.};
    pw_elements<int> p(xx, (int [4]){3, 4, 5, 6});

    auto void_fn = [](auto) {};
    pw_elements<> void_expected(xx);
    EXPECT_EQ(void_expected, pw_map(p, void_fn));

    auto str_fn = [](int j) { return std::string(j, '.'); };
    pw_elements<std::string> str_expected(xx, (const char* [4]){"...", "....", ".....", "......"});
    EXPECT_EQ(str_expected, pw_map(p, str_fn));
}

TEST(piecewise, zip) {
    pw_elements<int> p03;
    p03.assign((double [3]){0., 1.5, 3.}, (int [2]){10, 11});

    pw_elements<int> p14;
    p14.assign((double [5]){1, 2.25, 3.25, 3.5, 4.}, (int [4]){3, 4, 5, 6});

    using ii = std::pair<int, int>;
    using pipi = std::pair<pw_element<int>, pw_element<int>>;

    // Zipped elements are pairs of pw_element.
    pw_elements<pipi> p03_14_pw = pw_zip(p03, p14);
    EXPECT_EQ(1., p03_14_pw.bounds().first);
    EXPECT_EQ(3., p03_14_pw.bounds().second);

    EXPECT_EQ((std::vector<double>{1., 1.5, 2.25, 3.}), p03_14_pw.vertices());
    EXPECT_EQ((std::vector<pipi>{{p03[0], p14[0]}, {p03[1], p14[0]}, {p03[1], p14[1]}}),
        p03_14_pw.values());

    // To get pairs of just the element values, use pw_zip_with (with the default
    // pw_pairify map).

    pw_elements<ii> p03_14 = pw_zip_with(p03, p14);
    EXPECT_EQ(1., p03_14.bounds().first);
    EXPECT_EQ(3., p03_14.bounds().second);

    EXPECT_EQ((std::vector<double>{1., 1.5, 2.25, 3.}), p03_14.vertices());
    EXPECT_EQ((std::vector<ii>{{10, 3}, {11, 3}, {11, 4}}), p03_14.values());

    pw_elements<ii> p14_03 = pw_zip_with(p14, p03);
    EXPECT_EQ(p03_14.vertices(), p14_03.vertices());

    std::vector<ii> flipped = util::assign_from(util::transform_view(p14_03.values(),
        [](ii p) { return ii{p.second, p.first}; }));
    EXPECT_EQ(p03_14.values(), flipped);

    pw_elements<> v03;
    v03.assign((double [3]){0., 1.5, 3.});

    EXPECT_EQ((std::vector<int>{3, 3, 4}), pw_zip_with(v03, p14).values());
    EXPECT_EQ((std::vector<int>{3, 3, 4}), pw_zip_with(p14, v03).values());

    EXPECT_EQ((std::vector<double>{1., 1.5, 2.25, 3.}), pw_zip_with(v03, p14).vertices());
    EXPECT_EQ((std::vector<double>{1., 1.5, 2.25, 3.}), pw_zip_with(p14, v03).vertices());

    auto project = [](std::pair<double, double> extent, pw_element<void>, const pw_element<int>& b) -> double {
        auto [l, r] = extent;
        double b_width = b.extent.second-b.extent.first;
        return b.value*(r-l)/b_width;
    };

    pw_elements<void> vxx; // elements cover bounds of p14
    vxx.assign((double [6]){0.2, 1.7, 1.95, 2.325, 2.45, 4.9});

    pw_elements<double> pxx = pw_zip_with(vxx, p14, project);

    double p14_sum = util::sum(util::transform_view(p14, [](auto&& v) { return v.value; }));
    double pxx_sum = util::sum(util::transform_view(pxx, [](auto&& v) { return v.value; }));
    EXPECT_DOUBLE_EQ(p14_sum, pxx_sum);
}

TEST(piecewise, zip_zero_length_elements) {
    pw_elements<int> p03a;
    p03a.assign((double [5]){0, 0, 1.5, 3, 3}, (int [4]){10, 11, 12, 13});

    pw_elements<int> p03b;
    p03b.assign((double [7]){0, 0, 0, 1, 3, 3, 3.}, (int [6]){20, 21, 22, 23, 24, 25});

    pw_elements<int> p33;
    p33.assign((double [3]){3, 3, 3}, (int [2]){30, 31});

    pw_elements<int> p14;
    p14.assign((double [3]){1, 2, 4}, (int [2]){40, 41});

    auto flip = [](auto& pairs) { for (auto& [l, r]: pairs) std::swap(l, r); };
    using ii = std::pair<int, int>;

    {
        pw_elements<ii> zz = pw_zip_with(p03a, p03b);
        EXPECT_EQ(0., zz.bounds().first);
        EXPECT_EQ(3., zz.bounds().second);

        std::vector<double> expected_vertices = {0, 0, 0, 1, 1.5, 3, 3, 3};
        std::vector<ii> expected_values = {ii(10, 20), ii(11, 21), ii(11,22), ii(11,23), ii(12, 23), ii(13,24), ii(13,25)};

        EXPECT_EQ(expected_vertices, zz.vertices());
        EXPECT_EQ(expected_values, zz.values());

        pw_elements<ii> yy = pw_zip_with(p03b, p03a);
        flip(expected_values);

        EXPECT_EQ(expected_vertices, yy.vertices());
        EXPECT_EQ(expected_values, yy.values());
    }

    {
        pw_elements<ii> zz = pw_zip_with(p03a, p33);
        EXPECT_EQ(3., zz.bounds().first);
        EXPECT_EQ(3., zz.bounds().second);

        std::vector<double> expected_vertices = {3, 3, 3};
        std::vector<ii> expected_values = {ii(12, 30), ii(13, 31)};

        EXPECT_EQ(expected_vertices, zz.vertices());
        EXPECT_EQ(expected_values, zz.values());

        pw_elements<ii> yy = pw_zip_with(p33, p03a);
        flip(expected_values);

        EXPECT_EQ(expected_vertices, yy.vertices());
        EXPECT_EQ(expected_values, yy.values());
    }

    {
        pw_elements<ii> zz = pw_zip_with(p03a, p14);
        EXPECT_EQ(1., zz.bounds().first);
        EXPECT_EQ(3., zz.bounds().second);

        std::vector<double> expected_vertices = {1, 1.5, 2, 3, 3};
        std::vector<ii> expected_values = {ii(11, 40), ii(12, 40), ii(12, 41), ii(13, 41)};

        EXPECT_EQ(expected_vertices, zz.vertices());
        EXPECT_EQ(expected_values, zz.values());

        pw_elements<ii> yy = pw_zip_with(p14, p03a);
        flip(expected_values);

        EXPECT_EQ(expected_vertices, yy.vertices());
        EXPECT_EQ(expected_values, yy.values());
    }

    {
        // Check void version too!
        pw_elements<> v03a(p03a), v03b(p03b);
        pw_elements<> zz = pw_zip_with(v03a, v03b);
        EXPECT_EQ(0., zz.bounds().first);
        EXPECT_EQ(3., zz.bounds().second);

        std::vector<double> expected_vertices = {0, 0, 0, 1, 1.5, 3, 3, 3};
        EXPECT_EQ(expected_vertices, zz.vertices());

        pw_elements<> yy = pw_zip_with(v03b, v03a);
        EXPECT_EQ(expected_vertices, yy.vertices());
    }
}
