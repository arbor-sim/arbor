#include "../test/gtest.h"

#include <arbor/morph/mcable_map.hpp>
#include <arbor/morph/primitives.hpp>

#include "util/span.hpp"

using namespace arb;

TEST(mcable_map, insertion) {
    mcable_list cl{{0, 0.2, 0.8}, {1, 0, 1}, {2, 0.2, 0.4}, {2, 0.4, 0.7}, {2, 0.8, 1.}, {4, 0.2, 0.4}};
    mcable_map<int> mm;
    bool ok = false;

    int j = 10;
    for (const mcable& c: cl) {
        ok = mm.insert(c, j++);
        EXPECT_TRUE(ok);
    }
    auto expected_size = cl.size();
    EXPECT_EQ(expected_size, mm.size());

    // insertions at front or in middle:
    ok = mm.insert(mcable{0, 0.1, 0.2}, 1);
    EXPECT_TRUE(ok);
    ++expected_size;
    EXPECT_EQ(expected_size, mm.size());

    ok = mm.insert(mcable{0, 0.05, 0.2}, 1);
    EXPECT_FALSE(ok); // overlap
    EXPECT_EQ(expected_size, mm.size());

    ok = mm.insert(mcable{1, 1., 1.}, 1);
    EXPECT_TRUE(ok); // overlapping one point is fine
    ++expected_size;
    EXPECT_EQ(expected_size, mm.size());

    ok = mm.insert(mcable{3, 0., 1.}, 1);
    EXPECT_TRUE(ok);
    ++expected_size;
    EXPECT_EQ(expected_size, mm.size());

    ok = mm.insert(mcable{2, 0.7, 0.8}, 1);
    EXPECT_TRUE(ok);
    ++expected_size;
    EXPECT_EQ(expected_size, mm.size());

    ok = mm.insert(mcable{2, 0., 0.3}, 1);
    EXPECT_FALSE(ok); // overlap
    EXPECT_EQ(expected_size, mm.size());
}

TEST(mcable_map, emplace) {
    struct foo {
        foo(char a, int b): sa(a+1), sb(b+2) {}
        foo(const foo&) = delete;
        foo(foo&&) = default;

        foo& operator=(const foo&) = delete;
        foo& operator=(foo&&) = default;

        char sa;
        int sb;
    };

    mcable_map<foo> mm;
    mm.emplace(mcable{1, 0.2, 0.4}, 'a', 3);
    mm.emplace(mcable{2, 0.2, 0.4}, 'x', 5);
    ASSERT_EQ(2u, mm.size());

    EXPECT_EQ('b', mm[0].second.sa);
    EXPECT_EQ(5, mm[0].second.sb);

    EXPECT_EQ('y', mm[1].second.sa);
    EXPECT_EQ(7, mm[1].second.sb);
}

TEST(mcable_map, access) {
    mcable_list cl{{0, 0.2, 0.8}, {1, 0, 1}, {2, 0.2, 0.4}, {2, 0.4, 0.7}, {2, 0.8, 1.}, {4, 0.2, 0.4}};
    mcable_map<int> mm;
    bool ok = false;

    int k = 10;
    for (const mcable& c: cl) {
        ok = mm.insert(c, k++);
        ASSERT_TRUE(ok);
    }
    auto size = cl.size();
    ASSERT_EQ(size, mm.size());

    for (std::size_t i = 0; i<size; ++i) {
        EXPECT_EQ(cl[i], mm[i].first);
        EXPECT_EQ(int(10+i), mm[i].second);
    }

    std::size_t j = 0;
    for (const auto& el: mm) {
        EXPECT_EQ(cl[j], el.first);
        EXPECT_EQ(int(10+j), el.second);
        ++j;
    }

    j = size;
    for (auto r = mm.rbegin(); r!=mm.rend(); ++r) {
        --j;
        EXPECT_EQ(cl[j], r->first);
        EXPECT_EQ(int(10+j), r->second);
    }
}
