#include <gtest/gtest.h>

#include <forward_list>
#include <vector>

#include <util/mergeview.hpp>

#include "common.hpp"

using namespace arb;

static auto cstr_range(const char* b) { return util::make_range(b, testing::null_terminated); }

TEST(mergeview, ctor_eq_iter) {
    using std::begin;
    using std::end;

    // Make view from sequences with the same iterator types.
    {
        int a[3] = {1, 3, 5};
        int b[3] = {2, 4, 7};

        auto merged = util::merge_view(a, b);
        EXPECT_TRUE((std::is_same<decltype(merged.begin())::pointer, int*>::value));
    }

    {
        const char c[5] = "fish";
        const char* x = "cakes";

        auto merged = util::merge_view(c, cstr_range(x));
        EXPECT_TRUE((std::is_same<decltype(merged.begin())::pointer, const char*>::value));
    }
}

TEST(mergeview, ctor_eq_compat) {
    // Make view from sequences with compatible iterator types.
    {
        int a[3] = {1, 3, 5};
        const int b[3] = {2, 4, 7};

        auto merged = util::merge_view(a, b);
        EXPECT_TRUE((std::is_same<decltype(merged.begin())::pointer, const int*>::value));
    }

    {
        const std::vector<int> a = {1, 3, 5};
        std::vector<int> b = {2, 4, 6};

        auto merged = util::merge_view(a, b);
        EXPECT_TRUE((std::is_same<decltype(merged.begin())::pointer, std::vector<int>::const_iterator>::value));
    }
}

TEST(mergeview, ctor_value_compat) {
    // Make view from sequences with compatible value types.
    int a[3] = {1, 3, 5};
    double b[3] = {2, 4, 7};

    auto merged = util::merge_view(a, b);
    EXPECT_TRUE((std::is_same<std::decay_t<decltype(*merged.begin())>, double>::value));
}

TEST(mergeview, empty) {
    std::vector<int> a, b;
    auto merged = util::merge_view(a, b);
    EXPECT_EQ(0, std::distance(merged.begin(), merged.end()));

    b = {1, 3, 4};
    merged = util::merge_view(a, b);
    EXPECT_EQ(3, std::distance(merged.begin(), merged.end()));
    EXPECT_TRUE(testing::seq_eq(merged, b));

    std::swap(a, b);
    merged = util::merge_view(a, b);
    EXPECT_EQ(3, std::distance(merged.begin(), merged.end()));
    EXPECT_TRUE(testing::seq_eq(merged, a));
}

TEST(mergeview, merge) {
    int a[] = {1, 3, 5};
    double b[] = {1., 2.5, 7.};
    double expected[] = {1., 1., 2.5, 3., 5., 7.};

    EXPECT_TRUE(testing::seq_eq(expected, util::merge_view(a, b)));
    EXPECT_TRUE(testing::seq_eq(expected, util::merge_view(b, a)));
}

TEST(mergeview, iter_ptr) {
    struct X {
        X(double v): v(v) {}
        double v;
        bool written = false;
        void write(std::vector<double>& buf) { written = true; buf.push_back(v); }

        bool operator<(X b) const { return v<b.v; }
    };

    std::vector<double> out;
    X a[] = {2., 4., 4., 6.};
    X b[] = {5., 9.};

    auto merged = util::merge_view(a, b);
    for (auto i = merged.begin(); i!=merged.end(); ++i) {
        i->write(out);
    }

    EXPECT_EQ((std::vector<double>{2., 4., 4., 5., 6., 9.}), out);
    for (auto& x: a) EXPECT_TRUE(x.written);
    for (auto& x: b) EXPECT_TRUE(x.written);
}
