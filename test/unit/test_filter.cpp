#include <cstring>
#include <list>

#include <util/range.hpp>
#include <util/rangeutil.hpp>
#include <util/filter.hpp>
#include <util/transform.hpp>

#include "../gtest.h"

#include "common.hpp"

using namespace arb;
using util::filter;
using util::assign;
using util::canonical_view;

util::range<const char*, testing::null_terminated_t>
cstring(const char* s) {
    return {s, testing::null_terminated};
}

util::range<char*, testing::null_terminated_t>
cstring(char* s) {
    return {s, testing::null_terminated};
}

TEST(filter, const_regular) {
    std::list<int> ten = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    std::list<int> odd = {1, 3, 5, 7, 9};
    std::list<int> even = {2, 4, 6, 8, 10};
    std::list<int> check;

    assign(check, filter(ten, [](int i) { return i%2!=0; }));
    EXPECT_EQ(odd, check);

    assign(check, filter(ten, [](int i) { return i%2==0; }));
    EXPECT_EQ(even, check);
}

TEST(filter, const_sentinel) {
    auto xyzzy = cstring("xyzzy");
    std::string check;

    assign(check, filter(xyzzy, [](char c) { return c!='y'; }));
    EXPECT_EQ(check, "xzz");

    assign(check, filter(xyzzy, [](char c) { return c!='x'; }));
    EXPECT_EQ(check, "yzzy");
}

TEST(filter, modify_regular) {
    int nums[] = {1, -2, 3, -4, -5};

    for (auto& n: filter(nums, [](int i) { return i<0; })) {
        n *= n;
    }

    int check[] = {1, 4, 3, 16, 25};
    for (unsigned i = 0; i<5; ++i) {
        EXPECT_EQ(check[i], nums[i]);
    }
}

TEST(filter, modify_sentinel) {
    char xoxo[] = "xoxo";
    for (auto& c: canonical_view(filter(cstring(xoxo), [](char c) { return c=='o'; }))) {
        c = 'q';
    }

    EXPECT_TRUE(!std::strcmp(xoxo, "xqxq"));
}


TEST(filter, laziness) {
    std::list<int> ten = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    int derefs = 0;
    auto count_derefs = [&derefs](int i) { ++derefs; return i; };

    auto odd = filter(util::transform_view(ten, count_derefs),
        [](int i) { return i%2!=0; });

    auto iter = std::begin(odd);

    EXPECT_EQ(0, derefs);

    EXPECT_EQ(1, *iter);
    EXPECT_EQ(2, derefs); // one for check, one for operator*()
    ++iter; // does not need to scan ahead
    EXPECT_EQ(2, derefs);

    ++iter; // now it does need to scan ahead, testing values 2 and 3.
    EXPECT_EQ(4, derefs);
    EXPECT_EQ(5, *iter);
}
