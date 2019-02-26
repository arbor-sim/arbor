#include <cstdio>
#include <memory>
#include <string>

#include <util/strprintf.hpp>

#include "../gtest.h"
#include "common.hpp"

using namespace std::string_literals;
using namespace arb::util;

TEST(strprintf, simple) {
    char buf[200];

    const char* fmt1 = " %% %04d % 3.2f %#016x %c";
    sprintf(buf, fmt1, 3, 7.1e-3, 0x1234ul, 'x');
    auto result = strprintf(fmt1, 3, 7.1e-3, 0x1234ul, 'x');
    EXPECT_EQ(std::string(buf), result);

    const char* fmt2 = "%5s %3s";
    sprintf(buf, fmt2, "bear", "pear");
    result = strprintf(fmt2, "bear", "pear");
    EXPECT_EQ(std::string(buf), result);
}

TEST(strprintf, longstr) {
    std::string aaaa(2000, 'a');
    std::string bbbb(2000, 'b');
    std::string x;

    x = aaaa;
    ASSERT_EQ(x, strprintf("%s", x.c_str()));

    x += bbbb+x;
    ASSERT_EQ(x, strprintf("%s", x.c_str()));

    x += bbbb+x;
    ASSERT_EQ(x, strprintf("%s", x.c_str()));

    x += bbbb+x;
    ASSERT_EQ(x, strprintf("%s", x.c_str()));

    x += bbbb+x;
    ASSERT_EQ(x, strprintf("%s", x.c_str()));
}

TEST(strprintf, wrappers) {
    // can we print strings and smart-pointers directly?

    char buf[200];

    auto uptr = std::unique_ptr<int>{new int(17)};
    sprintf(buf, "uptr %p", uptr.get());

    EXPECT_EQ(std::string(buf), strprintf("uptr %p", uptr));

    auto sptr = std::shared_ptr<double>{new double(19.)};
    sprintf(buf, "sptr %p", sptr.get());

    EXPECT_EQ(std::string(buf), strprintf("sptr %p", sptr));

    EXPECT_EQ("fish"s, strprintf("fi%s", "sh"s));
}

