#include "gtest.h"

#include <s_expr.hpp>


TEST(s_expr, identifier) {
    EXPECT_TRUE(pyarb::test_identifier("foo"));
    EXPECT_TRUE(pyarb::test_identifier("f1"));
    EXPECT_TRUE(pyarb::test_identifier("f_"));
    EXPECT_TRUE(pyarb::test_identifier("f_1__"));
    EXPECT_TRUE(pyarb::test_identifier("A_1__"));

    EXPECT_FALSE(pyarb::test_identifier("_foobar"));
    EXPECT_FALSE(pyarb::test_identifier("2dogs"));
    EXPECT_FALSE(pyarb::test_identifier("1"));
    EXPECT_FALSE(pyarb::test_identifier("_"));
    EXPECT_FALSE(pyarb::test_identifier(""));
    EXPECT_FALSE(pyarb::test_identifier(" foo"));
    EXPECT_FALSE(pyarb::test_identifier("foo "));
    EXPECT_FALSE(pyarb::test_identifier("foo bar"));
    EXPECT_FALSE(pyarb::test_identifier("foo-bar"));
    EXPECT_FALSE(pyarb::test_identifier(""));
}
