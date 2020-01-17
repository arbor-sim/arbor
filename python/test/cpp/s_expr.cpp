#include "gtest.h"

#include <arbor/morph/region.hpp>
#include <arbor/morph/locset.hpp>

#include <morph_parse.hpp>
#include <s_expr.hpp>

using namespace pyarb;
using namespace std::string_literals;

TEST(s_expr, identifier) {
    EXPECT_TRUE(test_identifier("foo"));
    EXPECT_TRUE(test_identifier("f1"));
    EXPECT_TRUE(test_identifier("f_"));
    EXPECT_TRUE(test_identifier("f_1__"));
    EXPECT_TRUE(test_identifier("A_1__"));

    EXPECT_FALSE(test_identifier("_foobar"));
    EXPECT_FALSE(test_identifier("2dogs"));
    EXPECT_FALSE(test_identifier("1"));
    EXPECT_FALSE(test_identifier("_"));
    EXPECT_FALSE(test_identifier(""));
    EXPECT_FALSE(test_identifier(" foo"));
    EXPECT_FALSE(test_identifier("foo "));
    EXPECT_FALSE(test_identifier("foo bar"));
    EXPECT_FALSE(test_identifier("foo-bar"));
    EXPECT_FALSE(test_identifier(""));
}

TEST(s_expr, atoms) {
    auto get_atom = [](s_expr e) {
        EXPECT_EQ(1u, length(e));
        EXPECT_TRUE(e.head().is_atom());
        return e.head().atom();
    };

    for (auto spelling: {"foo", "bar_", "car1", "car_1", "x_1__"}){
        auto a = get_atom(parse("("s+spelling+")"));
        EXPECT_EQ(a.kind, tok::name);
        EXPECT_EQ(a.spelling, spelling);
    }
    // test parsing of integers
    for (auto spelling: {"0", "-0", "1", "42", "-3287"}){
        auto a = get_atom(parse("("s+spelling+")"));
        EXPECT_EQ(a.kind, tok::integer);
        EXPECT_EQ(a.spelling, spelling);
    }
    // test parsing of real numbers
    for (auto spelling: {"0.", "-0.0", "1.21", "42.", "-3287.12"}){
        auto a = get_atom(parse("("s+spelling+")"));
        EXPECT_EQ(a.kind, tok::real);
        EXPECT_EQ(a.spelling, spelling);
    }
    // test parsing of strings
    for (auto spelling: {"foo", "dog cat", ""}) {
        auto s = "(\""s+spelling+"\")";
        auto a = get_atom(parse(s));
        EXPECT_EQ(a.kind, tok::string);
    }
}

TEST(s_expr, parse) {
    auto round_trip_reg = [](const char* in) {
        auto x = eval(parse(in));
        return util::pprintf("{}", arb::util::any_cast<arb::region>(*x));
    };
    auto round_trip_loc = [](const char* in) {
        auto x = eval(parse(in));
        return util::pprintf("{}", arb::util::any_cast<arb::locset>(*x));
    };

    EXPECT_EQ("(cable 3 0 1)",      round_trip_reg("(branch 3)"));
    EXPECT_EQ("(cable 2 0.1 0.4)",  round_trip_reg("(cable 2 0.1 0.4)"));
    EXPECT_EQ("(all)",              round_trip_reg("(all)"));
    EXPECT_EQ("(region \"foo\")",   round_trip_reg("(region \"foo\")"));

    EXPECT_EQ("(terminal)", round_trip_loc("(terminal)"));
    EXPECT_EQ("(root)",     round_trip_loc("(root)"));
    EXPECT_EQ("(locset \"cat_burgler\")", round_trip_loc("(locset \"cat_burgler\")"));

    auto lhs = arb::util::any_cast<arb::region>(*eval(parse("(region \"dend\")")));
    auto rhs = arb::util::any_cast<arb::region>(*eval(parse("(all)")));

    EXPECT_EQ(util::pprintf("{}", join(lhs,rhs)), "(join (region \"dend\") (all))");
}
