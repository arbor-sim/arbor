#include "../test/gtest.h"

#include <arbor/morph/region.hpp>
#include <arbor/morph/locset.hpp>
#include <arbor/morph/label_parse.hpp>

#include "s_expr.hpp"
#include "util/strprintf.hpp"

using namespace arb;
using namespace std::string_literals;

TEST(s_expr, identifier) {
    EXPECT_TRUE(valid_label_name("foo"));
    EXPECT_TRUE(valid_label_name("f1"));
    EXPECT_TRUE(valid_label_name("f_"));
    EXPECT_TRUE(valid_label_name("f_1__"));
    EXPECT_TRUE(valid_label_name("A_1__"));

    EXPECT_TRUE(valid_label_name("A-1"));
    EXPECT_TRUE(valid_label_name("hello-world"));
    EXPECT_TRUE(valid_label_name("hello--world"));
    EXPECT_TRUE(valid_label_name("hello--world_"));

    EXPECT_FALSE(valid_label_name("_foobar"));
    EXPECT_FALSE(valid_label_name("-foobar"));
    EXPECT_FALSE(valid_label_name("2dogs"));
    EXPECT_FALSE(valid_label_name("1"));
    EXPECT_FALSE(valid_label_name("_"));
    EXPECT_FALSE(valid_label_name("-"));
    EXPECT_FALSE(valid_label_name(""));
    EXPECT_FALSE(valid_label_name(" foo"));
    EXPECT_FALSE(valid_label_name("foo "));
    EXPECT_FALSE(valid_label_name("foo bar"));
    EXPECT_FALSE(valid_label_name(""));
}

TEST(s_expr, atoms) {
    auto get_atom = [](s_expr e) {
        EXPECT_EQ(1u, length(e));
        EXPECT_TRUE(e.head().is_atom());
        return e.head().atom();
    };

    for (auto spelling: {"foo", "bar_", "car1", "car_1", "x_1__"}){
        auto a = get_atom(parse_s_expr("("s+spelling+")"));
        EXPECT_EQ(a.kind, tok::name);
        EXPECT_EQ(a.spelling, spelling);
    }
    // test parsing of integers
    for (auto spelling: {"0", "-0", "1", "42", "-3287"}){
        auto a = get_atom(parse_s_expr("("s+spelling+")"));
        EXPECT_EQ(a.kind, tok::integer);
        EXPECT_EQ(a.spelling, spelling);
    }
    // test parsing of real numbers
    for (auto spelling: {"0.", "-0.0", "1.21", "42.", "-3287.12"}){
        auto a = get_atom(parse_s_expr("("s+spelling+")"));
        EXPECT_EQ(a.kind, tok::real);
        EXPECT_EQ(a.spelling, spelling);
    }
    // test parsing of strings
    for (auto spelling: {"foo", "dog cat", ""}) {
        auto s = "(\""s+spelling+"\")";
        auto a = get_atom(parse_s_expr(s));
        EXPECT_EQ(a.kind, tok::string);
    }
}

std::string round_trip_reg(const char* in) {
    auto x = parse_label_expression(in);
    return util::pprintf("{}", arb::util::any_cast<arb::region>(*x));
}
std::string round_trip_loc(const char* in) {
    auto x = parse_label_expression(in);
    return util::pprintf("{}", arb::util::any_cast<arb::locset>(*x));
}

TEST(regloc, parse_str) {
    EXPECT_EQ("(cable 3 0 1)",      round_trip_reg("(branch 3)"));
    EXPECT_EQ("(cable 2 0.1 0.4)",  round_trip_reg("(cable 2 0.1 0.4)"));
    EXPECT_EQ("(all)",              round_trip_reg("(all)"));
    EXPECT_EQ("(region \"foo\")",   round_trip_reg("(region \"foo\")"));

    EXPECT_EQ("(terminal)", round_trip_loc("(terminal)"));
    EXPECT_EQ("(root)",     round_trip_loc("(root)"));
    EXPECT_EQ("(locset \"cat_burgler\")", round_trip_loc("(locset \"cat_burgler\")"));

    auto lhs = arb::util::any_cast<arb::region>(*parse_label_expression("(region \"dend\")"));
    auto rhs = arb::util::any_cast<arb::region>(*parse_label_expression("(all)"));

    EXPECT_EQ(util::pprintf("{}", join(lhs,rhs)), "(join (region \"dend\") (all))");
}

TEST(regloc, comments) {
    EXPECT_EQ("(all)",  round_trip_reg("(all) ; a comment"));
    const char *multi_line = 
        "; comment at start\n"
        "(radius_lt\n"
        "    (join\n"
        "        (tag 3) ; end of line\n"
        " ;comment on whole line\n"
        "        (tag 4))\n"
        "    0.5) ; end of string";
    EXPECT_EQ("(radius_lt (join (tag 3) (tag 4)) 0.5)",
              round_trip_reg(multi_line));
}
