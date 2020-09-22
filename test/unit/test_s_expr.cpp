#include <any>

#include "../test/gtest.h"

#include <arbor/morph/region.hpp>
#include <arbor/morph/locset.hpp>
#include <arbor/morph/label_parse.hpp>
#include <typeinfo>

#include "s_expr.hpp"
#include "util/strprintf.hpp"

using namespace arb;
using namespace std::string_literals;

TEST(s_expr, transmogrify) {
    using map_t = std::unordered_map<char, std::string>;
    auto transform = [](std::string in, map_t map) {
        auto t = transmogrifier(in, map);
        std::string s;
        while (t) s.push_back(*t++);
        return s;
    };
    EXPECT_EQ(transform("(42,24)", {{',', " "}}), "(42 24)");
    EXPECT_EQ(transform("(42,24)", {{',', "hello"}}), "(42hello24)");
    EXPECT_EQ(transform("(42,24)", {{',', " "}}), "(42 24)");
    EXPECT_EQ(transform("(42,,24)", {{',', " "}}), "(42  24)");
    map_t asc_map = {{',', " "},
                     {'|', ")("},
                     {'<', "(spine "},
                     {'>', ")"}};
    EXPECT_EQ(transform("(RGB 128,128,128)", asc_map), "(RGB 128 128 128)");
    EXPECT_EQ(transform("<color blue>", asc_map), "(spine color blue)");
    EXPECT_EQ(transform("(1 2 3 | 4 5 6)", asc_map), "(1 2 3 )( 4 5 6)");
    EXPECT_EQ(transform("", asc_map), "");
    EXPECT_EQ(transform("<>", asc_map), "(spine )");
    EXPECT_EQ(transform("<32|>", asc_map), "(spine 32)()");
}

TEST(s_expr, atoms) {
    auto get_atom = [](s_expr e) {
        return e.atom();
    };

    for (auto spelling: {"foo", "bar_", "car1", "car_1", "x_1__", "x/bar", "x/bar@4.2"}){
        auto a = get_atom(parse_s_expr(spelling));
        EXPECT_EQ(a.kind, tok::symbol);
        EXPECT_EQ(a.spelling, spelling);
    }
    // test parsing of integers
    for (auto spelling: {"0", "-0", "1", "42", "-3287"}){
        auto a = get_atom(parse_s_expr(spelling));
        EXPECT_EQ(a.kind, tok::integer);
        EXPECT_EQ(a.spelling, spelling);
    }
    // test parsing of real numbers
    for (auto spelling: {"0.", "-0.0", "1.21", "42.", "-3287.12"}){
        auto a = get_atom(parse_s_expr(spelling));
        EXPECT_EQ(a.kind, tok::real);
        EXPECT_EQ(a.spelling, spelling);
    }
    // test parsing of strings
    for (auto spelling: {"foo", "dog cat", ""}) {
        auto s = "\""s+spelling+"\"";
        auto a = get_atom(parse_s_expr(s));
        EXPECT_EQ(a.kind, tok::string);
    }
}

TEST(s_expr, atoms_in_parens) {
    auto get_atom = [](s_expr e) {
        EXPECT_EQ(1u, length(e));
        EXPECT_TRUE(e.head().is_atom());
        return e.head().atom();
    };

    for (auto spelling: {"foo", "bar_", "car1", "car_1", "x_1__", "x/bar", "x/bar@4.2"}){
        auto a = get_atom(parse_s_expr("("s+spelling+")"));
        EXPECT_EQ(a.kind, tok::symbol);
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

template <typename L>
std::string round_trip_label(const char* in) {
    if (auto x = parse_label_expression(in)) {
        return util::pprintf("{}", std::any_cast<L>(*x));
    }
    else {
        return x.error().what();
    }
}

std::string round_trip_region(const char* in) {
    if (auto x = parse_region_expression(in)) {
        return util::pprintf("{}", std::any_cast<arb::region>(*x));
    }
    else {
        return x.error().what();
    }
}

std::string round_trip_locset(const char* in) {
    if (auto x = parse_locset_expression(in)) {
        return util::pprintf("{}", std::any_cast<arb::locset>(*x));
    }
    else {
        return x.error().what();
    }
}

TEST(regloc, round_tripping) {
    EXPECT_EQ("(cable 3 0 1)", round_trip_label<arb::region>("(branch 3)"));
    EXPECT_EQ("(intersect (tag 1) (intersect (tag 2) (tag 3)))", round_trip_label<arb::region>("(intersect (tag 1) (tag 2) (tag 3))"));
    auto region_literals = {
        "(cable 2 0.1 0.4)",
        "(region \"foo\")",
        "(all)",
        "(tag 42)",
        "(distal-interval (location 3 0))",
        "(distal-interval (location 3 0) 3.2)",
        "(proximal-interval (location 3 0))",
        "(proximal-interval (location 3 0) 3.2)",
        "(join (region \"dend\") (all))",
        "(radius-lt (tag 1) 1)",
        "(radius-le (tag 2) 1)",
        "(radius-gt (tag 3) 1)",
        "(radius-ge (tag 4) 3)",
        "(intersect (cable 2 0 0.5) (region \"axon\"))",
    };
    for (auto l: region_literals) {
        EXPECT_EQ(l, round_trip_label<arb::region>(l));
        EXPECT_EQ(l, round_trip_region(l));
    }

    auto locset_literals = {
        "(root)",
        "(locset \"cat man do\")",
        "(location 3 0.2)",
        "(terminal)",
        "(distal (tag 2))",
        "(proximal (join (tag 1) (tag 2)))",
        "(uniform (tag 1) 0 100 52)",
        "(restrict (terminal) (tag 12))",
        "(join (terminal) (root))",
        "(sum (terminal) (root))",
    };
    for (auto l: locset_literals) {
        EXPECT_EQ(l, round_trip_label<arb::locset>(l));
        EXPECT_EQ(l, round_trip_locset(l));
    }
}

TEST(regloc, comments) {
    EXPECT_EQ("(all)",  round_trip_region("(all) ; a comment"));
    const char *multi_line = 
        "; comment at start\n"
        "(radius-lt\n"
        "    (join\n"
        "        (tag 3) ; end of line\n"
        " ;comment on whole line\n"
        "        (tag 4))\n"
        "    0.5) ; end of string";
    EXPECT_EQ("(radius-lt (join (tag 3) (tag 4)) 0.5)",
              round_trip_region(multi_line));
}

TEST(regloc, errors) {
    for (auto expr: {"axon",         // unquoted region name
                     "(tag 1.2)",    // invalid argument in an otherwise valid region expression
                     "(tag 1 2)",    // too many arguments to otherwise valid region expression
                     "(tag 1) (tag 2)", // more than one valid description
                     "(tag",         // syntax error in region expression
                     "(terminal)",   // a valid locset expression
                     "\"my region",  // unclosed quote on label
                     })
    {
        // If an invalid label/expression was passed and handled correctly the parse
        // call will return without throwing, with the error stored in the return type.
        // So it is sufficient to assert that it evaluates to false.
        EXPECT_FALSE(parse_region_expression(expr));
    }

    for (auto expr: {"axon",         // unquoted locset name
                     "(location 1 \"0.5\")",  // invalid argument in an otherwise valid locset expression
                     "(location 1 0.2 0.2)",  // too many arguments to otherwise valid locset expression
                     "(root) (location 2 0)", // more than one valid description
                     "(tag",         // syntax error in locset expression
                     "(tag 3)",      // a valid region expression
                     "\"my locset",  // unclosed quote on label
                     })
    {
        // If an invalid label/expression was passed and handled correctly the parse
        // call will return without throwing, with the error stored in the return type.
        // So it is sufficient to assert that it evaluates to false.
        EXPECT_FALSE(parse_locset_expression(expr));
    }

    for (auto expr: {"axon",         // unquoted locset name
                     "(location 1 \"0.5\")",  // invalid argument in an otherwise valid locset expression
                     "(location 1 0.2 0.2)",  // too many arguments to otherwise valid locset expression
                     "(root) (location 2 0)", // more than one valid description
                     "(tag",         // syntax error in locset expression
                     "\"my locset",  // unclosed quote on label
                     })
    {
        // If an invalid label/expression was passed and handled correctly the parse
        // call will return without throwing, with the error stored in the return type.
        // So it is sufficient to assert that it evaluates to false.
        EXPECT_FALSE(parse_label_expression(expr));
    }
}
