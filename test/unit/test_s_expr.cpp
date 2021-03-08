#include <any>
#include <typeinfo>

#include "../test/gtest.h"

#include <arbor/morph/region.hpp>
#include <arbor/morph/locset.hpp>
#include <arbor/morph/label_parse.hpp>
#include <arbor/s_expr.hpp>

#include <arborio/cableio.hpp>

#include "parse_expression.hpp"
#include "util/strprintf.hpp"

using namespace arb;
using namespace std::string_literals;

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

TEST(s_expr, list) {
    {
        auto l = slist();
        EXPECT_EQ(0u, std::distance(l.begin(), l.end()));
    }
    {
        auto l = slist(1);
        EXPECT_EQ(1u, std::distance(l.begin(), l.end()));
    }
    {
        auto l = slist(1, 2.3);
        EXPECT_EQ(2u, std::distance(l.begin(), l.end()));
    }
    {
        auto l = slist(1, 2.3, "hello");
        EXPECT_EQ(3u, std::distance(l.begin(), l.end()));
    }
    {
        auto l = slist(1, 2.3, "hello"_symbol);
        EXPECT_EQ(3u, std::distance(l.begin(), l.end()));
        EXPECT_EQ(tok::symbol, (l.begin()+2)->atom().kind);
    }
    {
        auto l = slist(1, slist(1, 2), 3);
        EXPECT_EQ(3u, std::distance(l.begin(), l.end()));
        EXPECT_EQ(tok::integer, (l.begin()+2)->atom().kind);
        auto l1 = *(l.begin()+1);
        EXPECT_EQ(2u, std::distance(l1.begin(), l1.end()));
    }
}

TEST(s_expr, list_range) {
    std::cout << slist_range(std::vector{1,2,3}) << "\n";
    std::cout << slist_range(std::vector<int>{}) << "\n";
    std::cout << slist_range(std::vector{12.1, 0.1}) << "\n";
    std::cout << slist_range(slist(1, 2, "hello world")) << "\n";
}

TEST(s_expr, iterate) {
    {
        auto l = slist();
        EXPECT_EQ(0u, std::distance(l.begin(), l.end()));
    }
    {
        auto l = slist(1, 2, 3., "hello");
        auto b = l.begin();
        auto e = l.end();
        EXPECT_EQ(4u, std::distance(b, e));
        EXPECT_EQ(tok::integer, b->atom().kind);
        ++b;
        EXPECT_EQ(tok::integer, b->atom().kind);
        ++b;
        EXPECT_EQ(tok::real, b->atom().kind);
        ++b;
        EXPECT_EQ(tok::string, b->atom().kind);
        EXPECT_EQ("hello", b->atom().spelling);
    }
}

template <typename L>
std::string round_trip_label(const char* in) {
    if (auto x = parse_label_expression(std::string(in))) {
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
        EXPECT_FALSE(parse_label_expression(std::string(expr)));
    }
}

namespace arb {
namespace cable_s_expr {
template <typename T, std::size_t I = 0>
std::optional<T> eval_cast_variant(const std::any& a) {
    if constexpr (I<std::variant_size_v<T>) {
        using var_type = std::variant_alternative_t<I, T>;
        return (typeid(var_type)==a.type())? std::any_cast<var_type>(a): eval_cast_variant<T, I+1>(a);
    }
    return std::nullopt;
}

using branch = std::tuple<int, int, std::vector<arb::msegment>>;
using place_pair = std::pair<arb::locset, arb::placeable>;
using paint_pair = std::pair<arb::region, arb::paintable>;
using locset_pair = std::pair<std::string, locset>;
using region_pair = std::pair<std::string, region>;

std::ostream& operator<<(std::ostream& o, const cv_policy&) {
    return o;
}
std::ostream& operator<<(std::ostream& o, const branch& b) {
    o << "(branch " << std::to_string(std::get<0>(b)) << " " << std::to_string(std::get<1>(b));
    for (auto s: std::get<2>(b)) {
        o << " " << s;
    }
    return o << ")";
}
std::ostream& operator<<(std::ostream& o, const paint_pair& p) {
    o << "(paint " << p.first << " ";
    std::visit([&](auto&& x) {o << x;}, p.second);
    return o << ")";
}
std::ostream& operator<<(std::ostream& o, const place_pair& p) {
    o << "(place " << p.first << " ";
    std::visit([&](auto&& x) {o << x;}, p.second);
    return o << ")";
}
std::ostream& operator<<(std::ostream& o, const defaultable& p) {
    o << "(default ";
    std::visit([&](auto&& x) {o << x;}, p);
    return o << ")";
}
std::ostream& operator<<(std::ostream& o, const locset_pair& p) {
    return o << "(locset-def \"" << p.first << "\" " << p.second << ")";
}
std::ostream& operator<<(std::ostream& o, const region_pair& p) {
    return o << "(region-def \"" << p.first << "\" " << p.second << ")";
}

template <typename T>
std::string to_string(const T& obj) {
    std::stringstream s;
    s << obj;
    return s.str();
}
std::string to_string(const arborio::cable_cell_component& c) {
    std::stringstream s;
    arborio::write_component(s, c);
    return s.str();
}
} // namespace
} // namespace arb

template <typename T>
std::string round_trip_variant(const char* in) {
    using namespace cable_s_expr;
    if (auto x = arborio::parse_expression(std::string(in))) {
        std::string str;
        std::visit([&](auto&& p){str = to_string(p);}, *(eval_cast_variant<T>(*x)));
        return str;
    }
    else {
        return x.error().what();
    }
}

template <typename T>
std::string round_trip(const char* in) {
    using namespace cable_s_expr;
    if (auto x = arborio::parse_expression(std::string(in))) {
        return to_string(std::any_cast<T>(*x));
    }
    else {
        return x.error().what();
    }
}

std::string round_trip_component(const char* in) {
    using namespace cable_s_expr;
    if (auto x = arborio::parse_component(std::string(in))) {
        return to_string(x.value());
    }
    else {
        return x.error().what();
    }
}

TEST(decor_literals, round_tripping) {
    auto paint_default_literals = {
        "(membrane-potential -65.1)",
        "(temperature-kelvin 301)",
        "(axial-resistivity 102)",
        "(membrane-capacitance 0.01)",
        "(ion-internal-concentration \"ca\" 75.1)",
        "(ion-external-concentration \"h\" -50.1)",
        "(ion-reversal-potential \"na\" 30)"};
    auto paint_literals = {
        "(mechanism \"hh\")",
        "(mechanism \"pas\" (\"g\" 0.02))",
    };
    auto default_literals = {
        "(ion-reversal-potential-method \"ca\" (mechanism \"nernst/ca\"))"};
    auto place_literals = {
        "(current-clamp 10 100 0.5)",
        "(threshold-detector -10)",
        "(gap-junction-site)",
        "(mechanism \"expsyn\")"};
    for (auto l: paint_default_literals) {
        EXPECT_EQ(l, round_trip_variant<defaultable>(l));
        EXPECT_EQ(l, round_trip_variant<paintable>(l));
    }
    for (auto l: paint_literals) {
        EXPECT_EQ(l, round_trip_variant<paintable>(l));
    }
    for (auto l: default_literals) {
        EXPECT_EQ(l, round_trip_variant<defaultable>(l));
    }
    for (auto l: place_literals) {
        EXPECT_EQ(l, round_trip_variant<placeable>(l));
    }

    std::string mech_str = "(mechanism \"kamt\" (\"gbar\" 50) (\"zetam\" 0.1) (\"q10\" 5))";
    auto maybe_mech = arborio::parse_expression(mech_str);
    EXPECT_TRUE(maybe_mech);

    auto any_mech = maybe_mech.value();
    EXPECT_TRUE(typeid(mechanism_desc) == any_mech.type());

    auto mech = std::any_cast<mechanism_desc>(any_mech);
    EXPECT_EQ("kamt", mech.name());
    EXPECT_EQ(3u, mech.values().size());

    EXPECT_EQ(50, mech.values().at("gbar"));
    EXPECT_EQ(0.1, mech.values().at("zetam"));
    EXPECT_EQ(5, mech.values().at("q10"));
}

TEST(decor_expressions, round_tripping) {
    using namespace cable_s_expr;
    auto decorate_paint_literals = {
        "(paint (region \"all\") (membrane-potential -65.1))",
        "(paint (tag 1) (temperature-kelvin 301))",
        "(paint (distal-interval (location 3 0)) (axial-resistivity 102))",
        "(paint (join (region \"dend\") (all)) (membrane-capacitance 0.01))",
        "(paint (radius-gt (tag 3) 1) (ion-internal-concentration \"ca\" 75.1))",
        "(paint (intersect (cable 2 0 0.5) (region \"axon\")) (ion-external-concentration \"h\" -50.1))",
        "(paint (region \"my_region\") (ion-reversal-potential \"na\" 30))",
        "(paint (cable 2 0.1 0.4) (mechanism \"hh\"))",
        "(paint (all) (mechanism \"pas\" (\"g\" 0.02)))"
    };
    auto decorate_default_literals = {
        "(default (membrane-potential -65.1))",
        "(default (temperature-kelvin 301))",
        "(default (axial-resistivity 102))",
        "(default (membrane-capacitance 0.01))",
        "(default (ion-internal-concentration \"ca\" 75.1))",
        "(default (ion-external-concentration \"h\" -50.1))",
        "(default (ion-reversal-potential \"na\" 30))",
        "(default (ion-reversal-potential-method \"ca\" (mechanism \"nernst/ca\")))"
    };
    auto decorate_place_literals = {
        "(place (location 3 0.2) (current-clamp 10 100 0.5))",
        "(place (terminal) (threshold-detector -10))",
        "(place (root) (gap-junction-site))",
        "(place (locset \"my!ls\") (mechanism \"expsyn\"))"};

    for (auto l: decorate_paint_literals) {
        EXPECT_EQ(l, round_trip<paint_pair>(l));
    }
    for (auto l: decorate_place_literals) {
        EXPECT_EQ(l, round_trip<place_pair>(l));
    }
    for (auto l: decorate_default_literals) {
        EXPECT_EQ(l, round_trip<defaultable>(l));
    }
}

TEST(label_dict_expressions, round_tripping) {
    using namespace cable_s_expr;
    auto locset_def_literals = {
        "(locset-def \"my! locset~\" (root))",
        "(locset-def \"ls0\" (location 0 1))"
    };
    auto region_def_literals = {
        "(region-def \"my region\" (all))",
        "(region-def \"reg42\" (cable 0 0.1 0.9))"
    };
    for (auto l: locset_def_literals) {
        EXPECT_EQ(l, round_trip<locset_pair>(l));
    }
    for (auto l: region_def_literals) {
        EXPECT_EQ(l, round_trip<region_pair>(l));
    }
}

TEST(morphology_literals, round_tripping) {
    using namespace cable_s_expr;
    auto point = "(point 701.6 -3.1 -10 0.6)";
    auto segment = "(segment 5 (point 5 2 3 1) (point 5 2 5 6) 42)";
    auto branches = {
        "(branch -1 5 (segment 5 (point 5 2 3 1) (point 5 2 3.1 0.5) 2))",
        "(branch -1 5"
        " (segment 2 (point 5 2 3 1) (point 5 2 5 6) 42)"
        " (segment 3 (point 5 2 3 1) (point 5 2 3.1 0.5) 2)"
        ")"
    };

    EXPECT_EQ(point, round_trip<mpoint>(point));
    EXPECT_EQ(segment, round_trip<msegment>(segment));
    for (auto l: branches) {
        EXPECT_EQ(l, round_trip<branch>(l));
    }
}

TEST(decor, round_tripping) {
    auto component_str = "(arbor-component \n"
                         "  (meta-data \n"
                         "    (version 1))\n"
                         "  (decorations \n"
                         "    (default \n"
                         "      (axial-resistivity 100.000000))\n"
                         "    (default \n"
                         "      (ion-reversal-potential-method \"na\" \n"
                         "        (mechanism \"nernst\")))\n"
                         "    (paint \n"
                         "      (region \"dend\")\n"
                         "      (mechanism \"pas\"))\n"
                         "    (paint \n"
                         "      (region \"soma\")\n"
                         "      (mechanism \"hh\"))\n"
                         "    (paint \n"
                         "      (join \n"
                         "        (tag 1)\n"
                         "        (tag 2))\n"
                         "      (ion-internal-concentration \"ca\" 0.500000))\n"
                         "    (place \n"
                         "      (location 0 0)\n"
                         "      (threshold-detector 10.000000))\n"
                         "    (place \n"
                         "      (location 0 0.5)\n"
                         "      (mechanism \"expsyn\" \n"
                         "        (\"tau\" 1.500000)))))";

    EXPECT_EQ(component_str, round_trip_component(component_str));
}

TEST(label_dict, round_tripping) {
    auto component_str = "(arbor-component \n"
                         "  (meta-data \n"
                         "    (version 1))\n"
                         "  (label-dict \n"
                         "    (region-def \"soma\" \n"
                         "      (tag 1))\n"
                         "    (region-def \"dend\" \n"
                         "      (tag 3))\n"
                         "    (locset-def \"root\" \n"
                         "      (root))))";

    EXPECT_EQ(component_str, round_trip_component(component_str));
}

TEST(morphology, round_tripping) {
    auto component_str = "(arbor-component \n"
                         "  (meta-data \n"
                         "    (version 1))\n"
                         "  (morphology \n"
                         "    (branch 0 -1 \n"
                         "      (segment 0 \n"
                         "        (point 0.000000 0.000000 -6.307850 6.307850)\n"
                         "        (point 0.000000 0.000000 6.307850 6.307850)\n"
                         "        1))\n"
                         "    (branch 1 0 \n"
                         "      (segment 1 \n"
                         "        (point 0.000000 0.000000 6.307850 6.307850)\n"
                         "        (point 0.000000 0.000000 72.974517 0.500000)\n"
                         "        3)\n"
                         "      (segment 2 \n"
                         "        (point 0.000000 0.000000 72.974517 0.500000)\n"
                         "        (point 0.000000 0.000000 139.641183 0.500000)\n"
                         "        3)\n"
                         "      (segment 3 \n"
                         "        (point 0.000000 0.000000 139.641183 0.500000)\n"
                         "        (point 0.000000 0.000000 206.307850 0.500000)\n"
                         "        3))\n"
                         "    (branch 2 0 \n"
                         "      (segment 4 \n"
                         "        (point 0.000000 0.000000 6.307850 6.307850)\n"
                         "        (point 0.000000 0.000000 72.974517 0.500000)\n"
                         "        3)\n"
                         "      (segment 5 \n"
                         "        (point 0.000000 0.000000 72.974517 0.500000)\n"
                         "        (point 0.000000 0.000000 139.641183 0.500000)\n"
                         "        3)\n"
                         "      (segment 6 \n"
                         "        (point 0.000000 0.000000 139.641183 0.500000)\n"
                         "        (point 0.000000 0.000000 206.307850 0.500000)\n"
                         "        3))\n"
                         "    (branch 3 2 \n"
                         "      (segment 7 \n"
                         "        (point 0.000000 0.000000 206.307850 0.500000)\n"
                         "        (point 0.000000 0.000000 257.974517 0.500000)\n"
                         "        3)\n"
                         "      (segment 8 \n"
                         "        (point 0.000000 0.000000 257.974517 0.500000)\n"
                         "        (point 0.000000 0.000000 309.641183 0.500000)\n"
                         "        3)\n"
                         "      (segment 9 \n"
                         "        (point 0.000000 0.000000 309.641183 0.500000)\n"
                         "        (point 0.000000 0.000000 361.307850 0.500000)\n"
                         "        3))\n"
                         "    (branch 4 2 \n"
                         "      (segment 10 \n"
                         "        (point 0.000000 0.000000 206.307850 0.500000)\n"
                         "        (point 0.000000 0.000000 257.974517 0.500000)\n"
                         "        3)\n"
                         "      (segment 11 \n"
                         "        (point 0.000000 0.000000 257.974517 0.500000)\n"
                         "        (point 0.000000 0.000000 309.641183 0.500000)\n"
                         "        3)\n"
                         "      (segment 12 \n"
                         "        (point 0.000000 0.000000 309.641183 0.500000)\n"
                         "        (point 0.000000 0.000000 361.307850 0.500000)\n"
                         "        3))\n"
                         "    (branch 5 3 \n"
                         "      (segment 13 \n"
                         "        (point 0.000000 0.000000 361.307850 0.500000)\n"
                         "        (point 0.000000 0.000000 416.307850 0.500000)\n"
                         "        3)\n"
                         "      (segment 14 \n"
                         "        (point 0.000000 0.000000 416.307850 0.500000)\n"
                         "        (point 0.000000 0.000000 471.307850 0.500000)\n"
                         "        3)\n"
                         "      (segment 21 \n"
                         "        (point 0.000000 0.000000 471.307850 0.500000)\n"
                         "        (point 0.000000 0.000000 503.807850 0.500000)\n"
                         "        3)\n"
                         "      (segment 22 \n"
                         "        (point 0.000000 0.000000 503.807850 0.500000)\n"
                         "        (point 0.000000 0.000000 536.307850 0.500000)\n"
                         "        3)\n"
                         "      (segment 29 \n"
                         "        (point 0.000000 0.000000 536.307850 0.500000)\n"
                         "        (point 0.000000 0.000000 556.307850 0.500000)\n"
                         "        3))\n"
                         "    (branch 6 3 \n"
                         "      (segment 15 \n"
                         "        (point 0.000000 0.000000 361.307850 0.500000)\n"
                         "        (point 0.000000 0.000000 416.307850 0.500000)\n"
                         "        3)\n"
                         "      (segment 16 \n"
                         "        (point 0.000000 0.000000 416.307850 0.500000)\n"
                         "        (point 0.000000 0.000000 471.307850 0.500000)\n"
                         "        3)\n"
                         "      (segment 23 \n"
                         "        (point 0.000000 0.000000 471.307850 0.500000)\n"
                         "        (point 0.000000 0.000000 503.807850 0.500000)\n"
                         "        3)\n"
                         "      (segment 24 \n"
                         "        (point 0.000000 0.000000 503.807850 0.500000)\n"
                         "        (point 0.000000 0.000000 536.307850 0.500000)\n"
                         "        3))\n"
                         "    (branch 7 4 \n"
                         "      (segment 17 \n"
                         "        (point 0.000000 0.000000 361.307850 0.500000)\n"
                         "        (point 0.000000 0.000000 416.307850 0.500000)\n"
                         "        3)\n"
                         "      (segment 18 \n"
                         "        (point 0.000000 0.000000 416.307850 0.500000)\n"
                         "        (point 0.000000 0.000000 471.307850 0.500000)\n"
                         "        3)\n"
                         "      (segment 25 \n"
                         "        (point 0.000000 0.000000 471.307850 0.500000)\n"
                         "        (point 0.000000 0.000000 503.807850 0.500000)\n"
                         "        3)\n"
                         "      (segment 26 \n"
                         "        (point 0.000000 0.000000 503.807850 0.500000)\n"
                         "        (point 0.000000 0.000000 536.307850 0.500000)\n"
                         "        3))\n"
                         "    (branch 8 4 \n"
                         "      (segment 19 \n"
                         "        (point 0.000000 0.000000 361.307850 0.500000)\n"
                         "        (point 0.000000 0.000000 416.307850 0.500000)\n"
                         "        3)\n"
                         "      (segment 20 \n"
                         "        (point 0.000000 0.000000 416.307850 0.500000)\n"
                         "        (point 0.000000 0.000000 471.307850 0.500000)\n"
                         "        3)\n"
                         "      (segment 27 \n"
                         "        (point 0.000000 0.000000 471.307850 0.500000)\n"
                         "        (point 0.000000 0.000000 503.807850 0.500000)\n"
                         "        3)\n"
                         "      (segment 28 \n"
                         "        (point 0.000000 0.000000 503.807850 0.500000)\n"
                         "        (point 0.000000 0.000000 536.307850 0.500000)\n"
                         "        3))))";

    EXPECT_EQ(component_str, round_trip_component(component_str));
}

TEST(cable_cell, round_tripping) {
    auto component_str = "(arbor-component \n"
                         "  (meta-data \n"
                         "    (version 1))\n"
                         "  (cable-cell \n"
                         "    (morphology \n"
                         "      (branch 0 -1 \n"
                         "        (segment 0 \n"
                         "          (point -6.300000 0.000000 0.000000 6.300000)\n"
                         "          (point 6.300000 0.000000 0.000000 6.300000)\n"
                         "          1)\n"
                         "        (segment 1 \n"
                         "          (point 6.300000 0.000000 0.000000 0.500000)\n"
                         "          (point 206.300000 0.000000 0.000000 0.200000)\n"
                         "          3)))\n"
                         "    (label-dict \n"
                         "      (region-def \"soma\" \n"
                         "        (tag 1))\n"
                         "      (region-def \"dend\" \n"
                         "        (join \n"
                         "          (join \n"
                         "            (tag 3)\n"
                         "            (tag 4))\n"
                         "          (tag 42))))\n"
                         "    (decorations \n"
                         "      (paint \n"
                         "        (region \"dend\")\n"
                         "        (mechanism \"pas\"))\n"
                         "      (paint \n"
                         "        (region \"soma\")\n"
                         "        (mechanism \"hh\"))\n"
                         "      (place \n"
                         "        (location 0 1)\n"
                         "        (mechanism \"exp2syn\")))))";

    EXPECT_EQ(component_str, round_trip_component(component_str));
}

TEST(cable_cell_literals, errors) {
    for (auto expr: {"(membrane-potential \"56\")",  // invalid argument
                     "(axial-resistivity 1 2)",      // too many arguments to otherwise valid decor literal
                     "(membrane-capacitance ",       // syntax error
                     "(mem-capacitance 3.5)",        // invalid function
                     "(ion-initial-concentration ca 0.1)",   // unquoted ion
                     "(mechanism \"hh\" (gl 3.5))",          // unqouted parameter
                     "(mechanism \"pas\" ((\"g\" 0.5) (\"e\" 0.2)))",   // paranthesis around params
                     "(gap-junction-site 0)",                // too many arguments
                     "(current-clamp 1 2)",                  // too few arguments
                     "(paint (region) (mechanism \"hh\"))",  // invalid region
                     "(paint (tag 1) (mechanims hh))",       // invalid painting
                     "(paint (terminals) (membrance-capacitance 0.2))", // can't paint a locset
                     "(paint (tag 3))",                      // too few arguments
                     "(place (locset) (gap-junction-site))", // invalid locset
                     "(place (gap-junction-site) (location 0 1))",      // swapped argument order
                     "(region-def my_region (tag 3))",       // unquoted region name
                     "(locset-def \"my_ls\" (tag 3))",       // invalid locset
                     "(locset-def \"my_ls\")",               // too few arguments
                     "(branch 0 1)",                         // branch with zero segments
                     "(segment -1 (point 1 2 3 4) 3)",       // segment with 1 point
                     "(point 1 2 3)",                        // too few arguments
                     "(morphology (segment -1 (point 1 2 3 4) (point 2 3 4 5) 3))", // morphology with segment instead of branch
                     "(decor (region-def \"reg\" (tag 3)))", // decor with label definiton
                     "(cable-cell (paint (tag 3) (axial-resistivity 3.1)))" // cable-cell with paint
    })
    {
        // If an expression was passed and handled correctly the parse call will
        // return without throwing, with the error stored in the return type.
        // So it is sufficient to assert that it evaluates to false.
        EXPECT_FALSE(arborio::parse_expression(expr));
    }
    for (auto expr: {"(arbor-component (meta-data (version 2)) (decorations))",  // invalid component
                     "(arbor-component (morphology))", // arbor-component missing meta-data
                     "(arbor-component (meta-data (version 1)))", // arbor-component missing component
                     "(arbor-component (meta-data (version 1)) (membrane-potential 56))",  // invalid component
                     "(arbor-component (meta-data (version 1)) (morphology (segment 1 (point 1 2 3 4) (point 2 3 4 5) 3)))", // morphology with segment instead of branch
                     "(arbor-component (meta-data (version 1)) (decorations (region-def \"reg\" (tag 3))))", // decor with label definition
                     "(arbor-component (meta-data (version 1)) (cable-cell (paint (tag 3) (axial-resistivity 3.1))))", // cable-cell with paint
                     "(morphology (branch 0 -1 (segment 0 (point 0 1 2 3 ) (point 1 2 3 4) 3)))", // morphology without arbor-component
    })
    {
        // If an expression was passed and handled correctly the parse call will
        // return without throwing, with the error stored in the return type.
        // So it is sufficient to assert that it evaluates to false.
        EXPECT_FALSE(arborio::parse_component(expr));
    }
}
