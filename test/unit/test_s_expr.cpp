#include <any>
#include <typeinfo>

#include <gtest/gtest.h>

#include <arbor/morph/region.hpp>
#include <arbor/morph/locset.hpp>
#include <arbor/cv_policy.hpp>
#include <arbor/s_expr.hpp>
#include <arbor/util/any_visitor.hpp>

#include <arborio/cv_policy_parse.hpp>
#include <arborio/cableio.hpp>
#include <arborio/label_parse.hpp>

#include "parse_s_expr.hpp"
#include "util/strprintf.hpp"

using namespace arb;
using namespace arborio;
using namespace arborio::literals;
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
    using namespace arborio;
    auto to_string = [](const s_expr& obj) {
      std::stringstream s;
      s << obj;
      return s.str();
    };
    {
        auto l = slist();
        EXPECT_EQ("()", to_string(l));
        EXPECT_EQ(0u, std::distance(l.begin(), l.end()));
    }
    {
        auto l = slist(1);
        EXPECT_EQ("(1)", to_string(l));
        EXPECT_EQ(1u, std::distance(l.begin(), l.end()));
    }
    {
        auto l = slist(1, 2.3);
        EXPECT_EQ("(1 2.300000)", to_string(l));
        EXPECT_EQ(2u, std::distance(l.begin(), l.end()));
    }
    {
        auto l = slist(1, 2.3, s_expr("hello"));
        EXPECT_EQ("(1 2.300000 \"hello\")", to_string(l));
        EXPECT_EQ(3u, std::distance(l.begin(), l.end()));
    }
    {
        auto l = slist(1, 2.3, "hello"_symbol);
        EXPECT_EQ("(1 2.300000 hello)", to_string(l));
        EXPECT_EQ(3u, std::distance(l.begin(), l.end()));
        EXPECT_EQ(tok::symbol, (l.begin()+2)->atom().kind);
    }
    {
        auto l = slist(1, slist(1, 2), 3);
        EXPECT_EQ("(1 \n  (1 2)\n  3)", to_string(l));
        EXPECT_EQ(3u, std::distance(l.begin(), l.end()));
        EXPECT_EQ(tok::integer, (l.begin()+2)->atom().kind);
        auto l1 = *(l.begin()+1);
        EXPECT_EQ(2u, std::distance(l1.begin(), l1.end()));
    }
}

TEST(s_expr, list_range) {
    using namespace arborio;
    auto to_string = [](const s_expr& obj) {
        std::stringstream s;
        s << obj;
        return s.str();
    };
    auto s0 = slist_range(std::vector{1,2,3});
    EXPECT_EQ("(1 2 3)", to_string(s0));
    EXPECT_EQ(3u, length(s0));

    auto s1 = slist_range(std::vector<int>{});
    EXPECT_EQ("()", to_string(s1));
    EXPECT_EQ(0u, length(s1));

    auto s2 = slist_range(std::vector{12.1, 0.1});
    EXPECT_EQ("(12.100000 0.100000)", to_string(s2));
    EXPECT_EQ(2u, length(s2));

    auto s3 = slist_range(slist(1, 2, s_expr("hello world")));
    EXPECT_EQ("(1 2 \"hello world\")", to_string(s3));
    EXPECT_EQ(3u, length(s3));
}

TEST(s_expr, iterate) {
    using namespace arborio;
    {
        auto l = slist();
        EXPECT_EQ(0u, std::distance(l.begin(), l.end()));
    }
    {
        auto l = slist(1, 2, 3., s_expr("hello"));
        auto b = l.begin();
        auto e = l.end();
        EXPECT_EQ(4u, std::distance(b, e));
        EXPECT_EQ(tok::integer, b++->atom().kind);
        EXPECT_EQ(tok::integer, b++->atom().kind);
        EXPECT_EQ(tok::real, b++->atom().kind);
        EXPECT_EQ(tok::string, b->atom().kind);
        EXPECT_EQ("hello", b->atom().spelling);
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

std::string round_trip_cv(const char* in) {
    if (auto x = parse_cv_policy_expression(in)) {
        return util::pprintf("{}", std::any_cast<cv_policy>(*x));
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

std::string round_trip_iexpr(const char* in) {
    if (auto x = parse_iexpr_expression(in)) {
        return util::pprintf("{}", std::any_cast<arb::iexpr>(*x));
    }
    else {
        return x.error().what();
    }
}


TEST(cv_policies, round_tripping) {
    auto literals = {"(every-segment (tag 42))",
                     "(fixed-per-branch 23 (segment 0) 1)",
                     "(max-extent 23.1 (segment 0) 1)",
                     "(single (segment 0))",
                     "(explicit (terminal) (segment 0))",
                     "(join (every-segment (tag 42)) (single (segment 0)))",
                     "(replace (every-segment (tag 42)) (single (segment 0)))",
    };
    for (const auto& literal: literals) {
        EXPECT_EQ(literal, round_trip_cv(literal));
    }
}

TEST(cv_policies, literals) {
    EXPECT_NO_THROW("(every-segment (tag 42))"_cvp);
    EXPECT_NO_THROW("(fixed-per-branch 23 (segment 0) 1)"_cvp);
    EXPECT_NO_THROW("(max-extent 23.1 (segment 0) 1)"_cvp);
    EXPECT_NO_THROW("(single (segment 0))"_cvp);
    EXPECT_NO_THROW("(explicit (terminal) (segment 0))"_cvp);
    EXPECT_NO_THROW("(join (every-segment (tag 42)) (single (segment 0)))"_cvp);
    EXPECT_NO_THROW("(replace (every-segment (tag 42)) (single (segment 0)))"_cvp);
}

TEST(cv_policies, bad) {
    auto check = [](const std::string& s) {
        auto cv = parse_cv_policy_expression(s);
        if (!cv.has_value()) throw cv.error();
        return cv.value();
    };

    EXPECT_THROW(check("(every-segment (tag 42) 1)"), cv_policy_parse_error); // extra arg
    EXPECT_THROW(check("(every-segment (terminal))"), cv_policy_parse_error); // locset instead of region
    EXPECT_THROW(check("(every-segment"), cv_policy_parse_error);             // missing paren
    EXPECT_THROW(check("(tag 42)"), cv_policy_parse_error);                   // not a cv_policy
}

TEST(iexpr, round_tripping) {
    EXPECT_EQ("(cable 3 0 1)", round_trip_label<arb::region>("(branch 3)"));
    EXPECT_EQ("(intersect (tag 1) (intersect (tag 2) (tag 3)))", round_trip_label<arb::region>("(intersect (tag 1) (tag 2) (tag 3))"));
    auto iexpr_literals = {
        "(scalar 2.1)",
        "(distance 3.2 (region \"foo\"))",
        "(distance 3.2 (location 3 0.2))",
        "(proximal-distance 3.2 (region \"foo\"))",
        "(proximal-distance 3.2 (location 3 0.2))",
        "(distal-distance 3.2 (region \"foo\"))",
        "(distal-distance 3.2 (location 3 0.2))",
        "(interpolation 3.2 (region \"foo\") 4.3 (radius-gt (tag 3) 1))",
        "(interpolation 3.2 (location 3 0.2) 4.3 (distal (tag 2)))",
        "(radius 2.1)",
        "(diameter 2.1)",
        "(exp (scalar 2.1))",
        "(step (scalar 2.1))",
        "(log (scalar 2.1))",
        "(add (scalar 2.1) (radius 3.2))",
        "(sub (scalar 2.1) (radius 3.2))",
        "(mul (scalar 2.1) (radius 3.2))",
        "(div (scalar 2.1) (radius 3.2))",
    };
    for (auto l: iexpr_literals) {
        EXPECT_EQ(l, round_trip_label<arb::iexpr>(l));
        EXPECT_EQ(l, round_trip_iexpr(l));
    }

    // check double values for input instead of explicit scalar iexpr
    auto mono_iexpr = {std::string("exp"), std::string("step"), std::string("log")};
    auto duo_iexpr = {std::string("add"), std::string("sub"), std::string("mul"), std::string("div")};
    constexpr auto v1 = "1.2";
    constexpr auto v2 = "1.2";
    for(const auto& l : mono_iexpr) {
        auto l_input = "(" + l + " " + v1 + ")";
        auto l_output = "(" + l + " (scalar " + v1 + "))";
        EXPECT_EQ(l_output.c_str(), round_trip_label<arb::iexpr>(l_input.c_str()));
        EXPECT_EQ(l_output.c_str(), round_trip_iexpr(l_input.c_str()));
    }
    for(const auto& l : duo_iexpr) {
        auto l_input_dd = "(" + l + " " + v1 + + " " + v2 +")";
        auto l_input_sd = "(" + l + " (scalar " + v1 + + ") " + v2 +")";
        auto l_input_ds = "(" + l + " " + v1 + + " (scalar " + v2 +"))";
        auto l_output = "(" + l + " (scalar " + v1 + ") (scalar " + v2 +"))";
        EXPECT_EQ(l_output.c_str(), round_trip_label<arb::iexpr>(l_input_dd.c_str()));
        EXPECT_EQ(l_output.c_str(), round_trip_iexpr(l_input_dd.c_str()));
        EXPECT_EQ(l_output.c_str(), round_trip_label<arb::iexpr>(l_input_sd.c_str()));
        EXPECT_EQ(l_output.c_str(), round_trip_iexpr(l_input_sd.c_str()));
        EXPECT_EQ(l_output.c_str(), round_trip_label<arb::iexpr>(l_input_ds.c_str()));
        EXPECT_EQ(l_output.c_str(), round_trip_iexpr(l_input_ds.c_str()));
    }

    // test order for more than two arguments
    EXPECT_EQ("(add (add (add (scalar 1.1) (scalar 2.2)) (scalar 3.3)) (scalar 4.4))",
        round_trip_label<arb::iexpr>("(add 1.1 2.2 3.3 4.4)"));
    EXPECT_EQ("(sub (sub (sub (scalar 1.1) (scalar 2.2)) (scalar 3.3)) (scalar 4.4))",
        round_trip_label<arb::iexpr>("(sub 1.1 2.2 3.3 4.4)"));
    EXPECT_EQ("(mul (mul (mul (scalar 1.1) (scalar 2.2)) (scalar 3.3)) (scalar 4.4))",
        round_trip_label<arb::iexpr>("(mul 1.1 2.2 3.3 4.4)"));
    EXPECT_EQ("(div (div (div (scalar 1.1) (scalar 2.2)) (scalar 3.3)) (scalar 4.4))",
        round_trip_label<arb::iexpr>("(div 1.1 2.2 3.3 4.4)"));

    // test default constructors
    EXPECT_EQ("(distance 1 (location 3 0.2))",
        round_trip_label<arb::iexpr>("(distance (location 3 0.2))"));
    EXPECT_EQ("(distance 1 (region \"foo\"))",
        round_trip_label<arb::iexpr>("(distance (region \"foo\"))"));
    EXPECT_EQ("(distal-distance 1 (location 3 0.2))",
        round_trip_label<arb::iexpr>("(distal-distance (location 3 0.2))"));
    EXPECT_EQ("(distal-distance 1 (region \"foo\"))",
        round_trip_label<arb::iexpr>("(distal-distance (region \"foo\"))"));
    EXPECT_EQ("(proximal-distance 1 (location 3 0.2))",
        round_trip_label<arb::iexpr>("(proximal-distance (location 3 0.2))"));
    EXPECT_EQ("(proximal-distance 1 (region \"foo\"))",
        round_trip_label<arb::iexpr>("(proximal-distance (region \"foo\"))"));
    EXPECT_EQ("(radius 1)",
        round_trip_label<arb::iexpr>("(radius)"));
    EXPECT_EQ("(diameter 1)",
        round_trip_label<arb::iexpr>("(diameter)"));
    EXPECT_EQ("(scalar 3.14159)",
        round_trip_label<arb::iexpr>("(pi)"));
}

TEST(regloc, round_tripping) {
    EXPECT_EQ("(cable 3 0 1)", round_trip_label<arb::region>("(branch 3)"));
    EXPECT_EQ("(intersect (tag 1) (intersect (tag 2) (tag 3)))", round_trip_label<arb::region>("(intersect (tag 1) (tag 2) (tag 3))"));
    auto region_literals = {
        "(cable 2 0.1 0.4)",
        "(region \"foo\")",
        "(all)",
        "(region-nil)",
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
        "(complement (region \"axon\"))",
        "(difference (region \"axon\") (region \"soma\"))",
    };
    for (auto l: region_literals) {
        EXPECT_EQ(l, round_trip_label<arb::region>(l));
        EXPECT_EQ(l, round_trip_region(l));
    }

    auto locset_literals = {
        "(root)",
        "(locset \"cat man do\")",
        "(locset-nil)",
        "(location 3 0.2)",
        "(terminal)",
        "(distal (tag 2))",
        "(proximal (join (tag 1) (tag 2)))",
        "(uniform (tag 1) 0 100 52)",
        "(restrict (terminal) (tag 12))",
        "(on-components 0.3 (segment 2))",
        "(join (terminal) (root))",
        "(sum (terminal) (root))",
        "(boundary (tag 2))",
        "(cboundary (join (tag 2) (region \"dend\")))",
        "(segment-boundaries)",
        "(support (distal (tag 2)))",
        "(proximal-translate (terminal) 20)",
        "(distal-translate (root) 20)",
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

TEST(regloc, reg_nil) {
    auto check = [](const std::string& s) {
        auto res = parse_region_expression(s);
        if (!res.has_value()) throw res.error();
        return true;
    };

    std::vector<std::string>
        args{"(nil)",
             "()",
             "nil",
             "(join () (segment 1)",
             "(intersect (segment 1) nil"};
    for (const auto& arg: args) {
        EXPECT_THROW(check(arg), arborio::label_parse_error);
    }
}

TEST(regloc, loc_nil) {
    auto check = [](const std::string& s) {
        auto res = parse_locset_expression(s);
        if (!res.has_value()) throw res.error();
        return true;
    };

    std::vector<std::string>
        args{"(nil)",
             "()",
             "nil",
             "(join () (root)",
             "(intersect (terminal) nil"};
    for (const auto& arg: args) {
        EXPECT_THROW(check(arg), arborio::label_parse_error);
    }
}

TEST(regloc, reg_fold_expressions) {
    auto check = [](const std::string& s) {
        auto res = parse_region_expression(s);
        if (!res.has_value()) throw res.error();
        return true;
    };

    std::vector<std::string>
        args{"(region-nil) (region-nil)",
             "(region-nil) (segment 1)",
             "(segment 0) (segment 1)",
             "(region-nil) (segment 0) (segment 1)"},
        funs{"join",
             "intersect"};
    for (const auto& fun: funs) {
        for (const auto& arg: args) {

            EXPECT_TRUE(check("(" + fun + " " + arg + ")"));
        }
    }
}

TEST(regloc, loc_fold_expressions) {
    auto check = [](const std::string& s) {
        auto res = parse_locset_expression(s);
        if (!res.has_value()) throw res.error();
        return true;
    };

    std::vector<std::string>
        args{"(locset-nil) (locset-nil)",
             "(locset-nil) (locset-nil) (locset-nil)",
             "(locset-nil) (terminal)",
             "(root) (terminal)",
             "(locset-nil) (root) (terminal)"},
        funs{"sum",
             "join"};
    for (const auto& fun: funs) {
        for (const auto& arg: args) {
            EXPECT_TRUE(check("(" + fun + " " + arg + ")"));
        }
    }
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
using place_tuple = std::tuple<arb::locset, arb::placeable, std::string>;
using paint_pair = std::pair<arb::region, arb::paintable>;
using locset_pair = std::pair<std::string, locset>;
using region_pair = std::pair<std::string, region>;
using iexpr_pair = std::pair<std::string, iexpr>;

std::ostream& operator<<(std::ostream& o, const i_clamp& c) {
    o << "(current-clamp (envelope";
    for (const auto& p: c.envelope) {
        o << " (" << p.t << " " << p.amplitude << ')';
    }
    return o << ") " << c.frequency << ' ' << c.phase << ')';
}
std::ostream& operator<<(std::ostream& o, const threshold_detector& p) {
    return o << "(threshold-detector " << p.threshold << ')';
}
std::ostream& operator<<(std::ostream& o, const init_membrane_potential& p) {
    return o << "(membrane-potential " << p.value << ')';
}
std::ostream& operator<<(std::ostream& o, const temperature_K& p) {
    return o << "(temperature-kelvin " << p.value << ')';
}
std::ostream& operator<<(std::ostream& o, const axial_resistivity& p) {
    return o << "(axial-resistivity " << p.value << ')';
}
std::ostream& operator<<(std::ostream& o, const membrane_capacitance& p) {
    return o << "(membrane-capacitance " << p.value << ')';
}
std::ostream& operator<<(std::ostream& o, const init_int_concentration& p) {
    return o << "(ion-internal-concentration \"" << p.ion << "\" " << p.value << ')';
}
std::ostream& operator<<(std::ostream& o, const ion_diffusivity& p) {
    return o << "(ion-diffusivity \"" << p.ion << "\" " << p.value << ')';
}
std::ostream& operator<<(std::ostream& o, const init_ext_concentration& p) {
    return o << "(ion-external-concentration \"" << p.ion << "\" " << p.value << ')';
}
std::ostream& operator<<(std::ostream& o, const init_reversal_potential& p) {
    return o << "(ion-reversal-potential \"" << p.ion << "\" " << p.value << ')';
}
std::ostream& operator<<(std::ostream& o, const mechanism_desc& m) {
    o << "(mechanism \"" << m.name() << "\"";
    for (const auto& p: m.values()) {
        o << " (\"" << p.first << "\" " << p.second << ')';
    }
    return o << ')';
}
std::ostream& operator<<(std::ostream& o, const junction& p) {
    return o << "(junction " << p.mech << ')';
}
std::ostream& operator<<(std::ostream& o, const synapse& p) {
    return o << "(synapse " << p.mech << ')';
}
std::ostream& operator<<(std::ostream& o, const density& p) {
    return o << "(density " << p.mech << ')';
}
std::ostream& operator<<(std::ostream& o, const voltage_process& p) {
    return o << "(voltage-process " << p.mech << ')';
}
template <typename TaggedMech>
std::ostream& operator<<(std::ostream& o, const scaled_mechanism<TaggedMech>& p) {
    o << "(scaled-mechanism " << p.t_mech;
    for (const auto& it: p.scale_expr) {
        o << " (\"" << it.first << "\" " << it.second << ')';
    }
    o << ")";
    return o;
}
std::ostream& operator<<(std::ostream& o, const ion_reversal_potential_method& p) {
    return o << "(ion-reversal-potential-method \"" << p.ion << "\" " << p.method << ')';
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
std::ostream& operator<<(std::ostream& o, const place_tuple& p) {
    o << "(place " << std::get<0>(p) << " ";
    std::visit([&](auto&& x) {o << x;}, std::get<1>(p));
    return o << " \"" << std::get<2>(p) << "\")";
}
std::ostream& operator<<(std::ostream& o, const defaultable& p) {
    auto default_visitor = arb::util::overload(
        [&](const cv_policy& p)   { o << "(cv-policy " << p << ")"; },
        [&](const auto& p){ o << p; });
    o << "(default ";
    std::visit(default_visitor, p);
    return o << ")";
}
std::ostream& operator<<(std::ostream& o, const locset_pair& p) {
    return o << "(locset-def \"" << p.first << "\" " << p.second << ")";
}
std::ostream& operator<<(std::ostream& o, const region_pair& p) {
    return o << "(region-def \"" << p.first << "\" " << p.second << ")";
}
std::ostream& operator<<(std::ostream& o, const iexpr_pair& p) {
    return o << "(iexpr-def \"" << p.first << "\" " << p.second << ")";
}

template <typename T>
std::string to_string(const T& obj) {
    std::stringstream s;
    s << obj;
    return s.str();
}
std::string to_string(const cv_policy& p) {
    std::stringstream s;
    s << "(cv-policy " << p << ')';
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
std::string round_trip(const char* in) {
    using namespace cable_s_expr;
    if (auto x = arborio::parse_expression(in)) {
        return to_string(std::any_cast<T>(*x));
    }
    else {
        return x.error().what();
    }
}

template <typename T>
std::string round_trip_variant(const char* in) {
    using namespace cable_s_expr;
    if (auto x = arborio::parse_expression(in)) {
        std::string str;
        std::visit([&](auto&& p){str = to_string(p);}, *(eval_cast_variant<T>(*x)));
        return str;
    }
    else {
        return x.error().what();
    }
}

std::string round_trip_component(const char* in) {
    using namespace cable_s_expr;
    if (auto x = arborio::parse_component(in)) {
        return to_string(x.value());
    }
    else {
        return x.error().what();
    }
}

std::string round_trip_component(std::istream& stream) {
    using namespace cable_s_expr;
    if (auto x = arborio::parse_component(stream)) {
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
        "(voltage-process (mechanism \"hh\"))",
        "(density (mechanism \"hh\"))",
        "(density (mechanism \"pas\" (\"g\" 0.02)))",
        "(scaled-mechanism (density (mechanism \"pas\" (\"g\" 0.02))))",
        "(scaled-mechanism (density (mechanism \"pas\" (\"g\" 0.02))) (\"g\" (exp (add (radius 2.1) (scalar 3.2)))))",
    };
    auto default_literals = {
        "(ion-reversal-potential-method \"ca\" (mechanism \"nernst/ca\"))",
        "(cv-policy (single (segment 0)))"
    };
    auto place_literals = {
        "(current-clamp (envelope (10 0.5) (110 0.5) (110 0)) 10 0.25)",
        "(threshold-detector -10)",
        "(junction (mechanism \"gj\"))",
        "(synapse (mechanism \"expsyn\"))"};
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
    auto clamp_literal = "(current-clamp (envelope-pulse 10 5 0.1) 50 0.5)";
    EXPECT_EQ("(current-clamp (envelope (10 0.1) (15 0.1) (15 0)) 50 0.5)", round_trip_variant<placeable>(clamp_literal));

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
        "(paint (cable 2 0.1 0.4) (density (mechanism \"hh\")))",
        "(paint (cable 2 0.1 0.4) (scaled-mechanism (density (mechanism \"pas\" (\"g\" 0.02))) (\"g\" (exp (add (distance 2.1 (region \"my_region\")) (scalar 3.2))))))",
        "(paint (all) (density (mechanism \"pas\" (\"g\" 0.02))))"
    };
    auto decorate_default_literals = {
        "(default (membrane-potential -65.1))",
        "(default (temperature-kelvin 301))",
        "(default (axial-resistivity 102))",
        "(default (membrane-capacitance 0.01))",
        "(default (ion-internal-concentration \"ca\" 75.1))",
        "(default (ion-external-concentration \"h\" -50.1))",
        "(default (ion-reversal-potential \"na\" 30))",
        "(default (ion-reversal-potential-method \"ca\" (mechanism \"nernst/ca\")))",
        "(default (cv-policy (max-extent 2 (region \"soma\") 2)))"
    };
    auto decorate_place_literals = {
        "(place (location 3 0.2) (current-clamp (envelope (10 0.5) (110 0.5) (110 0)) 0.5 0.25) \"clamp\")",
        "(place (terminal) (threshold-detector -10) \"detector\")",
        "(place (root) (junction (mechanism \"gj\")) \"gap_junction\")",
        "(place (locset \"my!ls\") (synapse (mechanism \"expsyn\")) \"synapse\")"};

    for (auto l: decorate_paint_literals) {
        EXPECT_EQ(l, round_trip<paint_pair>(l));
    }
    for (auto l: decorate_place_literals) {
        EXPECT_EQ(l, round_trip<place_tuple>(l));
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
    auto iexpr_def_literals = {
        "(iexpr-def \"my_iexpr\" (radius 1.2))",
        "(iexpr-def \"my_iexpr_2\" (iexpr \"my_iexpr\"))",
    };

    for (auto l: locset_def_literals) {
        EXPECT_EQ(l, round_trip<locset_pair>(l));
    }
    for (auto l: region_def_literals) {
        EXPECT_EQ(l, round_trip<region_pair>(l));
    }
    for (auto l: iexpr_def_literals) {
        EXPECT_EQ(l, round_trip<iexpr_pair>(l));
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
    std::string component_str = "(arbor-component \n"
                                "  (meta-data \n"
                                "    (version \"" + arborio::acc_version() +"\"))\n"
                                "  (decor \n"
                                "    (default \n"
                                "      (axial-resistivity 100.000000))\n"
                                "    (default \n"
                                "      (ion-reversal-potential-method \"na\" \n"
                                "        (mechanism \"nernst\")))\n"
                                "    (default \n"
                                "      (cv-policy \n"
                                "        (fixed-per-branch 10 \n"
                                "          (all)\n"
                                "          1)))\n"
                                "    (paint \n"
                                "      (region \"dend\")\n"
                                "      (density \n"
                                "        (mechanism \"pas\")))\n"
                                "    (paint \n"
                                "      (region \"dend\")\n"
                                "      (scaled-mechanism \n"
                                "        (density \n"
                                "          (mechanism \"pas\"))))\n"
                                "    (paint \n"
                                "      (region \"soma\")\n"
                                "      (scaled-mechanism \n"
                                "        (density \n"
                                "          (mechanism \"pas\"))\n"
                                "        (\"g\" \n"
                                "          (radius 2.1))))\n"
                                "    (paint \n"
                                "      (region \"soma\")\n"
                                "      (density \n"
                                "        (mechanism \"hh\")))\n"
                                "    (paint \n"
                                "      (join \n"
                                "        (tag 1)\n"
                                "        (tag 2))\n"
                                "      (ion-internal-concentration \"ca\" 0.500000))\n"
                                "    (place \n"
                                "      (location 0 0)\n"
                                "      (junction \n"
                                "        (mechanism \"gj\"))\n"
                                "      \"gap-junction\")\n"
                                "    (place \n"
                                "      (location 0 0)\n"
                                "      (threshold-detector 10.000000)\n"
                                "      \"detector\")\n"
                                "    (place \n"
                                "      (location 0 0.5)\n"
                                "      (synapse \n"
                                "        (mechanism \"expsyn\" \n"
                                "          (\"tau\" 1.500000)))\n"
                                "      \"synapse\")))";

    EXPECT_EQ(component_str, round_trip_component(component_str.c_str()));
}

TEST(label_dict, round_tripping) {
    std::string component_str = "(arbor-component \n"
                                "  (meta-data \n"
                                "    (version \"" + arborio::acc_version() + "\"))\n"
                                "  (label-dict \n"
                                "    (iexpr-def \"my_iexpr\" \n"
                                "      (log \n"
                                "        (mul \n"
                                "          (scalar 3.5)\n"
                                "          (diameter 4.3))))\n"
                                "    (region-def \"soma\" \n"
                                "      (tag 1))\n"
                                "    (region-def \"dend\" \n"
                                "      (tag 3))\n"
                                "    (locset-def \"root\" \n"
                                "      (root))))";

    EXPECT_EQ(component_str, round_trip_component(component_str.c_str()));
}

TEST(morphology, round_tripping) {
    std::string component_str = "(arbor-component \n"
                                "  (meta-data \n"
                                "    (version \"" + arborio::acc_version() +"\"))\n"
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

    EXPECT_EQ(component_str, round_trip_component(component_str.c_str()));
}

TEST(morphology, invalid) {
    auto component_str = "(morphology\n"
                         "   (branch 0 -1\n"
                         "     (segment 0 (point 0 0 0 2) (point 4 0 0 2) 1))\n"
                         "   (branch 1 0\n"
                         "     (segment 1 (point 4 0 0 0.8) (point 8 0 0 0.8) 3)\n"
                         "     (segment 2 (point 8 0 0 0.8) (point 12 -0.5 0 0.8) 3))\n"
                         ")";
    EXPECT_THROW(round_trip_component(component_str), arborio::cableio_morphology_error);
}

TEST(cable_cell, round_tripping) {
    std::string component_str = "(arbor-component \n"
                                "  (meta-data \n"
                                "    (version \"" + arborio::acc_version() +"\"))\n"
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
                                "      (iexpr-def \"my_iexpr\" \n"
                                "        (radius 2.1))\n"
                                "      (region-def \"soma\" \n"
                                "        (tag 1))\n"
                                "      (region-def \"dend\" \n"
                                "        (join \n"
                                "          (join \n"
                                "            (tag 3)\n"
                                "            (tag 4))\n"
                                "          (tag 42))))\n"
                                "    (decor \n"
                                "      (paint \n"
                                "        (region \"dend\")\n"
                                "        (density \n"
                                "          (mechanism \"pas\")))\n"
                                "      (paint \n"
                                "        (region \"soma\")\n"
                                "        (density \n"
                                "          (mechanism \"hh\" \n"
                                "            (\"el\" 0.500000))))\n"
                                "      (paint \n"
                                "        (region \"soma\")\n"
                                "        (scaled-mechanism \n"
                                "          (density \n"
                                "            (mechanism \"pas\"))\n"
                                "          (\"g\" \n"
                                "            (iexpr \"my_iexpr\"))))\n"
                                "      (place \n"
                                "        (location 0 1)\n"
                                "        (current-clamp \n"
                                "          (envelope \n"
                                "            (10.000000 0.500000)\n"
                                "            (110.000000 0.500000)\n"
                                "            (110.000000 0.000000))\n"
                                "          0.000000 0.000000)\n"
                                "        \"iclamp\"))))";

    EXPECT_EQ(component_str, round_trip_component(component_str.c_str()));

    std::stringstream stream(component_str);
    EXPECT_EQ(component_str, round_trip_component(stream));
}

TEST(cable_cell_literals, errors) {
    for (auto expr: {"(membrane-potential \"56\")",  // invalid argument
                     "(axial-resistivity 1 2)",      // too many arguments to otherwise valid decor literal
                     "(membrane-capacitance ",       // syntax error
                     "(mem-capacitance 3.5)",        // invalid function
                     "(ion-initial-concentration ca 0.1)",   // unquoted ion
                     "(density (mechanism \"hh\" (gl 3.5)))",// unqouted parameter
                     "(density (mechanism \"pas\" ((\"g\" 0.5) (\"e\" 0.2))))",   // paranthesis around params
                     "(density (mechanism \"pas\" (\"g\" 0.5 0.1) (\"e\" 0.2)))", // too many values
                     "(current-clamp (envelope (10 0.5) (110 0.5) (110 0)))",     // too few arguments
                     "(current-clamp (envelope (10 0.5) (110 0.5) (110 0)) 10)",  // too few arguments
                     "(paint (region) (mechanism \"hh\"))",  // invalid region
                     "(paint (tag 1) (mechanims hh))",       // invalid painting
                     "(paint (terminal) (membrance-capacitance 0.2))", // can't paint a locset
                     "(paint (tag 3))",                      // too few arguments
                     "(place (locset) (junction (mechanism \"gj\")) \"gj\")",        // invalid locset
                     "(place (junction (mechanism \"gj\")) (location 0 1), \"gj\")", // swapped argument order
                     "(place (location 0 1) (mechanism \"expsyn\"))",                // missing synapse
                     "(place (location 0 1) (synapse (mechanism \"expsyn\")))",      // missing label
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
    for (const std::string& expr: std::vector<std::string>{
         "(arbor-component (meta-data (version \"0.1\")) (decor))",  // invalid component
         "(arbor-component (morphology))", // arbor-component missing meta-data
         "(arbor-component (meta-data (version \"" + arborio::acc_version() +"\")))", // arbor-component missing component
         "(arbor-component (meta-data (version \"" + arborio::acc_version() +"\")) (membrane-potential 56))",  // invalid component
         "(arbor-component (meta-data (version \"" + arborio::acc_version() +"\")) (morphology (segment 1 (point 1 2 3 4) (point 2 3 4 5) 3)))", // morphology with segment instead of branch
         "(arbor-component (meta-data (version \"" + arborio::acc_version() +"\")) (decor (region-def \"reg\" (tag 3))))", // decor with label definition
         "(arbor-component (meta-data (version \"" + arborio::acc_version() +"\")) (cable-cell (paint (tag 3) (axial-resistivity 3.1))))", // cable-cell with paint
         "(morphology (branch 0 -1 (segment 0 (point 0 1 2 3 ) (point 1 2 3 4) 3)))" // morphology without arbor-component
    })
    {
        // If an expression was passed and handled correctly the parse call will
        // return without throwing, with the error stored in the return type.
        // So it is sufficient to assert that it evaluates to false.
        EXPECT_FALSE(arborio::parse_component(expr));
    }
}

// Check that the examples used in the docs are valid (formats/cable_cell.rst)
TEST(doc_expressions, parse) {
    // literals
    for (auto expr: {"(region-def \"my_region\" (branch 1))",
                     "(locset-def \"my_locset\" (location 3 0.5))",
                     "(mechanism \"hh\" (\"gl\" 0.5) (\"el\" 2))",
                     "(ion-reversal-potential-method \"ca\" (mechanism \"nernst/ca\"))",
                     "(current-clamp (envelope (0 10) (50 10) (50 0)) 40 0.25)",
                     "(paint (tag 1) (membrane-capacitance 0.02))",
                     "(place (locset \"mylocset\") (threshold-detector 10) \"mydetectors\")",
                     "(default (membrane-potential -65))",
                     "(segment 3 (point 0 0 0 5) (point 0 0 10 2) 1)"})
    {
        EXPECT_TRUE(arborio::parse_expression(expr));
    }

    // objects
    for (auto expr: {"(label-dict"
                     "  (region-def \"my_soma\" (tag 1))\n"
                     "  (locset-def \"root\" (root))\n"
                     "  (region-def \"all\" (all))\n"
                     "  (region-def \"my_region\" (radius-ge (region \"my_soma\") 1.5))\n"
                     "  (locset-def \"terminal\" (terminal)))",
                     "(decor\n"
                     "  (default (membrane-potential -55.000000))\n"
                     "  (paint (region \"custom\") (temperature-kelvin 270))\n"
                     "  (paint (region \"soma\") (membrane-potential -50.000000))\n"
                     "  (paint (all) (density (mechanism \"pas\")))\n"
                     "  (paint (tag 4) (density (mechanism \"Ih\" (\"gbar\" 0.001))))\n"
                     "  (place (locset \"root\") (synapse (mechanism \"expsyn\")) \"root_synapse\")\n"
                     "  (place (terminal) (junction (mechanism \"gj\")) \"terminal_gj\"))",
                     "(morphology\n"
                     "  (branch 0 -1\n"
                     "    (segment 0 (point 0 0 0 2) (point 4 0 0 2) 1)\n"
                     "    (segment 1 (point 4 0 0 0.8) (point 8 0 0 0.8) 3)\n"
                     "    (segment 2 (point 8 0 0 0.8) (point 12 -0.5 0 0.8) 3))\n"
                     "  (branch 1 0\n"
                     "    (segment 3 (point 12 -0.5 0 0.8) (point 20 4 0 0.4) 3)\n"
                     "    (segment 4 (point 20 4 0 0.4) (point 26 6 0 0.2) 3))\n"
                     "  (branch 2 0\n"
                     "    (segment 5 (point 12 -0.5 0 0.5) (point 19 -3 0 0.5) 3))\n"
                     "  (branch 3 2\n"
                     "    (segment 6 (point 19 -3 0 0.5) (point 24 -7 0 0.2) 3))\n"
                     "  (branch 4 2\n"
                     "    (segment 7 (point 19 -3 0 0.5) (point 23 -1 0 0.2) 3)\n"
                     "    (segment 8 (point 23 -1 0 0.3) (point 26 -2 0 0.2) 3))\n"
                     "  (branch 5 -1\n"
                     "    (segment 9 (point 0 0 0 2) (point -7 0 0 0.4) 2)\n"
                     "    (segment 10 (point -7 0 0 0.4) (point -10 0 0 0.4) 2)))",
                     "(cable-cell\n"
                     "  (label-dict\n"
                     "    (region-def \"my_soma\" (tag 1))\n"
                     "    (locset-def \"root\" (root))\n"
                     "    (region-def \"all\" (all))\n"
                     "    (region-def \"my_region\" (radius-ge (region \"my_soma\") 1.5))\n"
                     "    (locset-def \"terminal\" (terminal)))\n"
                     "  (decor\n"
                     "    (default (membrane-potential -55.000000))\n"
                     "    (paint (region \"my_soma\") (temperature-kelvin 270))\n"
                     "    (paint (region \"my_region\") (membrane-potential -50.000000))\n"
                     "    (paint (tag 4) (density (mechanism \"Ih\" (\"gbar\" 0.001))))\n"
                     "    (place (locset \"root\") (synapse (mechanism \"expsyn\")) \"root_synapse\")\n"
                     "    (place (location 1 0.2) (junction (mechanism \"gj\")) \"terminal_gj\"))\n"
                     "  (morphology\n"
                     "    (branch 0 -1\n"
                     "      (segment 0 (point 0 0 0 2) (point 4 0 0 2) 1)\n"
                     "      (segment 1 (point 4 0 0 0.8) (point 8 0 0 0.8) 3)\n"
                     "      (segment 2 (point 8 0 0 0.8) (point 12 -0.5 0 0.8) 3))\n"
                     "    (branch 1 0\n"
                     "      (segment 3 (point 12 -0.5 0 0.8) (point 20 4 0 0.4) 3)\n"
                     "      (segment 4 (point 20 4 0 0.4) (point 26 6 0 0.2) 3))\n"
                     "    (branch 2 0\n"
                     "      (segment 5 (point 12 -0.5 0 0.5) (point 19 -3 0 0.5) 3))\n"
                     "    (branch 3 2\n"
                     "      (segment 6 (point 19 -3 0 0.5) (point 24 -7 0 0.2) 3))\n"
                     "    (branch 4 2\n"
                     "      (segment 7 (point 19 -3 0 0.5) (point 23 -1 0 0.2) 3)\n"
                     "      (segment 8 (point 23 -1 0 0.3) (point 26 -2 0 0.2) 3))\n"
                     "    (branch 5 -1\n"
                     "      (segment 9 (point 0 0 0 2) (point -7 0 0 0.4) 2)\n"
                     "      (segment 10 (point -7 0 0 0.4) (point -10 0 0 0.4) 2))))"})
    {
            auto t = arborio::parse_expression(expr);
            if (!t) {
                std::cout << t.error().what() << std::endl;
            }
        EXPECT_TRUE(arborio::parse_expression(expr));
    }

    // components
    for (std::string expr: {"(arbor-component\n"
                            "  (meta-data (version \"" + arborio::acc_version() +"\"))\n"
                            "  (label-dict\n"
                            "    (region-def \"my_soma\" (tag 1))\n"
                            "    (locset-def \"root\" (root))))",
                            "(arbor-component\n"
                            "  (meta-data (version \"" + arborio::acc_version() +"\"))\n"
                            "  (decor\n"
                            "    (default (membrane-potential -55.000000))\n"
                            "    (place (locset \"root\") (synapse (mechanism \"expsyn\")) \"root_synapse\")\n"
                            "    (paint (region \"my_soma\") (temperature-kelvin 270))))",
                            "(arbor-component\n"
                            "  (meta-data (version \"" + arborio::acc_version() +"\"))\n"
                            "  (morphology\n"
                            "     (branch 0 -1\n"
                            "       (segment 0 (point 0 0 0 2) (point 4 0 0 2) 1)\n"
                            "       (segment 1 (point 4 0 0 0.8) (point 8 0 0 0.8) 3)\n"
                            "       (segment 2 (point 8 0 0 0.8) (point 12 -0.5 0 0.8) 3))))\n",
                            "(arbor-component\n"
                            "  (meta-data (version \"" + arborio::acc_version() +"\"))\n"
                            "  (cable-cell\n"
                            "    (label-dict\n"
                            "      (region-def \"my_soma\" (tag 1))\n"
                            "      (locset-def \"root\" (root)))\n"
                            "    (decor\n"
                            "      (default (membrane-potential -55.000000))\n"
                            "      (place (locset \"root\") (synapse (mechanism \"expsyn\")) \"root_synapse\")\n"
                            "      (paint (region \"my_soma\") (temperature-kelvin 270)))\n"
                            "    (morphology\n"
                            "       (branch 0 -1\n"
                            "         (segment 0 (point 0 0 0 2) (point 4 0 0 2) 1)\n"
                            "         (segment 1 (point 4 0 0 0.8) (point 8 0 0 0.8) 3)\n"
                            "         (segment 2 (point 8 0 0 0.8) (point 12 -0.5 0 0.8) 3)))))"})
    {
        EXPECT_TRUE(arborio::parse_expression(expr));
    }
}
