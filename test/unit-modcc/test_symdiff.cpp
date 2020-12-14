#include <cmath>

#include "common.hpp"

#include "symdiff.hpp"
#include "parser.hpp"

// Test visitors for extended constant reduction,
// identifier presence detection and symbolic differentiation.

TEST(involves_identifier, line_expr) {
    const char* expr_x[] = {
        "a=4+x",
        "x'=a",
        "a=exp(x)",
        "x=sin(2)",
        "~ 2x <-> (1, a)"
    };

    const char* expr_xy[] = {
        "a=4+x+y",
        "x'=exp(x*sin(y))",
        "x=sin(2*y)",
        "~ 2x <-> 3y (1, 1)"
    };

    const char* expr_xyz[] = {
        "a=4+x+y*log(z)",
        "x'=exp(x*sin(y))+func(2,z)",
        "x=sin(2*y)+(-z)",
        "~ 2x <-> 3y (a, z)"
    };

    identifier_set xyz_ids = { "x", "y", "z" };
    identifier_set yz_ids = { "y", "z" };
    identifier_set uvw_ids = { "u", "v", "w" };

    for (auto line: expr_x) {
        SCOPED_TRACE(std::string("expression: ")+line);
        Parser p(line);
        auto e = p.parse_statement();
        ASSERT_TRUE(e);

        EXPECT_TRUE(involves_identifier(e, "x"));
        EXPECT_FALSE(involves_identifier(e, "y"));
        EXPECT_FALSE(involves_identifier(e, "z"));

        EXPECT_TRUE(involves_identifier(e, xyz_ids));
        EXPECT_FALSE(involves_identifier(e, yz_ids));
        EXPECT_FALSE(involves_identifier(e, uvw_ids));
    }

    for (auto line: expr_xy) {
        SCOPED_TRACE(std::string("expression: ")+line);
        Parser p(line);
        auto e = p.parse_statement();
        ASSERT_TRUE(e);

        EXPECT_TRUE(involves_identifier(e, "x"));
        EXPECT_TRUE(involves_identifier(e, "y"));
        EXPECT_FALSE(involves_identifier(e, "z"));

        EXPECT_TRUE(involves_identifier(e, xyz_ids));
        EXPECT_TRUE(involves_identifier(e, yz_ids));
        EXPECT_FALSE(involves_identifier(e, uvw_ids));
    }

    for (auto line: expr_xyz) {
        SCOPED_TRACE(std::string("expression: ")+line);
        Parser p(line);
        auto e = p.parse_statement();
        ASSERT_TRUE(e);

        EXPECT_TRUE(involves_identifier(e, "x"));
        EXPECT_TRUE(involves_identifier(e, "y"));
        EXPECT_TRUE(involves_identifier(e, "z"));

        EXPECT_TRUE(involves_identifier(e, xyz_ids));
        EXPECT_TRUE(involves_identifier(e, yz_ids));
        EXPECT_FALSE(involves_identifier(e, uvw_ids));
    }
}

TEST(constant_simplify, constants) {
    struct { const char* repn; double value; } tests[] = {
        { "17+3",               20. },
        { "log(exp(2))+cos(0)", 3. },
        { "0/17-1",             -1. },
        { "2.5*(34/17-1.0e2)",  -245. },
        { "-sin(0.523598775598298873077107230546583814)", -0.5 }
    };

    for (const auto& item: tests) {
        SCOPED_TRACE(std::string("expression: ")+item.repn);
        Parser p(item.repn);
        auto e = p.parse_expression();
        ASSERT_TRUE(e);

        auto value = expr_value(constant_simplify(e));
        EXPECT_FALSE(std::isnan(value));
        EXPECT_NEAR(item.value, value, 1e-8);
    }
}

TEST(constant_simplify, simplified_expr) {
    // Expect simplification of 'before' expression matches parse of 'after'.
    // Use output string representation of expression for easy comparison.

    struct { const char* before; const char* after; } tests[] = {
        { "x+y/z",             "x+y/z" },
        { "(0*x)+y/(z-(0*w))", "y/z" },
        { "y*exp(0)",          "y" },
        { "x^(2-1)",           "x" },
        { "0-(y+0)",           "-y" }
    };

    for (const auto& item: tests) {
        SCOPED_TRACE(std::string("expressions: ")+item.before+"; "+item.after);
        auto before = Parser{item.before}.parse_expression();
        auto after = Parser{item.after}.parse_expression();
        ASSERT_TRUE(before);
        ASSERT_TRUE(after);

        EXPECT_EXPR_EQ(after, constant_simplify(before));
    }
}

TEST(constant_simplify, block_with_if) {
    const char* before_repn =
        "{\n"
        "    x = 0/z - y*log(1) + w*(3-2)\n"
        "    if (2>1) {\n"
        "        if (1==3) {\n"
        "            y = 64\n"
        "        }\n"
        "        else {\n"
        "            y = exp(0)*x\n"
        "            z = y-(x*0)\n"
        "        }\n"
        "    }\n"
        "    else {\n"
        "        y = 32\n"
        "    }\n"
        "}\n";

    const char* after_repn =
        "{\n"
        "    x = w\n"
        "    y = x\n"
        "    z = y\n"
        "}\n";

    auto before = Parser{before_repn}.parse_block(false);
    auto after = Parser{after_repn}.parse_block(false);
    ASSERT_TRUE(before);
    ASSERT_TRUE(after);

    EXPECT_EXPR_EQ(after, constant_simplify(before));
}

TEST(symbolic_pdiff, expressions) {
    struct { const char* before; const char* after; } tests[] = {
        { "y+z*4",     "0" },
        { "x",         "1" },
        { "x*3",       "3"},
        { "x-log(x)",  "1-1/x"},
        { "sin(x)",    "cos(x)"},
        { "cos(x)",    "-sin(x)"},
        { "exp(x)",    "exp(x)"},
    };

    for (const auto& item: tests) {
        SCOPED_TRACE(std::string("expressions: ")+item.before+"; "+item.after);
        auto before = Parser{item.before}.parse_expression();
        auto after = Parser{item.after}.parse_expression();
        ASSERT_TRUE(before);
        ASSERT_TRUE(after);

        EXPECT_EXPR_EQ(after, symbolic_pdiff(before, "x"));
    }
}

TEST(symbolic_pdiff, linear) {
    struct { const char* before; const char* after; } tests[] = {
        { "x+y/z",               "1" },
        { "3.0*x-x/2.0+x*y",     "2.5+y"},
        { "(1+2.*x)/(1-exp(y))", "2./(1-exp(y))"}
    };

    for (const auto& item: tests) {
        SCOPED_TRACE(std::string("expressions: ")+item.before+"; "+item.after);
        auto before = Parser{item.before}.parse_expression();
        auto after = Parser{item.after}.parse_expression();
        ASSERT_TRUE(before);
        ASSERT_TRUE(after);

        EXPECT_EXPR_EQ(after, symbolic_pdiff(before, "x"));
    }
}

TEST(symbolic_pdiff, nonlinear) {
    struct { const char* before; const char* after; } tests[] = {
        { "sin(x)",      "cos(x)" },
        { "exp(2*x)",    "2*exp(2*x)" },
        { "x^2",         "2*x" },
        { "a^x",         "log(a)*a^x" }
    };

    for (const auto& item: tests) {
        SCOPED_TRACE(std::string("expressions: ")+item.before+"; "+item.after);
        auto before = Parser{item.before}.parse_expression();
        auto after = Parser{item.after}.parse_expression();
        ASSERT_TRUE(before);
        ASSERT_TRUE(after);

        EXPECT_EXPR_EQ(after, symbolic_pdiff(before, "x"));
    }
}

TEST(symbolic_pdiff, non_differentiable) {
    struct { const char* exp; } tests[] = {
            { "max(x)"},
            { "min(a)"}
    };

    for (const auto& item: tests) {
        SCOPED_TRACE(std::string("expressions: ")+item.exp);
        auto exp = Parser{item.exp}.parse_expression();
        ASSERT_FALSE(exp);
    }
}

inline expression_ptr operator""_expr(const char* literal, std::size_t) {
    return Parser{literal}.parse_expression();
}

TEST(substitute, expressions) {
    struct { const char* before; const char* after; } tests[] = {
        { "x",             "y+z" },
        { "y",             "y"  },
        { "2.0",           "2.0" },
        { "sin(x)",        "sin(y+z)" },
        { "func(x,3+x)+y", "func(y+z,3+(y+z))+y" }
    };

    auto yplusz = "y+z"_expr;
    ASSERT_TRUE(yplusz);

    for (const auto& item: tests) {
        SCOPED_TRACE(std::string("expressions: ")+item.before+"; "+item.after);
        auto before = Parser{item.before}.parse_expression();
        auto after = Parser{item.after}.parse_expression();
        ASSERT_TRUE(before);
        ASSERT_TRUE(after);

        auto result = substitute(before.get(), "x", yplusz.get());
        EXPECT_EXPR_EQ(after, result);
    }
}

TEST(substitute, exprmap) {
    substitute_map subs;
    subs["x"] = "sin(y)"_expr;
    subs["y"] = "cos(x)"_expr;

    // note substitute is not recursive!
    auto before = "exp(x+y)"_expr;
    ASSERT_TRUE(before);

    auto after = "exp(sin(y)+cos(x))"_expr;
    ASSERT_TRUE(after);

    auto result = substitute(before.get(), subs);
    EXPECT_EXPR_EQ(after, result);
}

TEST(linear_test, homogeneous) {
    linear_test_result r;

    r = linear_test("3*x"_expr, {"x"});
    EXPECT_TRUE(r.is_linear);
    EXPECT_TRUE(r.is_homogeneous);
    EXPECT_TRUE(r.monolinear());
    EXPECT_EXPR_EQ(r.coef["x"], "3"_expr);

    r = linear_test("y-a*x+2*x"_expr, {"x", "y"});
    EXPECT_TRUE(r.is_linear);
    EXPECT_TRUE(r.is_homogeneous);
    EXPECT_FALSE(r.monolinear());
    EXPECT_EXPR_EQ(r.coef["x"], "-a+2"_expr);
    EXPECT_EXPR_EQ(r.coef["y"], "1"_expr);
}

TEST(linear_test, inhomogeneous) {
    linear_test_result r;

    r = linear_test("sin(y)+3*x"_expr, {"x"});
    EXPECT_TRUE(r.is_linear);
    EXPECT_FALSE(r.is_homogeneous);
    EXPECT_EXPR_EQ(r.coef["x"], "3"_expr);
    EXPECT_EXPR_EQ(r.constant, "sin(y)"_expr);

    r = linear_test("(x+y+1)*(a+b)"_expr, {"x", "y"});
    EXPECT_TRUE(r.is_linear);
    EXPECT_FALSE(r.is_homogeneous);
    EXPECT_EXPR_EQ(r.coef["x"], "a+b"_expr);
    EXPECT_EXPR_EQ(r.coef["y"], "a+b"_expr);
    EXPECT_EXPR_EQ(r.constant, "a+b"_expr);

    // check 'gating' case still works! (Use plus instead of minus
    // though because of -1 vs (- 1) parsing makes the test harder.)
    r = linear_test("(a+x)/b"_expr, {"x"});
    EXPECT_TRUE(r.is_linear);
    EXPECT_FALSE(r.is_homogeneous);
    EXPECT_EXPR_EQ(r.coef["x"], "1/b"_expr);
    EXPECT_EXPR_EQ(r.constant, "a/b"_expr);
}

TEST(linear_test, nonlinear) {
    linear_test_result r;

    r = linear_test("x+x^2"_expr, {"x", "y"});
    EXPECT_FALSE(r.is_linear);

    r = linear_test("x+y*x"_expr, {"x", "y"});
    EXPECT_FALSE(r.is_linear);
}

TEST(linear_test, non_differentiable) {
    linear_test_result r;

    r = linear_test("max(x, y)"_expr, {"x", "y"});
    EXPECT_FALSE(r.is_linear);
    EXPECT_FALSE(r.is_homogeneous);

    r = linear_test("min(x, y)"_expr, {"x", "y"});
    EXPECT_FALSE(r.is_linear);
    EXPECT_FALSE(r.is_homogeneous);
}

TEST(linear_test, diagonality) {
    auto xdot = "a*x"_expr;
    auto ydot = "-b*y/2"_expr;
    auto zdot = "x+y+z"_expr;

    // xdot, ydot diagonal linear
    EXPECT_TRUE(linear_test(xdot, {"x", "y"}).monolinear("x"));
    EXPECT_TRUE(linear_test(ydot, {"x", "y"}).monolinear("y"));

    // but xdot, ydot, zdot not diagonal
    EXPECT_TRUE(linear_test(xdot, {"x", "y", "z"}).monolinear("x"));
    EXPECT_TRUE(linear_test(ydot, {"x", "y", "z"}).monolinear("y"));
    EXPECT_FALSE(linear_test(zdot, {"x", "y", "z"}).monolinear("z"));
}
