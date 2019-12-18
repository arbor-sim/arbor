#include <cmath>
#include <memory>

#include "common.hpp"

#include "expression.hpp"
#include "functionexpander.hpp"
#include "functioninliner.hpp"
#include "parser.hpp"
#include "scope.hpp"

// Test FunctionCallLowerer

scope_ptr mock_symbols() {
    static scope_type::symbol_map symbols;
    if (symbols.empty()) {
        std::string globals[] = {
            "a", "b", "c"
        };

        std::string procdefs[] = {
            "PROCEDURE p0() { a = 3 }",
            "PROCEDURE p1(x) { a = x }",
            "PROCEDURE p2(x, y) { a = g(x, y+3) }",
        };

        std::string fndefs[] = {
            "FUNCTION f(p, q, r) { f = (q-p)/(r-p) }",
            "FUNCTION g(a, b) { g = a+b }",
            "FUNCTION h(x) { h = exp(x) }",
            "FUNCTION long(x) {\n"
            "    LOCAL y, z\n"
            "    y = h(x)\n"
            "    z = g(y, x)\n"
            "    if (y>z) { long = f(x, 3, z)\n }\n"
            "    else { long = h(z)\n }\n"
            "}",
            "FUNCTION assign2(x) {\n"
            "    assign2 = x * 2\n"
            "    if (x<2) { assign2 = 2\n }\n"
            "}",
            "FUNCTION recurse1(x) {\n"
            "    LOCAL y\n"
            "    if (x<2) { recurse1 = x\n }\n"
            "    else { y = x/2\n recurse1 = recurse2(y)\n }\n"
            "}",
            "FUNCTION recurse2(x) {\n"
            "    LOCAL y\n"
            "    if (x<2) { recurse2 = x\n }\n"
            "    else { y = x/5\n recuse2 = recurse1(y)\n }\n"
            "}",
            "FUNCTION shadow(x) {\n"
            "    LOCAL x\n"
            "    x = 2\n"
            "    shadow = 1\n"
            "}\n"
        };

        for (auto& g: globals) {
            auto var = new VariableExpression(Location{}, g);
            var->visibility(visibilityKind::global);
            symbols[g] = symbol_ptr{std::move(var)};
        }

        for (auto& f: fndefs) {
            auto s = Parser{f}.parse_function();
            symbols[s->name()] = std::move(s);
        }

        for (auto& p: procdefs) {
            auto s = Parser{p}.parse_procedure();
            symbols[s->name()] = std::move(s);
        }
    }

    scope_ptr scp = std::make_shared<scope_type>(symbols);
    scp->in_api_context(false);
    return scp;
}

expression_ptr parse_block(const std::string& defn, scope_ptr syms, bool inner = false) {
    Parser parser{defn};
    expression_ptr expr = parser.parse_block(inner);
    EXPECT_TRUE(expr) << parser.error_message();
    if (expr) expr->semantic(syms);
    return expr;
}

TEST(lower_functions, simple) {
    // If there are no call arguments to be lowered, expect block
    // to be unchanged.

    const char* tests[] = {
        "{ a = b + c\n b = g(a, c)\n }",
        "{ c = h(a)\n b = g(b, c)\n a = b + c\n c = f(a, b, c)\n }"
    };

    for (auto& defn: tests) {
        auto bexpr = parse_block(defn, mock_symbols());
        ASSERT_TRUE(bexpr);
        auto block = bexpr->is_block();
        ASSERT_TRUE(block);

        EXPECT_EXPR_EQ(block, lower_functions(block));
    }
}

// Note: ordering and name of introduced locals is dependent-upon
// the implementation.

TEST(lower_functions, compound_args) {
    struct { const char *before, *after; } tests[] = {
        {
            "{ a = g(a, a + b)\n }",
            "{ LOCAL ll0_\n"
            "  ll0_ = a + b\n"
            "  a = g(a, ll0_)\n }"
        },
        {
            "{ a = f(1, 2, a)\n }",
            "{ a = f(1, 2, a)\n }"
            // But possibly should be:
            // "{ LOCAL ll0_\n ll0_ = f(1, 2, a)\n a = ll0_\n }"
        },
        {
            "{ a = h(b + c)\n"
            "  b = g(a + b, b)\n"
            "  c = f(a, b, c)\n }",
            "{ LOCAL ll1_\n"
            "  LOCAL ll0_\n"
            "  ll0_ = b + c\n"
            "  a = h(ll0_)\n"
            "  ll1_ = a + b\n"
            "  b = g(ll1_, b)\n"
            "  c = f(a, b, c)\n }"
        }
    };

    auto syms = mock_symbols();
    for (auto& test: tests) {
        auto expr1 = parse_block(test.before, mock_symbols());
        ASSERT_TRUE(expr1);
        auto before = expr1->is_block();
        ASSERT_TRUE(before);

        auto expr2 = parse_block(test.after, mock_symbols());
        ASSERT_TRUE(expr2);
        auto expected = expr2->is_block();
        ASSERT_TRUE(expected);

        EXPECT_EXPR_EQ(expected, lower_functions(before));
    }
}

TEST(lower_functions, compound_rhs) {
    struct { const char *before, *after; } tests[] = {
        {
            "{ a = f(b, c) + g(a, a + b)\n }",
            "{ LOCAL ll2_\n"
            "  LOCAL ll1_\n"
            "  LOCAL ll0_\n"
            "  ll0_ = f(b, c)\n"
            "  ll1_ = a + b\n"
            "  ll2_ = g(a, ll1_)\n"
            "  a = ll0_ + ll2_\n }"
        },
        {
            "{ a = log(exp(h(b)))\n }",
            "{ LOCAL ll0_\n"
            "  ll0_ = h(b)\n"
            "  a = log(exp(ll0_))\n }"
        }
    };

    auto syms = mock_symbols();
    for (auto& test: tests) {
        auto expr1 = parse_block(test.before, mock_symbols());
        ASSERT_TRUE(expr1);
        auto before = expr1->is_block();
        ASSERT_TRUE(before);

        auto expr2 = parse_block(test.after, mock_symbols());
        ASSERT_TRUE(expr2);
        auto expected = expr2->is_block();
        ASSERT_TRUE(expected);

        EXPECT_EXPR_EQ(expected, lower_functions(before));
    }
}

TEST(lower_functions, nested_calls) {
    struct { const char *before, *after; } tests[] = {
        {
            "{ a = h(g(a, a + b))\n }",
            "{ LOCAL ll1_\n"
            "  LOCAL ll0_\n"
            "  ll0_ = a + b\n"
            "  ll1_ = g(a, ll0_)\n"
            "  a = h(ll1_)\n }"
        },
        {
            "{ p2(g(a, b), h(h(c)))\n }",
            "{ LOCAL ll2_\n"
            "  LOCAL ll1_\n"
            "  LOCAL ll0_\n"
            "  ll0_ = g(a, b)\n"
            "  ll1_ = h(c)\n"
            "  ll2_ = h(ll1_)\n"
            "  p2(ll0_, ll2_)\n }"
        }
    };

    auto syms = mock_symbols();
    for (auto& test: tests) {
        auto expr1 = parse_block(test.before, mock_symbols());
        ASSERT_TRUE(expr1);
        auto before = expr1->is_block();
        ASSERT_TRUE(before);

        auto expr2 = parse_block(test.after, mock_symbols());
        ASSERT_TRUE(expr2);
        auto expected = expr2->is_block();
        ASSERT_TRUE(expected);

        EXPECT_EXPR_EQ(expected, lower_functions(before));
    }
}

TEST(lower_functions, ifexpr) {
    struct { const char *before, *after; } tests[] = {

    // Can't test current implementation without shenanigans,
    // as LOCAL declarations will not be parsed in nested blocks.
#if 0
        {
            "{ if (a>1) { p1(h(a))\n } else { p1(2+a)\n }\n }",
            "{ if (a>1) {\n"
            "      LOCAL ll0_\n"
            "      ll0_ = h(a)\n"
            "      p1(ll0_)\n"
            "  } else {\n"
            "      LOCAL ll1_\n"
            "      ll1_ = 2 + a\n"
            "      p1(ll1_)\n"
            "  } } \n"
        },
#endif
        {
            "{ if (f(a, 1, c)) { p1(a)\n }\n }",
            "{ LOCAL ll0_\n"
            "  ll0_ = f(a, 1, c)\n"
            "  if (ll0_) { p1(a)\n }\n }"
        },
        {
            "{ if (h(a + 2) > 1) { p1(a)\n }\n }",
            "{ LOCAL ll1_\n"
            "  LOCAL ll0_\n"
            "  ll0_ = a + 2\n"
            "  ll1_ = h(ll0_)\n"
            "  if (ll1_ > 1) { p1(a)\n }\n }"
        }
    };

    auto syms = mock_symbols();
    for (auto& test: tests) {
        auto expr1 = parse_block(test.before, mock_symbols());
        ASSERT_TRUE(expr1);
        auto before = expr1->is_block();
        ASSERT_TRUE(before);

        auto expr2 = parse_block(test.after, mock_symbols());
        ASSERT_TRUE(expr2);
        auto expected = expr2->is_block();
        ASSERT_TRUE(expected);

        EXPECT_EXPR_EQ(expected, lower_functions(before));
    }
}

// Note: function inliner should only be run after all
// function calls in the block have been lowered.

TEST(inline_functions, simple) {
    // There should be no changes to blocks without
    // function calls.

    const char* tests[] = {
        "{ a = b + c\n p2(a, x)\n }",
        "{ if (a>b) { p1(exp(a))\n }\n }"
    };

    for (auto& defn: tests) {
        auto bexpr = parse_block(defn, mock_symbols());
        ASSERT_TRUE(bexpr);
        auto block = bexpr->is_block();
        ASSERT_TRUE(block);

        EXPECT_EXPR_EQ(block, inline_function_calls(block));
    }
}

TEST(inline_functions, compound) {
    // Check inlining with 'long' function that includes
    // locals, further function calls, and an if clause.

    const char* before_defn =
        "{ a = 2\n"
        "  if (b>3) { b = h(2)\n }\n"
        "  p2(a, b)\n"
        "  c = long(c)\n"
        "  a = long(b)\n }";

    const char* after_defn =
        "{ a = 2\n"
        "  if (b>3) { b = exp(2)\n }\n"
        "  p2(a, b)\n"
        "  LOCAL r_0_, r_1_\n"
        "  r_0_ = exp(c)\n"
        "  r_1_ = r_0_ + c\n"
        "  if (r_0_>r_1_) { c = (3-c)/(r_1_-c)\n }\n"
        "  else { c = exp(r_1_)\n }\n"
        "  LOCAL r_2_, r_3_\n"
        "  r_2_ = exp(b)\n"
        "  r_3_ = r_2_ + b\n"
        "  if (r_2_>r_3_) { a = (3-b)/(r_3_-b)\n }\n"
        "  else { a = exp(r_3_)\n }\n }";

    auto expr1 = parse_block(before_defn, mock_symbols());
    ASSERT_TRUE(expr1);
    auto before = expr1->is_block();
    ASSERT_TRUE(before);

    auto expr2 = parse_block(after_defn, mock_symbols());
    ASSERT_TRUE(expr2);
    auto expected = expr2->is_block();
    ASSERT_TRUE(expected);

    EXPECT_EXPR_EQ(expected, inline_function_calls(before));
}

TEST(inline_functions, twice_assign) {
    // What happens if function return variable is assigned twice?
    // Test uses mocked definition:
    //
    //      FUNCTION assign2(x) {
    //          assign2 = x * 2
    //          if (x<2) {
    //              assign2 = 2
    //          }
    //      }
    //
    // Expect that the function lowerer will take care of this.

    const char* before_defn =
        "{ a = assign2(a)\n }";

    const char* after_defn =
        "{ LOCAL ll0_\n"
        "  ll0_ = a * 2\n"
        "  if (a<2) {\n"
        "      ll0_ = 2\n"
        "  }\n"
        "  a = ll0_\n }";

    auto expr1 = parse_block(before_defn, mock_symbols());
    ASSERT_TRUE(expr1);
    auto before = expr1->is_block();
    ASSERT_TRUE(before);

    auto lowered_expr = lower_functions(before);
    ASSERT_TRUE(lowered_expr);
    auto lowered = lowered_expr->is_block();
    ASSERT_TRUE(lowered);

    auto expr2 = parse_block(after_defn, mock_symbols());
    ASSERT_TRUE(expr2);
    auto expected = expr2->is_block();
    ASSERT_TRUE(expected);

    EXPECT_EXPR_EQ(expected, inline_function_calls(lowered));
}

TEST(inline_functions, recursion) {
    // What happens if we try to inline a recursive function?
    // We should get an error, but error should mention recursion.

    const char* before_defn =
        "{ a = recurse1(b)\n }";

    const char* after_defn =
        "{ \n }";

    auto expr1 = parse_block(before_defn, mock_symbols());
    ASSERT_TRUE(expr1);
    auto before = expr1->is_block();
    ASSERT_TRUE(before);

    auto expr2 = parse_block(after_defn, mock_symbols());
    ASSERT_TRUE(expr2);
    auto expected = expr2->is_block();
    ASSERT_TRUE(expected);

    EXPECT_EXPR_EQ(expected, inline_function_calls(before));
}

TEST(inline_functions, local_shadow) {
    // shadow(x) has a local x inside, shadowing the parameter:
    //
    //      FUNCTION shadow(x) {
    //          LOCAL x
    //          x = 2
    //          shadow = 1
    //      }

    const char* before_defn =
        "{ a = shadow(b)\n }";

    const char* after_defn =
        "{ LOCAL r_0_\n"
        "  r_0_ = 2\n"
        "  a = 1\n }";

    auto expr1 = parse_block(before_defn, mock_symbols());
    ASSERT_TRUE(expr1);
    auto before = expr1->is_block();
    ASSERT_TRUE(before);

    auto expr2 = parse_block(after_defn, mock_symbols());
    ASSERT_TRUE(expr2);
    auto expected = expr2->is_block();
    ASSERT_TRUE(expected);

    EXPECT_EXPR_EQ(expected, inline_function_calls(before));
}
