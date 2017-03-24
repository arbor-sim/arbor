#pragma once

#include <string>

#include "../gtest.h"

#include "expression.hpp"
#include "parser.hpp"
#include "modccutil.hpp"

extern bool g_verbose_flag;

#define VERBOSE_PRINT(x) (g_verbose_flag && std::cout << (x) << "\n")

inline expression_ptr parse_line_expression(std::string const& s) {
    return Parser(s).parse_line_expression();
}

inline expression_ptr parse_expression(std::string const& s) {
    return Parser(s).parse_expression();
}

inline expression_ptr parse_function(std::string const& s) {
    return Parser(s).parse_function();
}

inline expression_ptr parse_procedure(std::string const& s) {
    return Parser(s).parse_procedure();
}

// Helpers for comparing expressions, and verbose expression printing.

// Strip ANSI control sequences from `to_string` output.
std::string plain_text(Expression* expr);

// Compare two expressions via their representation.
// Use with EXPECT_PRED_FORMAT2.
::testing::AssertionResult assert_expr_eq(const char *arg1, const char *arg2, Expression* expected, Expression* value);

#define EXPECT_EXPR_EQ(a,b) EXPECT_PRED_FORMAT2(assert_expr_eq, a, b)

// Print arguments, but only if verbose flag set.
// Use `to_string()` to print (smart) pointers to Expression or Scope objects.

namespace impl {
    template <typename X>
    struct has_to_string {
        template <typename T>
        static decltype(std::declval<T>()->to_string(), std::true_type{}) test(int);
        template <typename T>
        static std::false_type test(...);

        using type = decltype(test<X>(0));
    };

    template <typename X>
    void print(const X& x, std::true_type) {
        if (x) {
            std::cout << x->to_string();
        }
        else {
            std::cout << "null";
        }
    }

    template <typename X>
    void print(const X& x, std::false_type) {
        std::cout << x;
    }

    template <typename X>
    void print(const X& x) {
        print(x, typename has_to_string<X>::type{});
    }

}

inline void verbose_print() {
    if (!g_verbose_flag) return;
    std::cout << "\n";
}

template <typename X, typename... Args>
void verbose_print(const X& arg, const Args&... tail) {
    if (!g_verbose_flag) return;
    impl::print(arg);
    verbose_print(tail...);
}

