#include <iostream>
#include <string>

#include "expression.hpp"
#include "kinrewriter.hpp"
#include "parser.hpp"

#include "alg_collect.hpp"
#include "expr_expand.hpp"
#include "test.hpp"


stmt_list_type& proc_statements(Expression *e) {
    if (!e || !e->is_symbol() || ! e->is_symbol()->is_procedure()) {
        throw std::runtime_error("not a procedure");
    }

    return e->is_symbol()->is_procedure()->body()->statements();
}


inline symbol_ptr state_var(const char* name) {
    auto v = make_symbol<VariableExpression>(Location(), name);
    v->is_variable()->state(true);
    return v;
}

inline symbol_ptr assigned_var(const char* name) {
    return make_symbol<VariableExpression>(Location(), name);
}

static const char* kinetic_abc =
    "KINETIC kin {             \n"
    "    u = 3                 \n"
    "    ~ a <-> b (u, v)      \n"
    "    u = 4                 \n"
    "    v = sin(u)            \n"
    "    ~ b <-> 3b + c (u, v) \n"
    "}                         \n";

static const char* derivative_abc =
    "DERIVATIVE deriv {        \n"
    "    a' = -3*a + b*v       \n"
    "    LOCAL rev2            \n"
    "    rev2 = c*b^3*sin(4)   \n"
    "    b' = 3*a - v*b + 8*b - 2*rev2\n"
    "    c' = 4*b - rev2       \n"
    "}                         \n";

TEST(KineticRewriter, equiv) {
    auto visitor = make_unique<KineticRewriter>();
    auto kin = Parser(kinetic_abc).parse_procedure();
    auto deriv = Parser(derivative_abc).parse_procedure();

    ASSERT_NE(nullptr, kin);
    ASSERT_NE(nullptr, deriv);
    ASSERT_TRUE(kin->is_symbol() && kin->is_symbol()->is_procedure());
    ASSERT_TRUE(deriv->is_symbol() && deriv->is_symbol()->is_procedure());

    auto kin_weak = kin->is_symbol()->is_procedure();
    scope_type::symbol_map globals;
    globals["kin"] = std::move(kin);
    globals["a"] = state_var("a");
    globals["b"] = state_var("b");
    globals["c"] = state_var("c");
    globals["u"] = assigned_var("u");
    globals["v"] = assigned_var("v");

    kin_weak->semantic(globals);
    kin_weak->accept(visitor.get());

    auto kin_deriv = visitor->as_procedure();

    if (g_verbose_flag) {
        std::cout << "derivative procedure:\n" << deriv->to_string() << "\n";
        std::cout << "kin procedure:\n" << kin_weak->to_string() << "\n";
        std::cout << "rewritten kin procedure:\n" << kin_deriv->to_string() << "\n";
    }

    auto deriv_map = expand_assignments(proc_statements(deriv.get()));
    auto kin_map = expand_assignments(proc_statements(kin_deriv.get()));

    if (g_verbose_flag) {
        std::cout << "derivative assignments (canonical):\n";
        for (const auto&p: deriv_map) {
            std::cout << p.first << ": " << p.second << "\n";
        }
        std::cout << "rewritten kin assignments (canonical):\n";
        for (const auto&p: kin_map) {
            std::cout << p.first << ": " << p.second << "\n";
        }
    }

    EXPECT_EQ(deriv_map["a'"], kin_map["a'"]);
    EXPECT_EQ(deriv_map["b'"], kin_map["b'"]);
    EXPECT_EQ(deriv_map["c'"], kin_map["c'"]);
}

