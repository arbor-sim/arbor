#include <iostream>
#include <string>

#include "expression.hpp"
#include "kineticrewriter.hpp"
#include "parser.hpp"

#include "alg_collect.hpp"
#include "common.hpp"
#include "expr_expand.hpp"

expr_list_type& statements(Expression *e) {
    if (e) {
        if (auto block = e->is_block()) {
            return block->statements();
        }

        if (auto sym = e->is_symbol()) {
            if (auto proc = sym->is_procedure()) {
                return proc->body()->statements();
            }

            if (auto proc = sym->is_procedure()) {
                return proc->body()->statements();
            }
        }
    }

    throw std::runtime_error("not a block, function or procedure");
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
    auto kin = Parser(kinetic_abc).parse_procedure();
    auto deriv = Parser(derivative_abc).parse_procedure();

    auto kin_ptr = kin.get();
    auto deriv_ptr = deriv.get();

    ASSERT_NE(nullptr, kin);
    ASSERT_NE(nullptr, deriv);
    ASSERT_TRUE(kin->is_symbol() && kin->is_symbol()->is_procedure());
    ASSERT_TRUE(deriv->is_symbol() && deriv->is_symbol()->is_procedure());

    scope_type::symbol_map globals;
    globals["kin"] = std::move(kin);
    globals["deriv"] = std::move(deriv);
    globals["a"] = state_var("a");
    globals["b"] = state_var("b");
    globals["c"] = state_var("c");
    globals["u"] = assigned_var("u");
    globals["v"] = assigned_var("v");

    deriv_ptr->semantic(globals);

    auto kin_body = kin_ptr->is_procedure()->body();
    scope_ptr scope = std::make_shared<scope_type>(globals);
    kin_body->semantic(scope);

    auto kin_deriv = kinetic_rewrite(kin_body);

    verbose_print("derivative procedure:\n", deriv_ptr);
    verbose_print("kin procedure:\n", kin_ptr);
    verbose_print("rewritten kin body:\n", kin_deriv);

    auto deriv_map = expand_assignments(statements(deriv_ptr));
    auto kin_map = expand_assignments(statements(kin_deriv.get()));

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
