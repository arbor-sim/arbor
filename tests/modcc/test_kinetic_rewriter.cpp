#include "test.hpp"

#include "expression.hpp"
#include "kinrewriter.hpp"
#include "parser.hpp"

inline symbol_ptr state_var(const char* name) {
    auto v = make_symbol<VariableExpression>(Location(), name);
    v->is_variable()->state(true);
    return v;
}

inline symbol_ptr assigned_var(const char* name) {
    return make_symbol<VariableExpression>(Location(), name);
}

static const char* kinetic_simple =
    "KINETIC kin {             \n"
    "    u = 3                 \n"
    "    ~ a <-> b (u, v)      \n"
    "    ~ b <-> 3b + c (u, v) \n"
    "}                         \n";

TEST(KineticRewriter, match) {
    auto visitor = make_unique<KineticRewriter>();
    auto kin = Parser(kinetic_simple).parse_procedure();

    ASSERT_NE(nullptr, kin);
    ASSERT_TRUE(kin->is_symbol() && kin->is_symbol()->is_procedure());

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

    std::cout << kin_weak->to_string() << "\n";
    std::cout << visitor->as_procedure()->to_string() << "\n";
}
