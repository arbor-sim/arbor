#include "test.hpp"

#include "cprinter.hpp"

using scope_type = Scope<Symbol>;
using symbol_map = scope_type::symbol_map;
using symbol_ptr = Scope<Symbol>::symbol_ptr;

TEST(CPrinter, statement) {
    std::vector<const char*> expressions =
    {
"y=x+3",
"y=y^z",
"y=exp(x/2 + 3)",
    };

    // create a scope that contains the symbols used in the tests
    Scope<Symbol>::symbol_map globals;
    auto scope = std::make_shared<Scope<Symbol>>(globals);

    scope->add_local_symbol("x", make_symbol<LocalVariable>(Location(), "x", localVariableKind::local));
    scope->add_local_symbol("y", make_symbol<LocalVariable>(Location(), "y", localVariableKind::local));
    scope->add_local_symbol("z", make_symbol<LocalVariable>(Location(), "z", localVariableKind::local));

    for(auto const& expression : expressions) {
        auto e = parse_line_expression(expression);

        // sanity check the compiler
        EXPECT_NE(e, nullptr);
        if( e==nullptr ) continue;

        e->semantic(scope);
        auto v = make_unique<CPrinter>();
        e->accept(v.get());

        verbose_print(e->to_string());
        verbose_print(" :--: ", v->text());
    }
}

TEST(CPrinter, proc) {
    std::vector<const char*> expressions =
    {
"PROCEDURE trates(v) {\n"
"    LOCAL k\n"
"    minf=1-1/(1+exp((v-k)/k))\n"
"    hinf=1/(1+exp((v-k)/k))\n"
"    mtau = 0.6\n"
"    htau = 1500\n"
"}"
    };

    // create a scope that contains the symbols used in the tests
    Scope<Symbol>::symbol_map globals;
    globals["minf"] = make_symbol<VariableExpression>(Location(), "minf");
    globals["hinf"] = make_symbol<VariableExpression>(Location(), "hinf");
    globals["mtau"] = make_symbol<VariableExpression>(Location(), "mtau");
    globals["htau"] = make_symbol<VariableExpression>(Location(), "htau");
    globals["v"]    = make_symbol<VariableExpression>(Location(), "v");

    for(auto const& expression : expressions) {
        expression_ptr e = parse_procedure(expression);
        ASSERT_TRUE(e->is_symbol());

        auto procname = e->is_symbol()->name();
        auto& proc = (globals[procname] = symbol_ptr(e.release()->is_symbol()));

        proc->semantic(globals);
        auto v = make_unique<CPrinter>();
        proc->accept(v.get());

        verbose_print(proc->to_string());
        verbose_print(" :--: ", v->text());
    }
}
