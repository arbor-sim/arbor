#include "test.hpp"

#include "../src/cprinter.hpp"

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
    globals["x"] = make_symbol<Symbol>(Location(), "x", symbolKind::local);
    globals["y"] = make_symbol<Symbol>(Location(), "y", symbolKind::local);
    globals["z"] = make_symbol<Symbol>(Location(), "z", symbolKind::local);

    auto scope = std::make_shared<Scope<Symbol>>(globals);

    for(auto const& expression : expressions) {
        auto e = parse_line_expression(expression);

        // sanity check the compiler
        EXPECT_NE(e, nullptr);
        if( e==nullptr ) continue;

        e->semantic(scope);
        auto v = make_unique<CPrinter>();
        e->accept(v.get());

#ifdef VERBOSE_TEST
        std::cout << e->to_string() << std::endl;
                  << " :--: " << v->text() << std::endl;
#endif
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
        auto e = symbol_ptr{parse_procedure(expression)->is_symbol()};

        // sanity check the compiler
        EXPECT_NE(e, nullptr);

        if( e==nullptr ) continue;

        globals["trates"] = std::move(e);

        e->semantic(globals);
        auto v = make_unique<CPrinter>();
        e->accept(v.get());

#ifdef VERBOSE_TEST
        std::cout << e->to_string() << std::endl;
                  << " :--: " << v->text() << std::endl;
#endif
    }
}

