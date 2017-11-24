#include <regex>
#include <string>

#include "test.hpp"

#include "cprinter.hpp"
#include "expression.hpp"

//using scope_type = Scope<Symbol>;
//using symbol_map = scope_type::symbol_map;
//using symbol_ptr = Scope<Symbol>::symbol_ptr;

struct testcase {
    const char* source;
    const char* expected;
};

static std::string strip(std::string text) {
    static std::regex rx(R"(\s)");
    return std::regex_replace(text, rx, "");
}

TEST(CPrinter, statement) {
    std::vector<testcase> testcases = {
        {"y=x+3",            "y=x+3"},
        {"y=y^z",            "y=std::pow(y,z)"},
        {"y=exp((x/2) + 3)", "y=exp(x/2+3)"}
    };

    // create a scope that contains the symbols used in the tests
    Scope<Symbol>::symbol_map globals;
    auto scope = std::make_shared<Scope<Symbol>>(globals);

    scope->add_local_symbol("x", make_symbol<LocalVariable>(Location(), "x", localVariableKind::local));
    scope->add_local_symbol("y", make_symbol<LocalVariable>(Location(), "y", localVariableKind::local));
    scope->add_local_symbol("z", make_symbol<LocalVariable>(Location(), "z", localVariableKind::local));

    for (const auto& tc: testcases) {
        auto e = parse_line_expression(tc.source);
        ASSERT_TRUE(e);

        e->semantic(scope);
        auto v = make_unique<CPrinter>();
        e->accept(v.get());

        verbose_print(e->to_string());
        verbose_print(" :--: ", v->text());

        EXPECT_EQ(strip(tc.expected), strip(v->text()));
    }
}

TEST(CPrinter, proc) {
    std::vector<testcase> testcases = {
        {
            "PROCEDURE trates(v) {\n"
            "    LOCAL k\n"
            "    minf=1-1/(1+exp((v-k)/k))\n"
            "    hinf=1/(1+exp((v-k)/k))\n"
            "    mtau = 0.6\n"
            "    htau = 1500\n"
            "}"
            ,
            "void trates(int i_, value_type v) {\n"
            "value_type k;\n"
            "minf[i_] = 1-1/(1+exp((v-k)/k));\n"
            "hinf[i_] = 1/(1+exp((v-k)/k));\n"
            "mtau[i_] = 0.6;\n"
            "htau[i_] = 1500;\n"
            "}"
        }
    };

    // create a scope that contains the symbols used in the tests
    Scope<Symbol>::symbol_map globals;
    globals["minf"] = make_symbol<VariableExpression>(Location(), "minf");
    globals["hinf"] = make_symbol<VariableExpression>(Location(), "hinf");
    globals["mtau"] = make_symbol<VariableExpression>(Location(), "mtau");
    globals["htau"] = make_symbol<VariableExpression>(Location(), "htau");
    globals["v"]    = make_symbol<VariableExpression>(Location(), "v");

    for (const auto& tc: testcases) {
        expression_ptr e = parse_procedure(tc.source);
        ASSERT_TRUE(e->is_symbol());

        auto procname = e->is_symbol()->name();
        auto& proc = (globals[procname] = symbol_ptr(e.release()->is_symbol()));

        proc->semantic(globals);
        auto v = make_unique<CPrinter>();
        proc->accept(v.get());

        verbose_print(proc->to_string());
        verbose_print(" :--: ", v->text());

        EXPECT_EQ(strip(tc.expected), strip(v->text()));
    }
}
