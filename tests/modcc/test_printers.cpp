#include <regex>
#include <string>

#include "test.hpp"

#include "cprinter.hpp"
#include "cudaprinter.hpp"
#include "expression.hpp"
#include "textbuffer.hpp"

struct testcase {
    const char* source;
    const char* expected;
};

static std::string strip(std::string text) {
    // Strip all spaces, except when between two minus symbols, where instead
    // they should be replaced by a single space:
    //
    // 1. Replace whitespace with two spaces.
    // 2. Replace '-  -' with '- -'.
    // 3. Replace '  ' with ''.

    static std::regex rx1("\\s+");
    static std::regex rx2("-  -");
    static std::regex rx3("  ");

    text = std::regex_replace(text, rx1, "  ");
    text = std::regex_replace(text, rx2, "- -");
    text = std::regex_replace(text, rx3, "");

    return text;
}

TEST(scalar_printer, statement) {
    std::vector<testcase> testcases = {
        {"y=x+3",            "y=x+3"},
        {"y=y^z",            "y=std::pow(y,z)"},
        {"y=exp((x/2) + 3)", "y=exp(x/2+3)"},
        {"z=a/b/c",          "z=a/b/c"},
        {"z=a/(b/c)",        "z=a/(b/c)"},
        {"z=(a*b)/c",        "z=a*b/c"},
        {"z=a-(b+c)",        "z=a-(b+c)"},
        {"z=(a>0)<(b>0)",    "z=a>0<(b>0)"},
        {"z=a- -2",          "z=a- -2"},
        {"z=abs(x-z)",       "z=fabs(x-z)"},
        {"z=min(x,y)",       "z=min(x,y)"},
        {"z=min(max(a,b),y)","z=min(max(a,b),y)"},
    };

    // create a scope that contains the symbols used in the tests
    Scope<Symbol>::symbol_map globals;
    auto scope = std::make_shared<Scope<Symbol>>(globals);

    for (auto var: {"x", "y", "z", "a", "b", "c"}) {
        scope->add_local_symbol(var, make_symbol<LocalVariable>(Location(), var, localVariableKind::local));
    }

    for (const auto& tc: testcases) {
        auto e = parse_line_expression(tc.source);
        ASSERT_TRUE(e);

        e->semantic(scope);

        {
            SCOPED_TRACE("CPrinter");
            auto printer = make_unique<CPrinter>();
            e->accept(printer.get());
            std::string text = printer->text();

            verbose_print(e->to_string(), " :--: ", text);
            EXPECT_EQ(strip(tc.expected), strip(text));
        }

        {
            SCOPED_TRACE("CUDAPrinter");
            TextBuffer buf;
            auto printer = make_unique<CUDAPrinter>();
            printer->set_buffer(buf);

            e->accept(printer.get());
            std::string text = buf.str();

            verbose_print(e->to_string(), " :--: ", text);
            EXPECT_EQ(strip(tc.expected), strip(text));
        }
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
