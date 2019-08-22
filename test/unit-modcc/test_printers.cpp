#include <memory>
#include <regex>
#include <string>
#include <sstream>

#include "common.hpp"

#include "printer/cexpr_emit.hpp"
#include "printer/cprinter.hpp"
#include "printer/cudaprinter.hpp"
#include "expression.hpp"
#include "symdiff.hpp"

// Note: CUDA printer disabled until new implementation finished.
//#include "printer/cudaprinter.hpp"

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

TEST(scalar_printer, constants) {
    testcase testcases[] = {
        {"1./0.",      "INFINITY"},
        {"-1./0.",     "-INFINITY"},
        {"(-1)^0.5",   "NAN"},
        {"1/(-1./0.)", "-0."},
        {"1-1",        "0."},
    };

    for (const auto& tc: testcases) {
        auto expr = constant_simplify(parse_expression(tc.source));
        ASSERT_TRUE(expr && expr->is_number());

        std::stringstream s;
        s << as_c_double(expr->is_number()->value());

        EXPECT_EQ(std::string(tc.expected), s.str());
    }
}

TEST(scalar_printer, statement) {
    std::vector<testcase> testcases = {
        {"y=x+3",            "y=x+3"},
        {"y=y^z",            "y=pow(y,z)"},
        {"y=exp((x/2) + 3)", "y=exp(x/2+3)"},
        {"z=a/b/c",          "z=a/b/c"},
        {"z=a/(b/c)",        "z=a/(b/c)"},
        {"z=(a*b)/c",        "z=a*b/c"},
        {"z=a-(b+c)",        "z=a-(b+c)"},
        {"z=(a>0)<(b>0)",    "z=a>0.<(b>0.)"},
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
            std::stringstream out;
            auto printer = std::make_unique<CPrinter>(out);
            e->accept(printer.get());
            std::string text = out.str();

            verbose_print(e->to_string(), " :--: ", text);
            EXPECT_EQ(strip(tc.expected), strip(text));
        }

        {
            SCOPED_TRACE("CudaPrinter");
            std::stringstream out;
            auto printer = std::make_unique<CudaPrinter>(out);
            e->accept(printer.get());
            std::string text = out.str();

            verbose_print(e->to_string(), " :--: ", text);
            EXPECT_EQ(strip(tc.expected), strip(text));
        }
    }
}

TEST(CPrinter, proc_body) {
    std::vector<testcase> testcases = {
        {
            "PROCEDURE trates(v) {\n"
            "    LOCAL k\n"
            "    minf = 1-1/(1+exp((v-k)/k))\n"
            "    hinf = 1/(1+exp((v-k)/k))\n"
            "    mtau = 0.5\n"
            "    htau = 1500\n"
            "}"
            ,
            "value_type k;\n"
            "minf[i_] = 1-1/(1+exp((v-k)/k));\n"
            "hinf[i_] = 1/(1+exp((v-k)/k));\n"
            "mtau[i_] = 0.5;\n"
            "htau[i_] = 1500;\n"
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
        std::stringstream out;
        auto v = std::make_unique<CPrinter>(out);
        proc->is_procedure()->body()->accept(v.get());
        std::string text = out.str();

        verbose_print(proc->is_procedure()->body()->to_string());
        verbose_print(" :--: ", text);

        EXPECT_EQ(strip(tc.expected), strip(text));
    }
}

TEST(CPrinter, proc_body_const) {
    std::vector<testcase> testcases = {
            {
                    "PROCEDURE trates(v) {\n"
                    "    mtau = 0.5 - t0 + t1\n"
                    "}"
                    ,
                    "mtau[i_] = 0.5 - -0.5 + 1.2;\n"
            }
    };

    // create a scope that contains the symbols used in the tests
    Scope<Symbol>::symbol_map globals;
    globals["mtau"] = make_symbol<VariableExpression>(Location(), "mtau");

    for (const auto& tc: testcases) {
        Parser p(tc.source);
        p.constants_map_.insert({"t0","-0.5"});
        p.constants_map_.insert({"t1","1.2"});
        expression_ptr e = p.parse_procedure();
        ASSERT_TRUE(e->is_symbol());

        auto procname = e->is_symbol()->name();
        auto& proc = (globals[procname] = symbol_ptr(e.release()->is_symbol()));

        proc->semantic(globals);
        std::stringstream out;
        auto v = std::make_unique<CPrinter>(out);
        proc->is_procedure()->body()->accept(v.get());
        std::string text = out.str();

        verbose_print(proc->is_procedure()->body()->to_string());
        verbose_print(" :--: ", text);

        EXPECT_EQ(strip(tc.expected), strip(text));
    }
}