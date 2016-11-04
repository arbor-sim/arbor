#include <cmath>
#include <memory>

#include "test.hpp"
#include "module.hpp"
#include "modccutil.hpp"
#include "parser.hpp"

template <typename EPtr>
void verbose_print(const EPtr& e, Parser& p, const char* text) {
    if (!g_verbose_flag) return;

    if (e) std::cout << e->to_string() << "\n";
    if (p.status()==lexerStatus::error)
        std::cout << "in " << red(text) << "\t" << p.error_message() << "\n";
}

template <typename Derived, typename RetUniqPtr>
::testing::AssertionResult check_parse(
    std::unique_ptr<Derived>& derived,
    RetUniqPtr (Parser::*pmemfn)(),
    const char* text)
{
    Parser p(text);
    auto e = (p.*pmemfn)();
    verbose_print(e, p, text);

    if (e==nullptr) {
        return ::testing::AssertionFailure() << "failed to parse '" << text << "'";
    }

    if (p.status()!=lexerStatus::happy) {
        return ::testing::AssertionFailure() << "parser status is not happy";
    }

    Derived *ptr = e? dynamic_cast<Derived*>(e.get()): nullptr;
    if (ptr==nullptr) {
        return ::testing::AssertionFailure() << "failed to cast to derived type";
    }
    else {
        e.release();
        derived.reset(ptr);
    }

    return ::testing::AssertionSuccess();
}

template <typename RetUniqPtr>
::testing::AssertionResult check_parse(RetUniqPtr (Parser::*pmemfn)(), const char* text) {
    std::unique_ptr<Expression> e;
    return check_parse(e, pmemfn, text);
}

template <typename RetUniqPtr>
::testing::AssertionResult check_parse_fail(RetUniqPtr (Parser::*pmemfn)(), const char* text) {
    Parser p(text);
    auto e = (p.*pmemfn)();
    verbose_print(e, p, text);

    if (p.status()!=lexerStatus::error) {
        return ::testing::AssertionFailure() << "parser status is not error";
    }

    if (e!=nullptr) {
        return ::testing::AssertionFailure() << "parser returned non-null expression";
    }

    return ::testing::AssertionSuccess();
}

TEST(Parser, full_file) {
    Module m(DATADIR "/test.mod");
    if (m.buffer().size()==0) {
        std::cout << "skipping Parser.full_file test because unable to open input file" << std::endl;
        return;
    }
    Parser p(m);
    EXPECT_EQ(p.status(), lexerStatus::happy);
}

TEST(Parser, procedure) {
    std::vector<const char*> calls = {
        "PROCEDURE foo(x, y) {\n"
        "  LOCAL a\n"
        "  LOCAL b\n"
        "  LOCAL c\n"
        "  a = 3\n"
        "  b = x * y + 2\n"
        "  y = x + y * 2\n"
        "  y = a + b +c + a + b\n"
        "  y = a + b *c + a + b\n"
        "}"
        ,
        "PROCEDURE trates(v) {\n"
        "    LOCAL qt\n"
        "    qt=q10^((celsius-22)/10)\n"
        "    minf=1-1/(1+exp((v-vhalfm)/km))\n"
        "    hinf=1/(1+exp((v-vhalfh)/kh))\n"
        "    mtau = 0.6\n"
        "    htau = 1500\n"
        "}"
    };

    for (const auto& str: calls) {
        EXPECT_TRUE(check_parse(&Parser::parse_procedure, str));
    }
}

TEST(Parser, net_receive) {
    char str[] =
        "NET_RECEIVE (x, y) {   \n"
        "  LOCAL a              \n"
        "  a = 3                \n"
        "  x = a+3              \n"
        "  y = x+a              \n"
        "}";

    std::unique_ptr<Symbol> sym;

    EXPECT_TRUE(check_parse(sym, &Parser::parse_procedure, str));
    if (sym) {
        auto nr = sym->is_net_receive();
        EXPECT_NE(nullptr, nr);
        if (nr) {
            EXPECT_EQ(2u, nr->args().size());
        }
    }
}

TEST(Parser, function) {
    char str[] =
        "FUNCTION foo(x, y) {"
        "  LOCAL a\n"
        "  a = 3\n"
        "  b = x * y + 2\n"
        "  y = x + y * 2\n"
        "  foo = a * x + y\n"
        "}";

    std::unique_ptr<Symbol> sym;
    EXPECT_TRUE(check_parse(sym, &Parser::parse_function, str));
}

TEST(Parser, parse_solve) {
    std::unique_ptr<SolveExpression> s;

    EXPECT_TRUE(check_parse(s, &Parser::parse_solve, "SOLVE states METHOD cnexp"));
    if (s) {
        EXPECT_EQ(s->method(), solverMethod::cnexp);
        EXPECT_EQ(s->name(), "states");
    }

    EXPECT_TRUE(check_parse(s, &Parser::parse_solve, "SOLVE states"));
    if (s) {
        EXPECT_EQ(s->method(), solverMethod::none);
        EXPECT_EQ(s->name(), "states");
    }
}

TEST(Parser, parse_conductance) {
    std::unique_ptr<ConductanceExpression> s;

    EXPECT_TRUE(check_parse(s, &Parser::parse_conductance, "CONDUCTANCE g USEION na"));
    if (s) {
        EXPECT_EQ(s->ion_channel(), ionKind::Na);
        EXPECT_EQ(s->name(), "g");
    }

    EXPECT_TRUE(check_parse(s, &Parser::parse_conductance, "CONDUCTANCE gnda"));
    if (s) {
        EXPECT_EQ(s->ion_channel(), ionKind::nonspecific);
        EXPECT_EQ(s->name(), "gnda");
    }
}

TEST(Parser, parse_if) {
    std::unique_ptr<IfExpression> s;

    EXPECT_TRUE(check_parse(s, &Parser::parse_if,
        "   if(a<b) {      \n"
        "       a = 2+b    \n"
        "       b = 4^b    \n"
        "   }              \n"
    ));
    if (s) {
        EXPECT_NE(s->condition()->is_binary(), nullptr);
        EXPECT_NE(s->true_branch()->is_block(), nullptr);
        EXPECT_EQ(s->false_branch(), nullptr);
    }

    EXPECT_TRUE(check_parse(s, &Parser::parse_if,
        "   if(a<b) {      \n"
        "       a = 2+b    \n"
        "   } else {       \n"
        "       a = 2+b    \n"
        "   }                "
    ));
    if (s) {
        EXPECT_NE(s->condition()->is_binary(), nullptr);
        EXPECT_NE(s->true_branch()->is_block(), nullptr);
        EXPECT_NE(s->false_branch(), nullptr);
    }

    EXPECT_TRUE(check_parse(s, &Parser::parse_if,
        "   if(a<b) {      \n"
        "       a = 2+b    \n"
        "   } else if(b>a){\n"
        "       a = 2+b    \n"
        "   }              "
    ));
    if (s) {
        EXPECT_NE(s->condition()->is_binary(), nullptr);
        EXPECT_NE(s->true_branch()->is_block(), nullptr);
        ASSERT_NE(s->false_branch(), nullptr);
        ASSERT_NE(s->false_branch()->is_if(), nullptr);
        EXPECT_EQ(s->false_branch()->is_if()->false_branch(), nullptr);
    }
}

TEST(Parser, parse_local) {
    std::unique_ptr<LocalDeclaration> s;
    EXPECT_TRUE(check_parse(s, &Parser::parse_local, "LOCAL xyz"));
    if (s) {
        ASSERT_EQ(1u, s->variables().size());
    }

    EXPECT_TRUE(check_parse(s, &Parser::parse_local, "LOCAL x, y, z"));
    if (s) {
        auto vars = s->variables();
        ASSERT_EQ(3u, vars.size());
        ASSERT_TRUE(vars.count("x"));
        ASSERT_TRUE(vars.count("y"));
        ASSERT_TRUE(vars.count("z"));
    }

    EXPECT_TRUE(check_parse_fail(&Parser::parse_local, "LOCAL x,"));
}

TEST(Parser, parse_unary_expression) {
    const char* good_expr[] = {
        "+x             ",
        "-x             ",
        "(x + -y)       ",
        "-(x - + -y)    ",
        "exp(x + y)     ",
        "-exp(x + -y)   "
    };

    for (auto& text: good_expr) {
        EXPECT_TRUE(check_parse(&Parser::parse_unaryop, text));
    }
}

// test parsing of parenthesis expressions
TEST(Parser, parse_parenthesis_expression) {
    const char* good_expr[] = {
        "((celsius-22)/10)      ",
        "((celsius-22)+10)      ",
        "(x+2)                  ",
        "((x))                  ",
        "(((x)))                ",
        "(x + (x * (y*(2)) + 4))",
    };

    for (auto& text: good_expr) {
        EXPECT_TRUE(check_parse(&Parser::parse_parenthesis_expression, text));
    }

    const char* bad_expr[] = {
        "(x             ",
        "((x+3)         ",
        "(x+ +)         ",
        "(x=3)          ",  // assignment inside parenthesis isn't allowed
        "(a + (b*2^(x)) ",  // missing closing parenthesis
    };

    for (auto& text: bad_expr) {
        EXPECT_TRUE(check_parse_fail(&Parser::parse_parenthesis_expression, text));
    }
}

// test parsing of line expressions
TEST(Parser, parse_line_expression) {
    const char* good_expr[] = {
        "qt=q10^((celsius-22)/10)"
        "x=2        ",
        "x=2        ",
        "x = -y\n   "
        "x=2*y      ",
        "x=y + 2 * z",
        "x=(y + 2) * z      ",
        "x=(y + 2) * z ^ 3  ",
        "x=(y + 2 * z ^ 3)  ",
        "foo(x+3, y, bar(21.4))",
        "y=exp(x+3) + log(exp(x/y))",
        "a=x^y^z",
        "a=x/y/z"
    };

    for (auto& text: good_expr) {
        EXPECT_TRUE(check_parse(&Parser::parse_line_expression, text));
    }

    const char* bad_expr[] = {
        "x=2+        ",      // incomplete binary expression on rhs
        "x=          ",      // missing rhs of assignment
        "x=)y + 2 * z",
        "x=(y + 2    ",
        "x=(y ++ z   ",
        "x/=3        ",      // compound binary expressions not supported
        "foo+8       ",      // missing assignment
        "foo()=8     ",      // lhs of assingment must be an lvalue
    };

    for (auto& text: bad_expr) {
        EXPECT_TRUE(check_parse_fail(&Parser::parse_line_expression, text));
    }
}

TEST(Parser, parse_stoich_term) {
    const char* good_expr[] = {
        "B", "B3", "3B3", "0A", "12A"
    };

    for (auto& text: good_expr) {
        std::unique_ptr<StoichTermExpression> s;
        EXPECT_TRUE(check_parse(s, &Parser::parse_stoich_term, text));
    }

    const char* bad_expr[] = {
        "-A", "-3A", "0.2A", "5"
    };

    for (auto& text: bad_expr) {
        EXPECT_TRUE(check_parse_fail(&Parser::parse_stoich_term, text));
    }
}

TEST(Parser, parse_stoich_expression) {
    const char* single_expr[] = {
        "B", "B3", "3xy"
    };

    for (auto& text: single_expr) {
        std::unique_ptr<StoichExpression> s;
        EXPECT_TRUE(check_parse(s, &Parser::parse_stoich_expression, text));
        EXPECT_EQ(1, s->terms().size());
    }

    const char* double_expr[] = {
        "B+A", "a1 + 2bn", "4c+d"
    };

    for (auto& text: double_expr) {
        std::unique_ptr<StoichExpression> s;
        EXPECT_TRUE(check_parse(s, &Parser::parse_stoich_expression, text));
        EXPECT_EQ(2, s->terms().size());
    }

    const char* other_good_expr[] = {
        "", "a+b+c", "1a+2b+3c+4d"
    };

    for (auto& text: other_good_expr) {
        std::unique_ptr<StoichExpression> s;
        EXPECT_TRUE(check_parse(s, &Parser::parse_stoich_expression, text));
    }

    const char* bad_expr[] = {
        "A+B+", "A+5+B"
    };

    for (auto& text: bad_expr) {
        EXPECT_TRUE(check_parse_fail(&Parser::parse_stoich_expression, text));
    }
}

// test parsing of stoich and reaction expressions
TEST(Parser, parse_reaction_expression) {
    const char* good_expr[] = {
        "~ A + B <-> C + D (k1, k2)",
        "~ 2B <-> C + D + E (k1(3,v), k2)",
        "~ <-> C + D + 7 E (k1, f(a,b)-2)",
        "~ <-> (f,g)",
        "~ A + 3B + C<-> (f,g)"
    };

    for (auto& text: good_expr) {
        std::unique_ptr<ReactionExpression> s;
        EXPECT_TRUE(check_parse(s, &Parser::parse_reaction_expression, text));
    }

    const char* bad_expr[] = {
        "~ A + B <-> C + D (k1, k2, k3)",
        "~ A + B <-> C + (k1, k2)",
        "~ 2.3B <-> C + D + E (k1(3,v), k2)",
        "~ <-> C + D + 7E",
        "~ <-> (,g)",
        "~ A - 3B + C<-> (f,g)",
        "  A <-> B (k1, k2)",
        "~ A <- B (k1)",
        "~ A -> B (k2)",
    };

    for (auto& text: bad_expr) {
        EXPECT_TRUE(check_parse_fail(&Parser::parse_reaction_expression, text));
    }
}

long double eval(Expression *e) {
    if (auto n = e->is_number()) {
        return n->value();
    }
    if (auto b = e->is_binary()) {
        auto lhs = eval(b->lhs());
        auto rhs = eval(b->rhs());
        switch(b->op()) {
            case tok::plus  : return lhs+rhs;
            case tok::minus : return lhs-rhs;
            case tok::times : return lhs*rhs;
            case tok::divide: return lhs/rhs;
            case tok::pow   : return std::pow(lhs,rhs);
            default:;
        }
    }
    if (auto u = e->is_unary()) {
        auto val = eval(u->expression());
        switch(u->op()) {
            case tok::plus  : return  val;
            case tok::minus : return -val;
            default:;
        }
    }
    return std::numeric_limits<long double>::quiet_NaN();
}

// test parsing of expressions for correctness
// by parsing rvalue expressions with numeric atoms, which can be evalutated using eval
TEST(Parser, parse_binop) {
    std::pair<const char*, double> tests[] = {
        // simple
        {"2+3", 2.+3.},
        {"2-3", 2.-3.},
        {"2*3", 2.*3.},
        {"2/3", 2./3.},
        {"2^3", std::pow(2., 3.)},

        // more complicated
        {"2+3*2", 2.+(3*2)},
        {"2*3-5", (2.*3)-5.},
        {"2+3*(-2)", 2.+(3*-2)},
        {"2+3*(-+2)", 2.+(3*-+2)},
        {"2/3*4", (2./3.)*4.},

        // right associative
        {"2^3^1.5", std::pow(2.,std::pow(3.,1.5))},
        {"2^3^1.5^2", std::pow(2.,std::pow(3.,std::pow(1.5,2.)))},
        {"2^2^3", std::pow(2.,std::pow(2.,3.))},
        {"(2^2)^3", std::pow(std::pow(2.,2.),3.)},
        {"3./2^7.", 3./std::pow(2.,7.)},
        {"3^2*5.", std::pow(3.,2.)*5.},
    };

    for (const auto& test_case: tests) {
        std::unique_ptr<Expression> e;
        EXPECT_TRUE(check_parse(e, &Parser::parse_expression, test_case.first));

        // A loose tolerance of 1d-10 is required here because the eval()
        // function uses long double for intermediate results (like constant
        // folding in modparser).  For expressions with transcendental
        // operations this can see relatively large divergence between the
        // double and long double results.
        EXPECT_NEAR(eval(e.get()), test_case.second, 1e-10);
    }
}

TEST(Parser, parse_state_block) {
    const char* state_blocks[] = {
        "STATE {\n"
        "    h\n"
        "    m r\n"
        "}",
        "STATE {\n"
        "    h (nA)\n"
        "    m r\n"
        "}",
        "STATE {\n"
        "    h (nA)\n"
        "    m (nA) r\n"
        "}",
        "STATE {\n"
        "    h (nA)\n"
        "    m r (uA)\n"
        "}",
        "STATE {\n"
        "    h (nA)\n"
        "    m (nA) r (uA)\n"
        "}"
    };

    expression_ptr null;
    for (auto& text: state_blocks) {
        Module m(text, sizeof(text));
        Parser p(m, false);
        p.parse_state_block();
        EXPECT_EQ(lexerStatus::happy, p.status());
        verbose_print(null, p, text);
    }
}
