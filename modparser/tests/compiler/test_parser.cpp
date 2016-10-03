#include <cmath>

#include "test.hpp"

#include "../src/module.hpp"
#include "../src/parser.hpp"

TEST(Parser, full_file) {
    Module m("./modfiles/test.mod");
    if(m.buffer().size()==0) {
        std::cout << "skipping Parser.full_file test because unable to open input file" << std::endl;
        return;
    }
    Parser p(m);
    EXPECT_EQ(p.status(), lexerStatus::happy);
}

TEST(Parser, procedure) {
    std::vector< const char*> calls =
{
"PROCEDURE foo(x, y) {"
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
    for(auto const& str : calls) {
        Parser p(str);
        auto e = p.parse_procedure();
#ifdef VERBOSE_TEST
        if(e) std::cout << e->to_string() << std::endl;
#endif
        EXPECT_NE(e, nullptr);
        EXPECT_EQ(p.status(), lexerStatus::happy);
        if(p.status()==lexerStatus::error) {
            std::cout << str << std::endl;
            std::cout << red("error ") << p.error_message() << std::endl;
        }
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
    Parser p(str);
    auto e = p.parse_procedure();
    #ifdef VERBOSE_TEST
    if(e) std::cout << e->to_string() << std::endl;
    #endif

    EXPECT_NE(e, nullptr);
    EXPECT_EQ(p.status(), lexerStatus::happy);

    auto nr = e->is_symbol()->is_net_receive();
    EXPECT_NE(nr, nullptr);
    if(nr) {
        EXPECT_EQ(nr->args().size(), (unsigned)2);
    }
    if(p.status()==lexerStatus::error) {
        std::cout << str << std::endl;
        std::cout << red("error ") << p.error_message() << std::endl;
    }
}

TEST(Parser, function) {
    std::vector< const char*> calls =
{
"FUNCTION foo(x, y) {"
"  LOCAL a\n"
"  a = 3\n"
"  b = x * y + 2\n"
"  y = x + y * 2\n"
"  foo = a * x + y\n"
"}"
};
    for(auto const& str : calls) {
        Parser p(str);
        auto e = p.parse_function();
#ifdef VERBOSE_TEST
        if(e) std::cout << e->to_string() << std::endl;
#endif
        EXPECT_NE(e, nullptr);
        EXPECT_EQ(p.status(), lexerStatus::happy);
        if(p.status()==lexerStatus::error) {
            std::cout << str << std::endl;
            std::cout << red("error ") << p.error_message() << std::endl;
        }
    }
}

TEST(Parser, parse_solve) {
    {
        Parser p("SOLVE states METHOD cnexp");
        auto e = p.parse_solve();

#ifdef VERBOSE_TEST
        if(e) std::cout << e->to_string() << std::endl;
#endif
        EXPECT_NE(e, nullptr);
        EXPECT_EQ(p.status(), lexerStatus::happy);

        if(e) {
            SolveExpression* s = dynamic_cast<SolveExpression*>(e.get());
            EXPECT_EQ(s->method(), solverMethod::cnexp);
            EXPECT_EQ(s->name(), "states");
        }

        // always print the compiler errors, because they are unexpected
        if(p.status()==lexerStatus::error) {
            std::cout << red("error") << p.error_message() << std::endl;
        }
    }
    {
        Parser p("SOLVE states");
        auto e = p.parse_solve();

#ifdef VERBOSE_TEST
        if(e) std::cout << e->to_string() << std::endl;
#endif
        EXPECT_NE(e, nullptr);
        EXPECT_EQ(p.status(), lexerStatus::happy);

        if(e) {
            SolveExpression* s = dynamic_cast<SolveExpression*>(e.get());
            EXPECT_EQ(s->method(), solverMethod::none);
            EXPECT_EQ(s->name(), "states");
        }

        // always print the compiler errors, because they are unexpected
        if(p.status()==lexerStatus::error) {
            std::cout << red("error") << p.error_message() << std::endl;
        }
    }
}

TEST(Parser, parse_conductance) {
    {
        Parser p("CONDUCTANCE g USEION na");
        auto e = p.parse_conductance();

#ifdef VERBOSE_TEST
        if(e) std::cout << e->to_string() << std::endl;
#endif
        EXPECT_NE(e, nullptr);
        EXPECT_EQ(p.status(), lexerStatus::happy);

        if(e) {
            ConductanceExpression* s = dynamic_cast<ConductanceExpression*>(e.get());
            EXPECT_EQ(s->ion_channel(), ionKind::Na);
            EXPECT_EQ(s->name(), "g");
        }

        // always print the compiler errors, because they are unexpected
        if(p.status()==lexerStatus::error) {
            std::cout << red("error") << p.error_message() << std::endl;
        }
    }
    {
        Parser p("CONDUCTANCE gnda");
        auto e = p.parse_conductance();

#ifdef VERBOSE_TEST
        if(e) std::cout << e->to_string() << std::endl;
#endif
        EXPECT_NE(e, nullptr);
        EXPECT_EQ(p.status(), lexerStatus::happy);

        if(e) {
            ConductanceExpression* s = dynamic_cast<ConductanceExpression*>(e.get());
            EXPECT_EQ(s->ion_channel(), ionKind::nonspecific);
            EXPECT_EQ(s->name(), "gnda");
        }

        // always print the compiler errors, because they are unexpected
        if(p.status()==lexerStatus::error) {
            std::cout << red("error") << p.error_message() << std::endl;
        }
    }
}

TEST(Parser, parse_if) {
    {
        char expression[] =
        "   if(a<b) {      \n"
        "       a = 2+b    \n"
        "       b = 4^b    \n"
        "   }              \n";
        Parser p(expression);
        auto e = p.parse_if();
        EXPECT_NE(e, nullptr);
        if(e) {
            auto ife = e->is_if();
            EXPECT_NE(e->is_if(), nullptr);
            if(ife) {
                EXPECT_NE(ife->condition()->is_binary(), nullptr);
                EXPECT_NE(ife->true_branch()->is_block(), nullptr);
                EXPECT_EQ(ife->false_branch(), nullptr);
            }
            //std::cout << e->to_string() << std::endl;
        }
        else {
            std::cout << p.error_message() << std::endl;
        }
    }
    {
        char expression[] =
        "   if(a<b) {      \n"
        "       a = 2+b    \n"
        "   } else {       \n"
        "       a = 2+b    \n"
        "   }                ";
        Parser p(expression);
        auto e = p.parse_if();
        EXPECT_NE(e, nullptr);
        if(e) {
            auto ife = e->is_if();
            EXPECT_NE(ife, nullptr);
            if(ife) {
                EXPECT_NE(ife->condition()->is_binary(), nullptr);
                EXPECT_NE(ife->true_branch()->is_block(), nullptr);
                EXPECT_NE(ife->false_branch(), nullptr);
            }
            //std::cout << std::endl << e->to_string() << std::endl;
        }
        else {
            std::cout << p.error_message() << std::endl;
        }
    }
    {
        char expression[] =
        "   if(a<b) {      \n"
        "       a = 2+b    \n"
        "   } else if(b>a){\n"
        "       a = 2+b    \n"
        "   }              ";
        Parser p(expression);
        auto e = p.parse_if();
        EXPECT_NE(e, nullptr);
        if(e) {
            auto ife = e->is_if();
            EXPECT_NE(ife, nullptr);
            if(ife) {
                EXPECT_NE(ife->condition()->is_binary(), nullptr);
                EXPECT_NE(ife->true_branch()->is_block(), nullptr);
                EXPECT_NE(ife->false_branch(), nullptr);
                EXPECT_NE(ife->false_branch()->is_if(), nullptr);
                EXPECT_EQ(ife->false_branch()->is_if()->false_branch(), nullptr);
            }
            //std::cout << std::endl << e->to_string() << std::endl;
        }
        else {
            std::cout << p.error_message() << std::endl;
        }
    }
}

TEST(Parser, parse_local) {
    ////////////////////// test for valid expressions //////////////////////
    {
        Parser p("LOCAL xyz");
        auto e = p.parse_local();

        #ifdef VERBOSE_TEST
        if(e) std::cout << e->to_string() << std::endl;
        #endif
        EXPECT_NE(e, nullptr);
        if(e) {
            EXPECT_NE(e->is_local_declaration(), nullptr);
            EXPECT_EQ(p.status(), lexerStatus::happy);
        }

        // always print the compiler errors, because they are unexpected
        if(p.status()==lexerStatus::error)
            std::cout << red("error") << p.error_message() << std::endl;
    }

    {
        Parser p("LOCAL x, y, z");
        auto e = p.parse_local();

        #ifdef VERBOSE_TEST
        if(e) std::cout << e->to_string() << std::endl;
        #endif
        EXPECT_NE(e, nullptr);
        if(e) {
            EXPECT_NE(e->is_local_declaration(), nullptr);
            EXPECT_EQ(p.status(), lexerStatus::happy);
            auto vars = e->is_local_declaration()->variables();
            EXPECT_EQ(vars.size(), (unsigned)3);
            EXPECT_NE(vars.find("x"), vars.end());
            EXPECT_NE(vars.find("y"), vars.end());
            EXPECT_NE(vars.find("z"), vars.end());
        }

        // always print the compiler errors, because they are unexpected
        if(p.status()==lexerStatus::error)
            std::cout << red("error") << p.error_message() << std::endl;
    }

    ////////////////////// test for invalid expressions //////////////////////
    {
        Parser p("LOCAL 2");
        auto e = p.parse_local();

        EXPECT_EQ(e, nullptr);
        EXPECT_EQ(p.status(), lexerStatus::error);

        #ifdef VERBOSE_TEST
        if(e) std::cout << e->to_string() << std::endl;
        if(p.status()==lexerStatus::error)
            std::cout << "in " << cyan(bad_expression) << "\t" << p.error_message() << std::endl;
        #endif
    }

    {
        Parser p("LOCAL x, ");
        auto e = p.parse_local();

        EXPECT_EQ(e, nullptr);
        EXPECT_EQ(p.status(), lexerStatus::error);

        #ifdef VERBOSE_TEST
        if(e) std::cout << e->to_string() << std::endl;
        if(p.status()==lexerStatus::error)
            std::cout << "in " << cyan(bad_expression) << "\t" << p.error_message() << std::endl;
        #endif
    }
}

TEST(Parser, parse_unary_expression) {
    std::vector<const char*> good_expressions =
    {
"+x             ",
"-x             ",
"(x + -y)       ",
"-(x - + -y)    ",
"exp(x + y)     ",
"-exp(x + -y)   ",
    };

    for(auto const& expression : good_expressions) {
        Parser p(expression);
        auto e = p.parse_unaryop();

#ifdef VERBOSE_TEST
        if(e) std::cout << e->to_string() << std::endl;
#endif
        EXPECT_NE(e, nullptr);
        EXPECT_EQ(p.status(), lexerStatus::happy);

        // always print the compiler errors, because they are unexpected
        if(p.status()==lexerStatus::error)
            std::cout << red("error") << p.error_message() << std::endl;
    }
}

// test parsing of parenthesis expressions
TEST(Parser, parse_parenthesis_expression) {
    std::vector<const char*> good_expressions =
    {
"((celsius-22)/10)      ",
"((celsius-22)+10)      ",
"(x+2)                  ",
"((x))                  ",
"(((x)))                ",
"(x + (x * (y*(2)) + 4))",
    };

    for(auto const& expression : good_expressions) {
        Parser p(expression);
        auto e = p.parse_parenthesis_expression();

#ifdef VERBOSE_TEST
        if(e) std::cout << e->to_string() << std::endl;
#endif
        EXPECT_NE(e, nullptr);
        EXPECT_EQ(p.status(), lexerStatus::happy);

        // always print the compiler errors, because they are unexpected
        if(p.status()==lexerStatus::error)
            std::cout << cyan(expression) << "\t"
                      << red("error") << p.error_message() << std::endl;
    }

    std::vector<const char*> bad_expressions =
    {
"(x             ",
"((x+3)         ",
"(x+ +)         ",
"(x=3)          ",  // assignment inside parenthesis isn't allowed
"(a + (b*2^(x)) ",  // missing closing parenthesis
    };

    for(auto const& expression : bad_expressions) {
        Parser p(expression);
        auto e = p.parse_parenthesis_expression();

        EXPECT_EQ(e, nullptr);
        EXPECT_EQ(p.status(), lexerStatus::error);

#ifdef VERBOSE_TEST
        if(e) std::cout << e->to_string() << std::endl;
        if(p.status()==lexerStatus::error)
            std::cout << "in " << cyan(expression) << "\t" << p.error_message() << std::endl;
#endif
    }
}

// test parsing of line expressions
TEST(Parser, parse_line_expression) {
    std::vector<const char*> good_expressions =
    {
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

    for(auto const& expression : good_expressions) {
        Parser p(expression);
        auto e = p.parse_line_expression();

#ifdef VERBOSE_TEST
        if(e) std::cout << e->to_string() << std::endl;
#endif
        EXPECT_NE(e, nullptr);
        EXPECT_EQ(p.status(), lexerStatus::happy);

        // always print the compiler errors, because they are unexpected
        if(p.status()==lexerStatus::error)
            std::cout << red("error") << p.error_message() << std::endl;
    }

    std::vector<const char*> bad_expressions =
    {
"x=2+       ",      // incomplete binary expression on rhs
"x=         ",      // missing rhs of assignment
"x=)y + 2 * z",
"x=(y + 2   ",
"x=(y ++ z  ",
"x/=3       ",      // compound binary expressions not supported
"foo+8      ",      // missing assignment
"foo()=8    ",      // lhs of assingment must be an lvalue
    };

    for(auto const& expression : bad_expressions) {
        Parser p(expression);
        auto e = p.parse_line_expression();

        EXPECT_EQ(e, nullptr);
        EXPECT_EQ(p.status(), lexerStatus::error);

#ifdef VERBOSE_TEST
        if(e) std::cout << e->to_string() << std::endl;
        if(p.status()==lexerStatus::error)
            std::cout << "in " << cyan(expression) << "\t" << p.error_message() << std::endl;
#endif
    }
}

long double eval(Expression *e) {
    if(auto n = e->is_number()) {
        return n->value();
    }
    if(auto b = e->is_binary()) {
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
    if(auto u = e->is_unary()) {
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
    std::vector<std::pair<const char*, double>> tests =
    {
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

    for(auto const& test_case : tests) {
        Parser p(test_case.first);
        auto e = p.parse_expression();

#ifdef VERBOSE_TEST
        if(e) std::cout << e->to_string() << std::endl;
#endif
        EXPECT_NE(e, nullptr);
        EXPECT_EQ(p.status(), lexerStatus::happy);

        // A loose tolerance of 1d-10 is required here because the eval() function uses long double
        // for intermediate results (like constant folding in modparser).
        // For expressions with transcendental operations this can see relatively large divergence between
        // the double and long double results.
        EXPECT_NEAR(eval(e.get()), test_case.second, 1e-10);

        // always print the compiler errors, because they are unexpected
        if(p.status()==lexerStatus::error)
            std::cout << red("error") << p.error_message() << std::endl;
    }
}

