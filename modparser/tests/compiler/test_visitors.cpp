#include "test.hpp"

#include "../src/constantfolder.hpp"
#include "../src/expressionclassifier.hpp"
//#include "../src/variablerenamer.hpp"
#include "../src/perfvisitor.hpp"

#include "../src/parser.hpp"
#include "../src/util.hpp"

/**************************************************************
 * visitors
 **************************************************************/

TEST(FlopVisitor, basic) {
    {
    auto visitor = make_unique<FlopVisitor>();
    auto e = parse_expression("x+y");
    e->accept(visitor.get());
    EXPECT_EQ(visitor->flops.add, 1);
    }

    {
    auto visitor = make_unique<FlopVisitor>();
    auto e = parse_expression("x-y");
    e->accept(visitor.get());
    EXPECT_EQ(visitor->flops.add, 1);
    }

    {
    auto visitor = make_unique<FlopVisitor>();
    auto e = parse_expression("x*y");
    e->accept(visitor.get());
    EXPECT_EQ(visitor->flops.mul, 1);
    }

    {
    auto visitor = make_unique<FlopVisitor>();
    auto e = parse_expression("x/y");
    e->accept(visitor.get());
    EXPECT_EQ(visitor->flops.div, 1);
    }

    {
    auto visitor = make_unique<FlopVisitor>();
    auto e = parse_expression("exp(x)");
    e->accept(visitor.get());
    EXPECT_EQ(visitor->flops.exp, 1);
    }

    {
    auto visitor = make_unique<FlopVisitor>();
    auto e = parse_expression("log(x)");
    e->accept(visitor.get());
    EXPECT_EQ(visitor->flops.log, 1);
    }

    {
    auto visitor = make_unique<FlopVisitor>();
    auto e = parse_expression("cos(x)");
    e->accept(visitor.get());
    EXPECT_EQ(visitor->flops.cos, 1);
    }

    {
    auto visitor = make_unique<FlopVisitor>();
    auto e = parse_expression("sin(x)");
    e->accept(visitor.get());
    EXPECT_EQ(visitor->flops.sin, 1);
    }
}

TEST(FlopVisitor, compound) {
    {
        auto visitor = make_unique<FlopVisitor>();
    auto e = parse_expression("x+y*z/a-b");
    e->accept(visitor.get());
    EXPECT_EQ(visitor->flops.add, 2);
    EXPECT_EQ(visitor->flops.mul, 1);
    EXPECT_EQ(visitor->flops.div, 1);
    }

    {
        auto visitor = make_unique<FlopVisitor>();
    auto e = parse_expression("exp(x+y+z)");
    e->accept(visitor.get());
    EXPECT_EQ(visitor->flops.add, 2);
    EXPECT_EQ(visitor->flops.exp, 1);
    }

    {
        auto visitor = make_unique<FlopVisitor>();
    auto e = parse_expression("exp(x+y) + 3/(12 + z)");
    e->accept(visitor.get());
    EXPECT_EQ(visitor->flops.add, 3);
    EXPECT_EQ(visitor->flops.div, 1);
    EXPECT_EQ(visitor->flops.exp, 1);
    }

    // test asssignment expression
    {
        auto visitor = make_unique<FlopVisitor>();
    auto e = parse_line_expression("x = exp(x+y) + 3/(12 + z)");
    e->accept(visitor.get());
    EXPECT_EQ(visitor->flops.add, 3);
    EXPECT_EQ(visitor->flops.div, 1);
    EXPECT_EQ(visitor->flops.exp, 1);
    }
}

TEST(FlopVisitor, procedure) {
    {
    const char *expression =
"PROCEDURE trates(v) {\n"
"    LOCAL qt\n"
"    qt=q10^((celsius-22)/10)\n"
"    minf=1-1/(1+exp((v-vhalfm)/km))\n"
"    hinf=1/(1+exp((v-vhalfh)/kh))\n"
"    mtau = 0.6\n"
"    htau = 1500\n"
"}";
    auto visitor = make_unique<FlopVisitor>();
    auto e = parse_procedure(expression);
    e->accept(visitor.get());
    EXPECT_EQ(visitor->flops.add, 6);
    EXPECT_EQ(visitor->flops.neg, 0);
    EXPECT_EQ(visitor->flops.mul, 0);
    EXPECT_EQ(visitor->flops.div, 5);
    EXPECT_EQ(visitor->flops.exp, 2);
    EXPECT_EQ(visitor->flops.pow, 1);
    }
}

TEST(FlopVisitor, function) {
    {
    const char *expression =
"FUNCTION foo(v) {\n"
"    LOCAL qt\n"
"    qt=q10^((celsius- -22)/10)\n"
"    minf=1-1/(1+exp((v-vhalfm)/km))\n"
"    hinf=1/(1+exp((v-vhalfh)/kh))\n"
"    foo = minf + hinf\n"
"}";
    auto visitor = make_unique<FlopVisitor>();
    auto e = parse_function(expression);
    e->accept(visitor.get());
    EXPECT_EQ(visitor->flops.add, 7);
    EXPECT_EQ(visitor->flops.neg, 1);
    EXPECT_EQ(visitor->flops.mul, 0);
    EXPECT_EQ(visitor->flops.div, 5);
    EXPECT_EQ(visitor->flops.exp, 2);
    EXPECT_EQ(visitor->flops.pow, 1);
    }
}

TEST(ClassificationVisitor, linear) {
    std::vector<const char*> expressions =
    {
"x + y + z",
"y + x + z",
"y + z + x",
"x - y - z",
"y - x - z",
"y - z - x",
"z*(x + y + 2)",
"(x + y)*z",
"(x + y)/z",
"x+3",
"-x",
"x+x+x+x",
"2*x     ",
"y*x     ",
"x + y   ",
"y + x   ",
"y + z*x ",
"2*(x*z + y)",
"z*x - y",
"(2+z)*(x*z + y)",
"x/y",
"(y - x)/z",
"(x - y)/z",
"y*(x - y)/z",
    };

    // create a scope that contains the symbols used in the tests
    Scope<Symbol>::symbol_map globals;
    globals["x"] = make_symbol<LocalVariable>(Location(), "x");
    globals["y"] = make_symbol<LocalVariable>(Location(), "y");
    globals["z"] = make_symbol<LocalVariable>(Location(), "z");
    auto x = globals["x"].get();

    auto scope = std::make_shared<Scope<Symbol>>(globals);

    for(auto const& expression : expressions) {
        auto e = parse_expression(expression);

        // sanity check the compiler
        EXPECT_NE(e, nullptr);
        if( e==nullptr ) continue;

        e->semantic(scope);
        auto v = new ExpressionClassifierVisitor(x);
        e->accept(v);
        //std::cout << "expression " << e->to_string() << std::endl;
        //std::cout << "linear     " << v->linear_coefficient()->to_string() << std::endl;
        //std::cout << "constant   " << v->constant_term()->to_string() << std::endl;
        EXPECT_EQ(v->classify(), expressionClassification::linear);

#ifdef VERBOSE_TEST
        std::cout << "eq    "   << e->to_string()
                  << "\ncoeff " << v->linear_coefficient()->to_string()
                  << "\nconst " << v-> constant_term()->to_string()
                  << "\n----"   << std::endl;
#endif
        delete v;
    }
}

TEST(ClassificationVisitor, constant) {
    std::vector<const char*> expressions =
    {
"y+3",
"-y",
"exp(y+z)",
"1",
"y^z",
    };

    // create a scope that contains the symbols used in the tests
    Scope<Symbol>::symbol_map globals;
    globals["x"] = make_symbol<LocalVariable>(Location(), "x");
    globals["y"] = make_symbol<LocalVariable>(Location(), "y");
    globals["z"] = make_symbol<LocalVariable>(Location(), "z");
    auto scope = std::make_shared<Scope<Symbol>>(globals);
    auto x = globals["x"].get();

    for(auto const& expression : expressions) {
        auto e = parse_expression(expression);

        // sanity check the compiler
        EXPECT_NE(e, nullptr);
        if( e==nullptr ) continue;

        e->semantic(scope);
        auto v = new ExpressionClassifierVisitor(x);
        e->accept(v);
        EXPECT_EQ(v->classify(), expressionClassification::constant);

#ifdef VERBOSE_TEST
        if(e) std::cout << e->to_string() << std::endl;
        if(p.status()==lexerStatus::error)
            std::cout << "in " << colorize(expression, kCyan) << "\t" << p.error_message() << std::endl;
#endif
        delete v;
    }
}

TEST(ClassificationVisitor, nonlinear) {
    std::vector<const char*> expressions =
    {
"x*x",
"x*2*x",
"x*(2+x)",
"y/x",
"x*(y + z*(x/y))",
"exp(x)",
"exp(x+y)",
"exp(z*(x+y))",
"log(x)",
"cos(x)",
"sin(x)",
"x^y",
"y^x",
    };

    // create a scope that contains the symbols used in the tests
    Scope<Symbol>::symbol_map globals;
    globals["x"] = make_symbol<LocalVariable>(Location(), "x");
    globals["y"] = make_symbol<LocalVariable>(Location(), "y");
    globals["z"] = make_symbol<LocalVariable>(Location(), "z");
    auto scope = std::make_shared<Scope<Symbol>>(globals);
    auto x = globals["x"].get();

    auto v = new ExpressionClassifierVisitor(x);
    for(auto const& expression : expressions) {
        auto e = parse_expression(expression);

        // sanity check the compiler
        EXPECT_NE(e, nullptr);
        if( e==nullptr ) continue;

        e->semantic(scope);
        v->reset();
        e->accept(v);
        EXPECT_EQ(v->classify(), expressionClassification::nonlinear);

#ifdef VERBOSE_TEST
        if(e) std::cout << e->to_string() << std::endl;
        if(p.status()==lexerStatus::error)
            std::cout << "in " << colorize(expression, kCyan) << "\t" << p.error_message() << std::endl;
#endif
    }
    delete v;
}

