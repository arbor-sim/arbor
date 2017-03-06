#include "test.hpp"

#include "constantfolder.hpp"
#include "expressionclassifier.hpp"
#include "perfvisitor.hpp"
#include "parser.hpp"
#include "modccutil.hpp"

// overload for parser errors
template <typename EPtr>
void verbose_print(const EPtr& e, Parser& p, const char* text) {
    verbose_print(e);
    if (p.status()==lexerStatus::error) {
        verbose_print("in ", cyan(text), "\t", p.error_message());
    }
}

/**************************************************************
 * visitors
 **************************************************************/

TEST(FlopVisitor, basic) {
    {
        FlopVisitor visitor;
        auto e = parse_expression("x+y");
        e->accept(&visitor);
        EXPECT_EQ(visitor.flops.add, 1);
    }

    {
        FlopVisitor visitor;
        auto e = parse_expression("x-y");
        e->accept(&visitor);
        EXPECT_EQ(visitor.flops.add, 1);
    }

    {
        FlopVisitor visitor;
        auto e = parse_expression("x*y");
        e->accept(&visitor);
        EXPECT_EQ(visitor.flops.mul, 1);
    }

    {
        FlopVisitor visitor;
        auto e = parse_expression("x/y");
        e->accept(&visitor);
        EXPECT_EQ(visitor.flops.div, 1);
    }

    {
        FlopVisitor visitor;
        auto e = parse_expression("exp(x)");
        e->accept(&visitor);
        EXPECT_EQ(visitor.flops.exp, 1);
    }

    {
        FlopVisitor visitor;
        auto e = parse_expression("log(x)");
        e->accept(&visitor);
        EXPECT_EQ(visitor.flops.log, 1);
    }

    {
        FlopVisitor visitor;
        auto e = parse_expression("cos(x)");
        e->accept(&visitor);
        EXPECT_EQ(visitor.flops.cos, 1);
    }

    {
        FlopVisitor visitor;
        auto e = parse_expression("sin(x)");
        e->accept(&visitor);
        EXPECT_EQ(visitor.flops.sin, 1);
    }
}

TEST(FlopVisitor, compound) {
    {
        FlopVisitor visitor;
        auto e = parse_expression("x+y*z/a-b");
        e->accept(&visitor);
        EXPECT_EQ(visitor.flops.add, 2);
        EXPECT_EQ(visitor.flops.mul, 1);
        EXPECT_EQ(visitor.flops.div, 1);
    }

    {
        FlopVisitor visitor;
        auto e = parse_expression("exp(x+y+z)");
        e->accept(&visitor);
        EXPECT_EQ(visitor.flops.add, 2);
        EXPECT_EQ(visitor.flops.exp, 1);
    }

    {
        FlopVisitor visitor;
        auto e = parse_expression("exp(x+y) + 3/(12 + z)");
        e->accept(&visitor);
        EXPECT_EQ(visitor.flops.add, 3);
        EXPECT_EQ(visitor.flops.div, 1);
        EXPECT_EQ(visitor.flops.exp, 1);
    }

    // test asssignment expression
    {
        FlopVisitor visitor;
        auto e = parse_line_expression("x = exp(x+y) + 3/(12 + z)");
        e->accept(&visitor);
        EXPECT_EQ(visitor.flops.add, 3);
        EXPECT_EQ(visitor.flops.div, 1);
        EXPECT_EQ(visitor.flops.exp, 1);
    }
}

TEST(FlopVisitor, procedure) {
    const char *expression =
"PROCEDURE trates(v) {\n"
"    LOCAL qt\n"
"    qt=q10^((celsius-22)/10)\n"
"    minf=1-1/(1+exp((v-vhalfm)/km))\n"
"    hinf=1/(1+exp((v-vhalfh)/kh))\n"
"    mtau = 0.6\n"
"    htau = 1500\n"
"}";
    FlopVisitor visitor;
    auto e = parse_procedure(expression);
    e->accept(&visitor);
    EXPECT_EQ(visitor.flops.add, 6);
    EXPECT_EQ(visitor.flops.neg, 0);
    EXPECT_EQ(visitor.flops.mul, 0);
    EXPECT_EQ(visitor.flops.div, 5);
    EXPECT_EQ(visitor.flops.exp, 2);
    EXPECT_EQ(visitor.flops.pow, 1);
}

TEST(FlopVisitor, function) {
    const char *expression =
"FUNCTION foo(v) {\n"
"    LOCAL qt\n"
"    qt=q10^((celsius- -22)/10)\n"
"    minf=1-1/(1+exp((v-vhalfm)/km))\n"
"    hinf=1/(1+exp((v-vhalfh)/kh))\n"
"    foo = minf + hinf\n"
"}";
    FlopVisitor visitor;
    auto e = parse_function(expression);
    e->accept(&visitor);
    EXPECT_EQ(visitor.flops.add, 7);
    EXPECT_EQ(visitor.flops.neg, 1);
    EXPECT_EQ(visitor.flops.mul, 0);
    EXPECT_EQ(visitor.flops.div, 5);
    EXPECT_EQ(visitor.flops.exp, 2);
    EXPECT_EQ(visitor.flops.pow, 1);
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
        ExpressionClassifierVisitor v(x);
        e->accept(&v);
        EXPECT_EQ(v.classify(), expressionClassification::linear);

        verbose_print("eq    ", e);
        verbose_print("coeff ", v.linear_coefficient());
        verbose_print("const ", v.constant_term());
        verbose_print("----");
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
        Parser p{expression};
        auto e = p.parse_expression();

        // sanity check the compiler
        EXPECT_NE(e, nullptr);
        if( e==nullptr ) continue;

        e->semantic(scope);
        ExpressionClassifierVisitor v(x);
        e->accept(&v);
        EXPECT_EQ(v.classify(), expressionClassification::constant);

        verbose_print(e, p, expression);
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

    ExpressionClassifierVisitor v(x);
    for(auto const& expression : expressions) {
        Parser p{expression};
        auto e = p.parse_expression();

        // sanity check the compiler
        EXPECT_NE(e, nullptr);
        if( e==nullptr ) continue;

        e->semantic(scope);
        v.reset();
        e->accept(&v);
        EXPECT_EQ(v.classify(), expressionClassification::nonlinear);

        verbose_print(e, p, expression);
    }
}
