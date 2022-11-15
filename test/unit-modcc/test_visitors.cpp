#include "common.hpp"

#include "perfvisitor.hpp"
#include "parser.hpp"

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

    {
        FlopVisitor visitor;
        auto e = parse_expression("sqrt(x)");
        e->accept(&visitor);
        EXPECT_EQ(visitor.flops.sqrt, 1);
    }

    {
        FlopVisitor visitor;
        auto e = parse_expression("signum(x)");
        e->accept(&visitor);
        EXPECT_EQ(visitor.flops.add, 1);
    }

    {
        FlopVisitor visitor;
        auto e = parse_expression("step(x)");
        e->accept(&visitor);
        EXPECT_EQ(visitor.flops.add, 2);
        EXPECT_EQ(visitor.flops.mul, 1);
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
"    rho  = step_left(c-theta) + 1/sqrt(tau)*sigma\n"
"}";
    FlopVisitor visitor;
    auto e = parse_procedure(expression);
    e->accept(&visitor);
    EXPECT_EQ(visitor.flops.add, 7);
    EXPECT_EQ(visitor.flops.neg, 0);
    EXPECT_EQ(visitor.flops.mul, 1);
    EXPECT_EQ(visitor.flops.div, 6);
    EXPECT_EQ(visitor.flops.exp, 2);
    EXPECT_EQ(visitor.flops.pow, 1);
    EXPECT_EQ(visitor.flops.sqrt, 1);
}

TEST(FlopVisitor, function) {
    const char *expression =
"FUNCTION foo(v) {\n"
"    LOCAL qt\n"
"    qt=q10^((celsius- -22)/10)\n"
"    minf=1-1/(1+exp((v-vhalfm)/km))\n"
"    hinf=1/(1+exp((v-vhalfh)/kh))\n"
"    foo = minf + hinf\n"
"    rho  = signum(c-theta)/tau + 1/sqrt(tau)*sigma\n"
"}";
    FlopVisitor visitor;
    auto e = parse_function(expression);
    e->accept(&visitor);
    EXPECT_EQ(visitor.flops.add, 10);
    EXPECT_EQ(visitor.flops.neg, 1);
    EXPECT_EQ(visitor.flops.mul, 1);
    EXPECT_EQ(visitor.flops.div, 7);
    EXPECT_EQ(visitor.flops.exp, 2);
    EXPECT_EQ(visitor.flops.pow, 1);
    EXPECT_EQ(visitor.flops.sqrt, 1);
}

