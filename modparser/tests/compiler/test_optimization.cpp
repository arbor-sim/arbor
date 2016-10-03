#include <cmath>

#include "test.hpp"

#include "../src/constantfolder.hpp"

#include "../src/util.hpp"

TEST(Optimizer, constant_folding) {
    auto v = make_unique<ConstantFolderVisitor>();
    {
        auto e = parse_line_expression("x = 2*3");
        VERBOSE_PRINT( e->to_string() )
        e->accept(v.get());
        EXPECT_EQ(e->is_assignment()->rhs()->is_number()->value(), 6);
        VERBOSE_PRINT( e->to_string() )
        VERBOSE_PRINT( "" )
    }
    {
        auto e = parse_line_expression("x = 1 + 2 + 3");
        VERBOSE_PRINT( e->to_string() )
        e->accept(v.get());
        EXPECT_EQ(e->is_assignment()->rhs()->is_number()->value(), 6);
        VERBOSE_PRINT( e->to_string() )
        VERBOSE_PRINT( "" )
    }
    {
        auto e = parse_line_expression("x = exp(2)");
        VERBOSE_PRINT( e->to_string() )
        e->accept(v.get());
        // The tolerance has to be loosend to 1e-15, because the optimizer performs
        // all intermediate calculations in 80 bit precision, which disagrees in
        // the 16 decimal place to the double precision value from std::exp(2.0).
        // This is a good thing: by using the constant folder we increase accuracy
        // over the unoptimized code!
        EXPECT_EQ(std::fabs(e->is_assignment()->rhs()->is_number()->value()-std::exp(2.0))<1e-15, true);
        VERBOSE_PRINT( e->to_string() )
        VERBOSE_PRINT( "" )
    }
    {
        auto e = parse_line_expression("x= 2*2 + 3");
        VERBOSE_PRINT( e->to_string() )
        e->accept(v.get());
        EXPECT_EQ(e->is_assignment()->rhs()->is_number()->value(), 7);
        VERBOSE_PRINT( e->to_string() )
        VERBOSE_PRINT( "" )
    }
    {
        auto e = parse_line_expression("x= 3 + 2*2");
        VERBOSE_PRINT( e->to_string() )
        e->accept(v.get());
        EXPECT_EQ(e->is_assignment()->rhs()->is_number()->value(), 7);
        VERBOSE_PRINT( e->to_string() )
        VERBOSE_PRINT( "" )
    }
    {
        // this doesn't work: the (y+2) expression is not a constant, so folding stops.
        // we need to fold the 2+3, which isn't possible with a simple walk.
        // one approach would be try sorting communtative operations so that numbers
        // are adjacent to one another in the tree
        auto e = parse_line_expression("x= y + 2 + 3");
        VERBOSE_PRINT( e->to_string() )
        e->accept(v.get());
        VERBOSE_PRINT( e->to_string() )
        VERBOSE_PRINT( "" )
    }
    {
        auto e = parse_line_expression("x= 2 + 3 + y");
        VERBOSE_PRINT( e->to_string() )
        e->accept(v.get());
        VERBOSE_PRINT( e->to_string() )
        VERBOSE_PRINT("");
    }
    {
        auto e = parse_line_expression("foo(2+3, log(32), 2*3 + x)");
        VERBOSE_PRINT( e->to_string() )
        e->accept(v.get());
        VERBOSE_PRINT( e->to_string() )
        VERBOSE_PRINT("");
    }
}

