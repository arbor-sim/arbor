#include <iostream>
#include <memory>

#include "astmanip.hpp"
#include "error.hpp"
#include "functionexpander.hpp"

ARB_LIBMODCC_API expression_ptr insert_unique_local_assignment(expr_list_type& stmts, Expression* e) {
    auto zero = make_expression<NumberExpression>(e->location(), 0.);
    auto exprs = make_unique_local_assign(e->scope(), zero);

    stmts.push_front(std::move(exprs.assignment));
    stmts.push_front(std::move(exprs.local_decl));

    auto assign =  make_expression<AssignmentExpression>(e->location(), exprs.id->clone(),  e->clone());
    assign->semantic(e->scope());
    stmts.push_back(std::move(assign));

    return std::move(exprs.id);
}


/////////////////////////////////////////////////////////////////////
// lower function call sites so that all function calls are of
// the form : variable = call(<args>)
// then lower function arguments that are not identifiers or literals
// e.g.
//      a = 2 + foo(2+x, y, 1)
// becomes
//      ll0_ = foo(2+x, y, 1)
//      a = 2 + ll0_
// becomes
//       ll1_ = 2+x
//       ll0_ = foo(ll1_, y, 1)
//       a = 2 + ll0_
/////////////////////////////////////////////////////////////////////
ARB_LIBMODCC_API expression_ptr lower_functions(BlockExpression* block) {
    auto v = std::make_unique<FunctionCallLowerer>();
    block->accept(v.get());
    return v->as_block(false);
}

// We only need to lower function arguments when visiting a Call expression
// Function arguments are checked for other Call expressions, which recurse.
// When all Call arguments are handled, other arguments are checked, and
// lowered if needed
// e.g. foo(bar(x + 2), y - 1)
// First, the visitor recurses for bar(x + 2) which gets its arguments lowered:
//      ll0_ = x + 2;
//      bar(ll0_);
// Then, bar(x + 2) gets expanded into
//      ll1_ = bar(ll0_);
//      foo(ll1_, y - 1);
// Finally, foo(ll1_, y - 1) gets its arguments lowered into
//      ll2_ = y - 1;
//      foo(ll1_, ll2_);
// which turns:
//      foo(bar(x + 2), y - 1)
// into:
//      ll0_ = x + 2;
//      ll1_ = bar(ll0_);
//      ll2_ = y - 1;
//      foo(ll1_, ll2_);
void FunctionCallLowerer::visit(CallExpression *e) {
    // Lower function calls
    for(auto& arg : e->args()) {
        if(auto func = arg->is_function_call()) {
            // Recurse on the Call Expression
            func->accept(this);
            expand_call(func, [&arg](expression_ptr&& p){arg = std::move(p);});
            arg->semantic(block_scope_);
        }
        else {
            arg->accept(this);
        }
    }
    // Lower function arguments
    for(auto& arg : e->args()) {
        if(arg->is_number() || arg->is_identifier()) {
            continue;
        }
        auto id = insert_unique_local_assignment(statements_, arg.get());
        std::swap(arg, id);
    }

    // Procedure Expressions need to be printed stand-alone
    // Function Expressions are always part of a bigger expression
    if (e->is_procedure_call()) {
        statements_.push_back(e->clone());
    }
}

void FunctionCallLowerer::visit(AssignmentExpression *e) {
    e->rhs()->accept(this);
    if (auto func = e->rhs()->is_function_call()) {
        for (auto& arg: func->args()) {
            if (auto id = arg->is_identifier()) {
                if (id->name() == e->lhs()->is_identifier()->name()) {
                    expand_call(func, [&e](expression_ptr&& p){e->replace_rhs(std::move(p));});
                    e->semantic(block_scope_);
                    break;
                }
            }
        }
    }
    statements_.push_back(e->clone());
}

void FunctionCallLowerer::visit(ConserveExpression *e) {
    statements_.push_back(e->clone());
}

void FunctionCallLowerer::visit(CompartmentExpression *e) {
    statements_.push_back(e->clone());
}

void FunctionCallLowerer::visit(LinearExpression *e) {
    statements_.push_back(e->clone());
}

// Binary Expressions need to handle function calls if they contain them
// Functions calls have to be visited and expanded out of the expression
void FunctionCallLowerer::visit(BinaryExpression *e) {
    if(auto func = e->lhs()->is_function_call()) {
        func->accept(this);
        expand_call(func, [&e](expression_ptr&& p){e->replace_lhs(std::move(p));});
        e->semantic(block_scope_);
    }
    else {
        e->lhs()->accept(this);
    }

    if(auto func = e->rhs()->is_function_call()) {
        func->accept(this);
        expand_call(func, [&e](expression_ptr&& p){e->replace_rhs(std::move(p));});
        e->semantic(block_scope_);
    }
    else {
        e->rhs()->accept(this);
    }
}

// Unary Expressions need to handle function calls if they contain them
// Functions calls have to be visited and expanded out of the expression
void FunctionCallLowerer::visit(UnaryExpression *e) {
    if(auto func = e->expression()->is_function_call()) {
        func->accept(this);
        expand_call(func, [&e](expression_ptr&& p){e->replace_expression(std::move(p));});
        e->semantic(block_scope_);
    }
    else {
        e->expression()->accept(this);
    }
}

// If expressions need to handle the condition before the true and false branches
// The condition should be handled by the Binary Expression visitor which will
// expand any contained function calls and lower their arguments
void FunctionCallLowerer::visit(IfExpression *e) {
    expr_list_type outer;

    e->condition()->accept(this);

    if(auto func = e->condition()->is_function_call()) {
        expand_call(func, [&e](expression_ptr&& p){
            auto zero_exp = make_expression<NumberExpression>(Location{}, 0.);
            p = make_expression<ConditionalExpression>(p->location(), tok::ne, p->clone(), std::move(zero_exp));
            e->replace_condition(std::move(p));
        });
        e->semantic(block_scope_);
    }

    std::swap(outer, statements_);

    e->true_branch()->accept(this);
    auto true_branch = make_expression<BlockExpression>(
            e->true_branch()->location(),
            std::move(statements_),
            true);

    statements_.clear();
    expression_ptr false_branch;
    if (e->false_branch()) {
        e->false_branch()->accept(this);
        false_branch = make_expression<BlockExpression>(
                e->false_branch()->location(),
                std::move(statements_),
                true);
    }

    statements_ = std::move(outer);
    statements_.push_back(make_expression<IfExpression>(
            e->location(),
            e->condition()->clone(),
            std::move(true_branch),
            std::move(false_branch)));
}

