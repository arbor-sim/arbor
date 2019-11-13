#include <iostream>
#include <memory>

#include "astmanip.hpp"
#include "error.hpp"
#include "functionexpander.hpp"

expression_ptr insert_unique_local_assignment(expr_list_type& stmts, Expression* e) {
    auto exprs = make_unique_local_assign(e->scope(), e);
    stmts.push_front(std::move(exprs.local_decl));
    stmts.push_back(std::move(exprs.assignment));
    return std::move(exprs.id);
}

///////////////////////////////////////////////////////////////////////////////
//  function call site lowering
///////////////////////////////////////////////////////////////////////////////

// lower function call sites so that all function calls are of
// the form : variable = call(<args>)
// e.g.
//      a = 2 + foo(2+x, y, 1)
// becomes
//      ll0_ = foo(2+x, y, 1)
//      a = 2 + ll0_
expression_ptr lower_function_calls(BlockExpression* block) {
    auto v = std::make_unique<FunctionCallLowerer>();
    block->accept(v.get());
    return v->as_block(false);
}

void FunctionCallLowerer::visit(CallExpression *e) {
    for(auto& arg : e->args()) {
        if(auto func = arg->is_function_call()) {
            func->accept(this);
#ifdef LOGGING
            std::cout << "  lowering : " << func->to_string() << "\n";
#endif
            lower_call_arguments(func->args());
            expand_call(func, [&arg](expression_ptr&& p){arg = std::move(p);});
            arg->semantic(block_scope_);
        }
        else {
            arg->accept(this);
        }
    }
    lower_call_arguments(e->args());

    if (e->is_procedure_call()) {
        statements_.push_back(e->clone());
    }
}

void FunctionCallLowerer::visit(AssignmentExpression *e) {
    e->rhs()->accept(this);
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

void FunctionCallLowerer::visit(BinaryExpression *e) {
    if(auto func = e->lhs()->is_function_call()) {
        func->accept(this);
#ifdef LOGGING
        std::cout << "  lowering : " << func->to_string() << "\n";
#endif
        lower_call_arguments(func->args());
        expand_call(func, [&e](expression_ptr&& p){e->replace_lhs(std::move(p));});
        e->semantic(block_scope_);
    }
    else {
        e->lhs()->accept(this);
    }

    if(auto func = e->rhs()->is_function_call()) {
        func->accept(this);
#ifdef LOGGING
        std::cout << "  lowering : " << func->to_string() << "\n";
#endif
        lower_call_arguments(func->args());
        expand_call(func, [&e](expression_ptr&& p){e->replace_rhs(std::move(p));});
        e->semantic(block_scope_);
    }
    else {
        e->rhs()->accept(this);
    }
}

void FunctionCallLowerer::visit(UnaryExpression *e) {
    if(auto func = e->expression()->is_function_call()) {
        func->accept(this);
#ifdef LOGGING
        std::cout << "  lowering : " << func->to_string() << "\n";
#endif
        lower_call_arguments(func->args());
        expand_call(func, [&e](expression_ptr&& p){e->replace_expression(std::move(p));});
        e->semantic(block_scope_);
    }
    else {
        e->expression()->accept(this);
    }
}
void FunctionCallLowerer::visit(IfExpression *e) {
    expr_list_type outer;

    e->condition()->accept(this);
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

///////////////////////////////////////////////////////////////////////////////
//  function argument lowering
///////////////////////////////////////////////////////////////////////////////

/*void lower_function_arguments(std::vector<expression_ptr>& args)
{
    expr_list_type new_statements;
    for(auto it=args.begin(); it!=args.end(); ++it) {
        // get reference to the unique_ptr with the expression
        auto& e = *it;
#ifdef LOGGING
        std::cout << "inspecting argument @ " << e->location() << " : " << e->to_string() << std::endl;
#endif

        if(e->is_number() || e->is_identifier()) {
            // do nothing, because identifiers and literals are in the correct form
            // for lowering
            continue;
        }

        auto id = insert_unique_local_assignment(new_statements, e.get());
#ifdef LOGGING
        std::cout << "  lowering to " << new_statements.back()->to_string() << "\n";
#endif
        // replace the function call in the original expression with the local
        // variable which holds the pre-computed value
        std::swap(e, id);
    }
#ifdef LOGGING
    std::cout << "\n";
#endif

    return new_statements;
}*/

