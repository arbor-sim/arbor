#include <iostream>
#include <memory>

#include "astmanip.hpp"
#include "error.hpp"
#include "functionexpander.hpp"

expression_ptr insert_unique_local_assignment(expr_list_type& stmts, Expression* e) {
    auto exprs = make_unique_local_assign(e->scope(), e, "fu");
    std::cout << "created: " << exprs.id->is_identifier()->spelling() << " for " << e->to_string() << std::endl;
    stmts.push_front(std::move(exprs.local_decl));
    stmts.push_back(std::move(exprs.assignment));
    return std::move(exprs.id);
}

///////////////////////////////////////////////////////////////////////////////
//  function call site lowering
///////////////////////////////////////////////////////////////////////////////

expr_list_type lower_function_calls(Expression* e)
{
    auto v = std::make_unique<FunctionCallLowerer>(e->scope());

    if(auto a=e->is_assignment()) {
#ifdef LOGGING
        std::cout << "lower_function_calls inspect expression " << e->to_string() << "\n";
#endif
        // recursively inspect and replace function calls with identifiers
        a->rhs()->accept(v.get());

    }

    // return the list of statements that assign function call return values
    // to identifiers, e.g.
    //      LOCAL ll1_
    //      ll1_ = mInf(v)
    return v->move_calls();
}

void FunctionCallLowerer::visit(Expression *e) {
    throw compiler_exception(
        "function lowering for expressions of the type " + e->to_string()
        + " has not been defined", e->location()
    );
}

void FunctionCallLowerer::visit(CallExpression *e) {
    for(auto& arg : e->args()) {
        if(auto func = arg->is_function_call()) {
            func->accept(this);
#ifdef LOGGING
            std::cout << "  lowering : " << func->to_string() << "\n";
#endif
            expand_call(
                func, [&arg](expression_ptr&& p){arg = std::move(p);}
            );
            arg->semantic(scope_);
        }
        else {
            arg->accept(this);
        }
    }
}

void FunctionCallLowerer::visit(UnaryExpression *e) {
    if(auto func = e->expression()->is_function_call()) {
        func->accept(this);
#ifdef LOGGING
        std::cout << "  lowering : " << func->to_string() << "\n";
#endif
        expand_call(func, [&e](expression_ptr&& p){e->replace_expression(std::move(p));});
        e->semantic(scope_);
    }
    else {
        e->expression()->accept(this);
    }
}

void FunctionCallLowerer::visit(BinaryExpression *e) {
    if(auto func = e->lhs()->is_function_call()) {
        func->accept(this);
#ifdef LOGGING
        std::cout << "  lowering : " << func->to_string() << "\n";
#endif
        expand_call(func, [&e](expression_ptr&& p){e->replace_lhs(std::move(p));});
        e->semantic(scope_);
    }
    else {
        e->lhs()->accept(this);
    }

    if(auto func = e->rhs()->is_function_call()) {
        func->accept(this);
#ifdef LOGGING
        std::cout << "  lowering : " << func->to_string() << "\n";
#endif
        expand_call(func, [&e](expression_ptr&& p){e->replace_rhs(std::move(p));});
        e->semantic(scope_);
    }
    else {
        e->rhs()->accept(this);
    }
}

///////////////////////////////////////////////////////////////////////////////
//  function argument lowering
///////////////////////////////////////////////////////////////////////////////

expr_list_type
lower_function_arguments(std::vector<expression_ptr>& args)
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
}

