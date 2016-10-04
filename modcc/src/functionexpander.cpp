#include <iostream>

#include "error.hpp"
#include "functionexpander.hpp"
#include "util.hpp"

///////////////////////////////////////////////////////////////////////////////
//  function call site lowering
///////////////////////////////////////////////////////////////////////////////

call_list_type lower_function_calls(Expression* e)
{
    auto v = make_unique<FunctionCallLowerer>(e->scope());

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
Symbol* make_unique_local(std::shared_ptr<Scope<Symbol>> scope) {
    std::string name;
    auto i = 0;
    do {
        name = pprintf("ll%_", i);
        ++i;
    } while(scope->find(name));

    return
        scope->add_local_symbol(
            name,
            make_symbol<LocalVariable>(
                Location(), name, localVariableKind::local
            )
        );
}

call_list_type
lower_function_arguments(std::vector<expression_ptr>& args)
{
    call_list_type new_statements;
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

        // use the source location of the original statement
        auto loc = e->location();

        // make an identifier for the new symbol which will store the result of
        // the function call
        auto id = make_expression<IdentifierExpression>
            (loc, make_unique_local(e->scope())->name());
        id->semantic(e->scope());

        // generate a LOCAL declaration for the variable
        new_statements.push_front(
            make_expression<LocalDeclaration>(loc, id->is_identifier()->spelling())
        );

        // make a binary expression which assigns the argument to the variable
        auto ass = binary_expression(loc, tok::eq, id->clone(), e->clone());
        ass->semantic(e->scope());
#ifdef LOGGING
        std::cout << "  lowering to " << ass->to_string() << "\n";
#endif
        new_statements.push_back(std::move(ass));

        // replace the function call in the original expression with the local
        // variable which holds the pre-computed value
        std::swap(e, id);
    }
#ifdef LOGGING
    std::cout << "\n";
#endif

    return new_statements;
}

