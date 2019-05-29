#include <iostream>

#include "error.hpp"
#include "functioninliner.hpp"
#include "errorvisitor.hpp"

expression_ptr inline_function_call(Expression* e)
{
    if(auto f=e->is_function_call()) {
        auto func = f->function();
#ifdef LOGGING
        std::cout << "inline_function_call for statement " << f->to_string()
                  << " with body" << func->body()->to_string() << "\n";
#endif
        auto& body = func->body()->statements();
        if(body.size() != 1) {
            throw compiler_exception(
                "can only inline functions with one statement", func->location()
            );
        }

        if(body.front()->is_if()) {
            throw compiler_exception(
                    "can not inline functions with if statements", func->location()
            );
        }

        // assume that the function body is correctly formed, with the last
        // statement being an assignment expression
        auto last = body.front()->is_assignment();
        auto new_e = last->rhs()->clone();

        auto& fargs = func->args(); // argument names for the function
        auto& cargs = f->args();    // arguments at the call site
        for(auto i=0u; i<fargs.size(); ++i) {
            if(auto id = cargs[i]->is_identifier()) {
#ifdef LOGGING
                std::cout << "inline_function_call symbol replacement "
                          << id->to_string() << " -> " << fargs[i]->to_string()
                          << " in the expression " << new_e->to_string() << "\n";
#endif
                VariableReplacer v(
                    fargs[i]->is_argument()->spelling(),
                    id->spelling()
                );
                new_e->accept(&v);
            }
            else if(auto value = cargs[i]->is_number()) {
#ifdef LOGGING
                std::cout << "inline_function_call symbol replacement "
                          << value->to_string() << " -> " << fargs[i]->to_string()
                          << " in the expression " << new_e->to_string() << "\n";
#endif
                ValueInliner v(
                    fargs[i]->is_argument()->spelling(),
                    value->value()
                );
                new_e->accept(&v);
            }
            else {
                throw compiler_exception(
                    "can't inline functions with expressions as arguments",
                     e->location()
                 );
            }
        }
        new_e->semantic(e->scope());

        ErrorVisitor v("");
        new_e->accept(&v);
#ifdef LOGGING
        std::cout << "inline_function_call result " << new_e->to_string() << "\n\n";
#endif
        if(v.num_errors()) {
            throw compiler_exception("something went wrong with inlined function call ",
                                     e->location());
        }

        return new_e;
    }

    return {};
}

///////////////////////////////////////////////////////////////////////////////
//  variable replacer
///////////////////////////////////////////////////////////////////////////////

void VariableReplacer::visit(Expression *e) {
    throw compiler_exception(
            "I don't know how to variable inlining for this statement : "
            + e->to_string(), e->location());
}

void VariableReplacer::visit(UnaryExpression *e) {
    auto exp = e->expression()->is_identifier();
    if(exp && exp->spelling()==source_) {
        e->replace_expression(
            make_expression<IdentifierExpression>(exp->location(), target_)
        );
    }
    else if(!exp) {
        e->expression()->accept(this);
    }
}

void VariableReplacer::visit(BinaryExpression *e) {
    auto lhs = e->lhs()->is_identifier();
    if(lhs && lhs->spelling()==source_) {
        e->replace_lhs(
            make_expression<IdentifierExpression>(lhs->location(), target_)
        );
    }
    else if(!lhs){ // only inspect subexpressions that are not themselves identifiers
        e->lhs()->accept(this);
    }

    auto rhs = e->rhs()->is_identifier();
    if(rhs && rhs->spelling()==source_) {
        e->replace_rhs(
            make_expression<IdentifierExpression>(rhs->location(), target_)
        );
    }
    else if(!rhs){ // only inspect subexpressions that are not themselves identifiers
        e->rhs()->accept(this);
    }
}

///////////////////////////////////////////////////////////////////////////////
//  value inliner
///////////////////////////////////////////////////////////////////////////////

void ValueInliner::visit(Expression *e) {
    throw compiler_exception(
            "I don't know how to value inlining for this statement : "
            + e->to_string(), e->location());
}

void ValueInliner::visit(UnaryExpression *e) {
    auto exp = e->expression()->is_identifier();
    if(exp && exp->spelling()==source_) {
        e->replace_expression(
            make_expression<NumberExpression>(exp->location(), value_)
        );
    }
    else if(!exp){
        e->expression()->accept(this);
    }
}

void ValueInliner::visit(BinaryExpression *e) {
    auto lhs = e->lhs()->is_identifier();
    if(lhs && lhs->spelling()==source_) {
        e->replace_lhs(
            make_expression<NumberExpression>(lhs->location(), value_)
        );
    }
    else if(!lhs) {
        e->lhs()->accept(this);
    }

    auto rhs = e->rhs()->is_identifier();
    if(rhs && rhs->spelling()==source_) {
        e->replace_rhs(
            make_expression<NumberExpression>(rhs->location(), value_)
        );
    }
    else if(!rhs){
        e->rhs()->accept(this);
    }
}
