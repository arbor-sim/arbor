#include <iostream>

#include "error.hpp"
#include "functioninliner.hpp"
#include "errorvisitor.hpp"

void replace_and_inline(expression_ptr& exp,
                        const expression_ptr& lhs,
                        const scope_ptr& scope,
                        const std::vector<expression_ptr>& fargs,
                        const std::vector<expression_ptr>& cargs) {
    std::cout << "In " << exp->to_string() << std::endl;


    auto fix_expression = [&fargs, &cargs, &scope, &lhs](expression_ptr& e) {

        const auto& to_fix =  e->is_assignment() ? e->is_assignment()->rhs() : e->is_if()->condition()->is_conditional();

        for(auto i=0u; i<fargs.size(); ++i) {
            if(auto id = cargs[i]->is_identifier()) {
#ifdef LOGGING
                std::cout << "inline_function_call symbol replacement "
                          << id->to_string() << " -> " << fargs[i]->to_string()
                          << " in the expression " << exp->to_string() << "\n";

#endif
                VariableReplacer v(
                        fargs[i]->is_argument()->spelling(),
                        id->spelling()
                );
                to_fix->accept(&v);
            }
            else if(auto value = cargs[i]->is_number()) {
#ifdef LOGGING
                std::cout << "inline_function_call symbol replacement "
                          << value->to_string() << " -> " << fargs[i]->to_string()
                          << " in the expression " << exp->to_string() << "\n";
#endif
                ValueInliner v(
                        fargs[i]->is_argument()->spelling(),
                        value->value()
                );
                to_fix->accept(&v);
            }
            else {
                throw compiler_exception(
                        "can't inline functions with expressions as arguments",
                        e->location()
                );
            }
        }


        to_fix->semantic(scope);

        ErrorVisitor v("");
        to_fix->accept(&v);

        if(v.num_errors()) {
            throw compiler_exception("something went wrong with inlined function call ",
                                     e->location());
        }

#ifdef LOGGING
        std::cout << "inline_function_call result " << exp->to_string() << "\n\n";
#endif
        if (e->is_assignment()) {
            e->is_assignment()->replace_lhs(lhs->clone());
        }
        std::cout << "\tCHEERIO "<< e->to_string() << std::endl;
    };

    if (exp->is_if()) {
        if (!exp->is_if()->false_branch()) {
            throw compiler_exception(
                    "can only inline if statements with associated else", exp->location()
            );
        }

        auto& true_branch = exp->is_if()->true_branch()->is_block()->statements();
        auto& false_branch = exp->is_if()->false_branch()->is_block()->statements();

        if (true_branch.size() != 1 || false_branch.size() != 1) {
            throw compiler_exception(
                    "can only inline functions with one statement", exp->location()
            );
        }

        fix_expression(exp);

        if (true_branch.front()->is_if()) {
            replace_and_inline(true_branch.front(), lhs, scope, fargs, cargs);
        } else if (true_branch.front()->is_assignment()) {
            fix_expression(true_branch.front());
        } else {
            throw compiler_exception(
                    "can only inline assignment expressions and if expressions containing single assignment expressions", exp->location()
            );
        }

        if (false_branch.front()->is_if()) {
            replace_and_inline(false_branch.front(), lhs, scope, fargs, cargs);
        } else if (false_branch.front()->is_assignment()) {
            fix_expression(false_branch.front());
        } else {
            throw compiler_exception(
                    "can only inline assignment expressions and if expressions containing single assignment expressions", exp->location()
            );
        }
    } else if (exp->is_assignment()) {
        fix_expression(exp);
    } else {
        throw compiler_exception(
                "can only inline assignment expressions and if expressions containing single assignment expressions", exp->location()
        );
    }
    std::cout << "Out " << exp->to_string() << std::endl;


};

void inline_function_call(expression_ptr& e)
{
    auto assign_to_func = e->is_assignment();

    if(auto f = assign_to_func->rhs()->is_function_call()) {
        auto &body = f->function()->body()->statements();
        if (body.size() != 1) {
            throw compiler_exception(
                    "can only inline functions with one statement", f->function()->location()
            );
        }
        auto body_stmt = body.front()->clone();
        replace_and_inline(body_stmt, assign_to_func->lhs()->is_identifier()->clone(), e->scope(), f->function()->args(), f->args());
    }
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
