#include <iostream>

#include "error.hpp"
#include "functioninliner.hpp"
#include "errorvisitor.hpp"

expression_ptr inline_function_call(Expression* e)
{
    auto assign_to_func = e->is_assignment();
    if(auto f = assign_to_func->rhs()->is_function_call()) {
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

        expression_ptr stmt, true_exp, false_exp, cond_exp;
        if(body.front()->is_if()) {
            stmt = body.front()->is_if()->clone();

            if (!stmt->is_if()->false_branch()) {
                throw compiler_exception(
                        "can only inline if statements with associated else", func->location()
                );
            }

            auto& true_branch = stmt->is_if()->true_branch()->is_block()->statements();
            auto& false_branch = stmt->is_if()->false_branch()->is_block()->statements();

            if (true_branch.size() != 1 || false_branch.size() != 1) {
                throw compiler_exception(
                        "can only inline functions with one statement", func->location()
                );
            }

            if (!true_branch.front()->is_assignment() || !false_branch.front()->is_assignment()) {
                throw compiler_exception(
                        "can only inline if statements containing assignments", func->location()
                );
            }

            true_exp = true_branch.front()->is_assignment()->rhs()->clone();
            false_exp = false_branch.front()->is_assignment()->rhs()->clone();
            cond_exp = stmt->is_if()->condition()->is_conditional()->clone();

        } else {
            stmt = body.front()->is_assignment()->clone();
            true_exp = body.front()->is_assignment()->rhs()->clone();
        }

        auto& fargs = func->args(); // argument names for the function
        auto& cargs = f->args();    // arguments at the call site
        for(auto i=0u; i<fargs.size(); ++i) {
            if(auto id = cargs[i]->is_identifier()) {
#ifdef LOGGING
                std::cout << "inline_function_call symbol replacement "
                          << id->to_string() << " -> " << fargs[i]->to_string()
                          << " in the expressions: \n" << true_exp->to_string() << "\n";
                if (false_exp) {
                    std::cout << false_exp->to_string() << "\n";
                }
                if (cond_exp) {
                    std::cout << cond_exp->to_string() << "\n";
                }
#endif
                VariableReplacer v(
                    fargs[i]->is_argument()->spelling(),
                    id->spelling()
                );
                true_exp->accept(&v);
                if (false_exp) {
                    false_exp->accept(&v);
                }
                if (cond_exp) {
                    cond_exp->accept(&v);
                }
            }
            else if(auto value = cargs[i]->is_number()) {
#ifdef LOGGING
                std::cout << "inline_function_call symbol replacement "
                          << value->to_string() << " -> " << fargs[i]->to_string()
                          << " in the expressions: \n" << true_exp->to_string() << "\n";
                if (false_exp) {
                    std::cout << false_exp->to_string() << "\n";
                }
                if (cond_exp) {
                    std::cout << cond_exp->to_string() << "\n";
                }
#endif
                ValueInliner v(
                    fargs[i]->is_argument()->spelling(),
                    value->value()
                );
                true_exp->accept(&v);
                if (false_exp) {
                    false_exp->accept(&v);
                }
                if (cond_exp) {
                    cond_exp->accept(&v);
                }
            }
            else {
                throw compiler_exception(
                    "can't inline functions with expressions as arguments",
                     e->location()
                 );
            }
        }
        true_exp->semantic(e->scope());
        if (false_exp) {
            false_exp->semantic(e->scope());
        }
        if (cond_exp) {
            cond_exp->semantic(e->scope());
        }

        ErrorVisitor v("");
        true_exp->accept(&v);
        if (false_exp) {
            false_exp->accept(&v);
        }
        if (cond_exp) {
            cond_exp->accept(&v);
        }

        if(v.num_errors()) {
            throw compiler_exception("something went wrong with inlined function call ",
                                     e->location());
        }

        expression_ptr new_exp;
        if (stmt->is_assignment()) {
            new_exp = make_expression<AssignmentExpression>(stmt->location(), assign_to_func->lhs()->clone(), std::move(true_exp));
        } else if (stmt->is_if()) {
            auto true_assign = make_expression<AssignmentExpression>(true_exp->location(), assign_to_func->lhs()->clone(), std::move(true_exp));
            auto false_assign = make_expression<AssignmentExpression>(false_exp->location(), assign_to_func->lhs()->clone(), std::move(false_exp));

            new_exp = make_expression<IfExpression>(stmt->location(), std::move(cond_exp), std::move(true_assign), std::move(false_assign));
        };

#ifdef LOGGING
        std::cout << "inline_function_call result " << new_exp->to_string() << "\n\n";
#endif
        return new_exp;
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
