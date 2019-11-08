#include <iostream>

#include "astmanip.hpp"
#include "error.hpp"
#include "functioninliner.hpp"
#include "errorvisitor.hpp"

// Do the inlining: supports multiline functions and if/else statements
// e.g. if the function foo in the examples above is defined as follows
//
//  function foo(a, b, c) {
//      Local t = b + c
//      foo = a*t
//  }
//
// the full inlined example is
//      ll1_ = 2+x
//      r_0_ = y+1
//      ll0_ = ll1_*r_0_
//      a = 2 + ll0_
expression_ptr inline_function_call(const expression_ptr& e)
{
    auto assign_to_func = e->is_assignment();
    auto ret_identifier = assign_to_func->lhs()->is_identifier();

    if(auto f = assign_to_func->rhs()->is_function_call()) {
        auto body = f->function()->body()->clone();

        for (auto&s: body->is_block()->statements()) {
            s->semantic(e->scope());
        }

        std::cout << "in progress -- \n" << body->to_string() << std::endl << std::endl;
        FunctionInliner func_inliner(f->name(), ret_identifier, f->function()->args(), f->args(), e->scope());
        body->accept(&func_inliner);

        for (auto& var: func_inliner.local_arg_map_) {
            std::cout << "++ local [" << var.first << ", " << var.second->to_string() << "]" << std::endl;
        }
        for (auto& var: func_inliner.call_arg_map_) {
            std::cout << "++ call  [" << var.first << ", " << var.second->to_string() << "]" << std::endl;
        }
        std::cout << body->to_string() << "\n\n-- in progress\n" << std::endl;

        if (!func_inliner.return_val_set()) {
            throw compiler_exception(pprintf("return variable of function % not set", f->name()), e->location());
        }
        return body;
    }
    return {};
}
///////////////////////////////////////////////////////////////////////////////
//  function inliner
///////////////////////////////////////////////////////////////////////////////

// Takes a Binary or Unary Expression and replaces its variables that match any
// function argument with the mappings in call_arg_map and local_arg_map
void FunctionInliner::replace_args(Expression* e) {
    auto map_variables = [&](auto& map) {
        for(auto& el: map) {
            if(auto id = el.second->is_identifier()) {
                VariableReplacer v(el.first, id->spelling());
                e->accept(&v);
            }
            else if(auto value = el.second->is_number()) {
                ValueInliner v(el.first, value->value());
                e->accept(&v);
            }
        }
    };

    map_variables(local_arg_map_);
    map_variables(call_arg_map_);

    e->semantic(scope_);

    ErrorVisitor v("");
    e->accept(&v);
    if(v.num_errors()) {
        throw compiler_exception("something went wrong with inlined function call ", e->location());
    }
}

void FunctionInliner::visit(Expression* e) {
    throw compiler_exception(
            "I don't know how to do function inlining for this statement : "
            + e->to_string(), e->location());
}

void FunctionInliner::visit(LocalDeclaration* e) {
    auto loc = e->location();

    std::map<std::string, Token> new_vars;
    for (auto& var: e->variables()) {
        auto unique_decl = make_unique_local_decl(scope_, loc, "r_");
        auto unique_name = unique_decl.id->is_identifier()->spelling();

        // Local variables must be renamed to avoid collisions with the calling function.
        // The mappings are stored in local_arg_map
        local_arg_map_.insert({var.first, std::move(unique_decl.id)});

//        std::cout << "++ local [" << var.first << ", " << local_arg_map_[var.first]->to_string() << "]" << std::endl;

        auto e_tok = var.second;
        e_tok.spelling = unique_name;
        new_vars[unique_name] =  e_tok;
    }
    e->variables().swap(new_vars);
}

void FunctionInliner::visit(BlockExpression* e) {
    for (auto& expr: e->statements()) {
        expr->accept(this);
    }
}

void FunctionInliner::visit(UnaryExpression* e) {
    replace_args(e);
}

void FunctionInliner::visit(BinaryExpression* e) {
    replace_args(e);
}

void FunctionInliner::visit(AssignmentExpression* e) {
//    std::cout << "\\\\ " << e->to_string() << std::flush;
//    std::cout << std::endl;
    if (auto lhs = e->lhs()->is_identifier()) {
        if (lhs->spelling() == func_name_) {
            e->replace_lhs(lhs_->clone());
            return_set_ = true;
        } else {
            if (local_arg_map_.count(lhs->spelling())) {
                e->replace_lhs(local_arg_map_.at(lhs->spelling())->clone());
            }
            if (call_arg_map_.count(lhs->spelling())) {
                e->replace_lhs(call_arg_map_.at(lhs->spelling())->clone());
            }
        }
    }

    if (auto rhs = e->rhs()->is_identifier()) {
        if (local_arg_map_.count(rhs->spelling())) {
            e->replace_rhs(local_arg_map_.at(rhs->spelling())->clone());
        }
        if (call_arg_map_.count(rhs->spelling())) {
            e->replace_rhs(call_arg_map_.at(rhs->spelling())->clone());
        }
    }
    else {
        e->rhs()->accept(this);
    }
//    std::cout << "// " << e->to_string() << std::endl;
}

void FunctionInliner::visit(IfExpression* e) {
    bool if_ret;
    bool save_ret = return_set_;

    return_set_ = false;

    e->condition()->accept(this);
    e->true_branch()->accept(this);

    if_ret = return_set_;
    return_set_ = false;

    if (e->false_branch()) {
        e->false_branch()->accept(this);
    }

    if_ret &= return_set_;

    return_set_ = save_ret? save_ret: if_ret;
}

void FunctionInliner::visit(CallExpression* e) {
    for (auto& a: e->is_function_call()->args()) {
        if (auto id = a->is_identifier()) {
            if (local_arg_map_.count(id->spelling())) {
                a = local_arg_map_.at(id->spelling())->clone();
            }
            if (call_arg_map_.count(id->spelling())) {
                a = call_arg_map_.at(id->spelling())->clone();
            }
        } else {
            a->accept(this);
        }
    }
    /*
    for (auto& a: e->is_function_call()->args()) {
        for (unsigned i = 0;  i < fargs_.size(); i++) {
            if (auto id = a->is_identifier()) {
                if (fargs_[i] == id->spelling()) {
                    a = cargs_[i]->clone();
                }
            } else {
                a->accept(this);
            }
        }
    }
     */
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
