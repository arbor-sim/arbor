#include <iostream>

#include "astmanip.hpp"
#include "error.hpp"
#include "functioninliner.hpp"
#include "errorvisitor.hpp"
expression_ptr inline_function_calls(std::string calling_func, BlockExpression* block) {
    auto inline_block = block->clone();

    // The function inliner will inline one function at a time
    // Once all functions in a block have been inlined, the
    // while loop will be broken
    while(true) {
        inline_block->semantic(block->scope());

        auto func_inliner = std::make_unique<FunctionInliner>(calling_func);
        inline_block->accept(func_inliner.get());

        if (!func_inliner->return_val_set()) {
            throw compiler_exception(pprintf("return variable of function not set", block->location()));
        }

        if (func_inliner->finished_inlining()) {
            return func_inliner->as_block(false);
        }

        inline_block = func_inliner->as_block(false);
    }
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

// The Function inliner works on inlining one function at a time.
// If no function is being inlined when an expression is being visited,
// the expression remains the same.
void FunctionInliner::visit(Expression* e) {
    if (!inlining_in_progress_) {
        statements_.push_back(e->clone());
        return;
    }
    throw compiler_exception(
            "I don't know how to do function inlining for this statement : "
            + e->to_string(), e->location());
}

// Only in procedures, always stays the same
void FunctionInliner::visit(ConserveExpression *e) {
    statements_.push_back(e->clone());
}

// Only in procedures, always stays the same
void FunctionInliner::visit(CompartmentExpression *e) {
    statements_.push_back(e->clone());
}

// Only in procedures, always stays the same
void FunctionInliner::visit(LinearExpression *e) {
    statements_.push_back(e->clone());
}

void FunctionInliner::visit(LocalDeclaration* e) {
    if (!inlining_in_progress_) {
        statements_.push_back(e->clone());
        return;
    }

    std::map<std::string, Token> new_vars;
    for (auto& var: e->variables()) {
        auto unique_decl = make_unique_local_decl(scope_, e->location(), "r_");
        auto unique_name = unique_decl.id->is_identifier()->spelling();

        // Local variables must be renamed to avoid collisions with the calling function.
        // The mappings are stored in local_arg_map
        local_arg_map_.emplace(std::make_pair(var.first, std::move(unique_decl.id)));

        auto e_tok = var.second;
        e_tok.spelling = unique_name;
        new_vars[unique_name] =  e_tok;
    }
    e->variables().swap(new_vars);
    statements_.push_back(e->clone());

}

void FunctionInliner::visit(UnaryExpression* e) {
    if (!inlining_in_progress_) {
        return;
    }
    replace_args(e);
}

void FunctionInliner::visit(BinaryExpression* e) {
    if (!inlining_in_progress_) {
        return;
    }
    replace_args(e);
}

void FunctionInliner::visit(AssignmentExpression* e) {
    // At this point, after function lowering, all function calls should be on the rhs of
    // an Assignment Expression.
    // If we find a new function to inline, we can do so, provided we aren't already inlining
    // another function and we haven't inlined a function already.
    if (!inlining_in_progress_ && !inlining_executed_ && e->rhs()->is_function_call()) {
        auto f = e->rhs()->is_function_call();
        auto& fargs = f->function()->args();
        auto& cargs = f->args();

        inlining_in_progress_ = true;
        inlining_func_ = f->name();
        lhs_ = e->lhs()->is_identifier()->clone();
        return_set_ = false;
        scope_ = e->scope();

        for (unsigned i = 0; i < fargs.size(); ++i) {
            call_arg_map_.emplace(std::make_pair(fargs[i]->is_argument()->spelling(), cargs[i]->clone()));
        }

        auto body = f->function()->body()->clone();
        for (auto&s: body->is_block()->statements()) {
            s->semantic(e->scope());
        }

        body->accept(this);
        inlining_in_progress_ = false;
        inlining_executed_ = true;
        return;
    }

    // If we're not inlining a function call, don't change anything in the expression
    if (!inlining_in_progress_) {
        statements_.push_back(e->clone());
        return;
    }

    // If we're inlining a function call, take care of variable renaming
    if (auto lhs = e->lhs()->is_identifier()) {
        std::string iden_name = lhs->spelling();

        // if the identifier name matches the function name, then we are setting the return value
        if (iden_name == inlining_func_) {
            e->replace_lhs(lhs_->clone());
            return_set_ = true;
        } else {
            if (local_arg_map_.count(iden_name)) {
                e->replace_lhs(local_arg_map_.at(iden_name)->clone());
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
    statements_.push_back(e->clone());
}

void FunctionInliner::visit(IfExpression* e) {
    expr_list_type outer;
    std::swap(outer, statements_);

    // Make sure if Expressions set the return value if not already set
    // return_set_ will always be true unless we are inlining a function
    bool if_ret;
    bool save_ret = return_set_;

    return_set_ = false;

    e->condition()->accept(this);
    e->true_branch()->accept(this);
    auto true_branch = make_expression<BlockExpression>(
            e->true_branch()->location(),
            std::move(statements_),
            true);

    statements_.clear();
    if_ret = return_set_;
    return_set_ = false;

    expression_ptr false_branch;
    if (e->false_branch()) {
        e->false_branch()->accept(this);
        false_branch = make_expression<BlockExpression>(
                e->false_branch()->location(),
                std::move(statements_),
                true);
    }

    statements_.clear();
    if_ret &= return_set_;
    return_set_ = save_ret? save_ret: if_ret;

    statements_ = std::move(outer);
    statements_.push_back(make_expression<IfExpression>(
            e->location(),
            e->condition()->clone(),
            std::move(true_branch),
            std::move(false_branch)));
}

void FunctionInliner::visit(CallExpression* e) {
    if (!inlining_in_progress_) {
        if (e->is_procedure_call()) {
            statements_.push_back(e->clone());
        }
        return;
    }

    if (e->is_function_call()->name() == inlining_func_ || e->is_function_call()->name() == calling_func_) {
        throw compiler_exception(pprintf("Recursive functions not allowed", e->location()));
    }

    auto& args = e->is_function_call() ? e->is_function_call()->args() : e->is_procedure_call()->args();

    for (auto& a: args) {
        if (auto id = a->is_identifier()) {
            std::string iden_name = id->spelling();
            if (local_arg_map_.count(iden_name)) {
                a = local_arg_map_.at(iden_name)->clone();
            }
            if (call_arg_map_.count(iden_name)) {
                a = call_arg_map_.at(iden_name)->clone();
            }
        } else {
            a->accept(this);
        }
    }
    if (e->is_procedure_call()) {
        statements_.push_back(e->clone());
        return;
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
