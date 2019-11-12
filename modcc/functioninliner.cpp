#include <iostream>

#include "astmanip.hpp"
#include "error.hpp"
#include "functioninliner.hpp"
#include "errorvisitor.hpp"
expression_ptr inline_function_calls(BlockExpression* block) {
    auto inline_block = block->clone();

    while(true) {
//        std::cout << "--before:\n" << inline_block->to_string() << std::endl;
        for (auto&s: inline_block->is_block()->statements()) {
            s->semantic(block->scope());
        }

        auto func_inliner = std::make_unique<FunctionInliner>();
        inline_block->accept(func_inliner.get());

        if (func_inliner->still_inlining()) {
            if (!func_inliner->return_val_set()) {
                throw compiler_exception(pprintf("return variable of function not set", block->location()));
            }
        } else {
            return func_inliner->as_block(false);
        }
        inline_block = func_inliner->as_block(false);
//        std::cout << "--after:\n" << inline_block->to_string() << std::endl;
    }
}

/*auto assign_to_func = e->is_assignment();
auto ret_identifier = assign_to_func->lhs()->is_identifier();

if(auto f = assign_to_func->rhs()->is_function_call()) {
    auto body = f->function()->body()->clone();

    for (auto&s: body->is_block()->statements()) {
        s->semantic(e->scope());
    }

    FunctionInliner func_inliner(f->name(), ret_identifier, f->function()->args(), f->args(), e->scope());
    body->accept(&func_inliner);

    if (!func_inliner.return_val_set()) {
        throw compiler_exception(pprintf("return variable of function % not set", f->name()), e->location());
    }
    return body;
}
return {};*/

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
    if (!processing_function_call_) {
        statements_.push_back(e->clone());
        return;
    }
    throw compiler_exception(
            "I don't know how to do function inlining for this statement : "
            + e->to_string(), e->location());
}
void FunctionInliner::visit(ConserveExpression *e) {
    statements_.push_back(e->clone());
}

void FunctionInliner::visit(CompartmentExpression *e) {
    statements_.push_back(e->clone());
}

void FunctionInliner::visit(LinearExpression *e) {
    statements_.push_back(e->clone());
}

void FunctionInliner::visit(LocalDeclaration* e) {
    if (!processing_function_call_) {
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
    if (!processing_function_call_) {
        return;
    }
    replace_args(e);
}

void FunctionInliner::visit(BinaryExpression* e) {
    if (!processing_function_call_) {
        return;
    }
    replace_args(e);
}

void FunctionInliner::visit(AssignmentExpression* e) {
    // If we're not already inlining a function and we have a function to inline,
    // Set up the visitor to inline this function call
    if (!processing_function_call_ && !inlined_func_ && e->rhs()->is_function_call()) {
        auto f = e->rhs()->is_function_call();
        auto& fargs = f->function()->args();
        auto& cargs = f->args();

        processing_function_call_ = true;
        func_name_ = f->name();
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
        processing_function_call_ = false;
        inlined_func_ = true;
        return;
    }

    // If we're not inlining a function call, don't change anything in the expression
    if (!processing_function_call_) {
        statements_.push_back(e->clone());
        return;
    }

    // If we're inlining a function call, take care of variable renaming
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
    statements_.push_back(e->clone());
}

void FunctionInliner::visit(IfExpression* e) {
    expr_list_type outer;
    std::swap(outer, statements_);

    bool if_ret;
    bool save_ret = return_set_;
    std::cout << " in " << func_name_ << " " << return_set_ << std::endl;

    return_set_ = false;

    e->condition()->accept(this);
    e->true_branch()->accept(this);
    auto true_branch = make_expression<BlockExpression>(
            e->true_branch()->location(),
            std::move(statements_),
            true);

    statements_.clear();

    if_ret = return_set_;
    std::cout << "after true " << return_set_ << std::endl;
    return_set_ = false;

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

    std::cout << "after false " << return_set_ << std::endl;
    if_ret &= return_set_;

    return_set_ = save_ret? save_ret: if_ret;
    std::cout << "end " << return_set_ << std::endl << std::endl;
}

void FunctionInliner::visit(CallExpression* e) {
    if (e->is_procedure_call()) {
        statements_.push_back(e->clone());
        return;
    }
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
