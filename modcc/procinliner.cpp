#include "procinliner.hpp"

#include <iostream>

#include "astmanip.hpp"
#include "error.hpp"
#include "errorvisitor.hpp"
#include "symdiff.hpp"

// Note: on renaming variables when inlining:
// Identifiers will refer to one of
// - LOCAL variables
// - argument,
// - global variable
// - PARAMETER
// - ASSIGNED
//
// All local variables are renamed and the mapping is stored in local_arg_map_.
// The mapping from arguments to local names is in call_args_map_.
//
// Local variable renaming of identifiers should be performed before call
// argument renaming. This means that if a local variable shadows an argument,
// the local variable takes precedence.

void check_errors(Expression* e) {
    ErrorVisitor v("");
    e->accept(&v);
    if(v.num_errors()) throw compiler_exception("something went wrong with inlined procedure call ", e->location());
}

ARB_LIBMODCC_API expression_ptr inline_procedure_calls(std::string caller, BlockExpression* block) {
    // The inliner will inline one procedure at a time. Once all procedures in a
    // block have been inlined, the while loop will be broken
    auto inline_block = block->clone();
    for(;;) {
        inline_block->semantic(block->scope());

        auto inliner = std::make_unique<ProcedureInliner>(caller);
        inline_block->accept(inliner.get());
        inline_block = inliner->as_block(false);
        if (inliner->state_ == ProcedureInliner::state::Done) break;
        break;
    }
    return inline_block;
}

// The inliner works on inlining one item at a time.
void ProcedureInliner::visit(Expression* e) {
    if (state_ == state::Running)
        throw compiler_exception("I don't know how to do procedure inlining for this statement: "
                                 + e->to_string(),
                                 e->location());
    statements_.push_back(e->clone());
}

// Only in procedures, always stays the same
void ProcedureInliner::visit(ConserveExpression *e) {
    statements_.push_back(e->clone());
}

// Only in procedures, always stays the same
void ProcedureInliner::visit(CompartmentExpression *e) {
    statements_.push_back(e->clone());
}

// Only in procedures, always stays the same
void ProcedureInliner::visit(LinearExpression *e) {
    statements_.push_back(e->clone());
}

void ProcedureInliner::visit(LocalDeclaration* e) {
    // If we are active, we need to rename variables
    if (state_ == state::Running) {
        std::map<std::string, Token> new_vars;
        for (auto& var: e->variables()) {
            auto unique_decl = make_unique_local_decl(scope_, e->location(), "r_");
            auto unique_name = unique_decl.id->is_identifier()->spelling();

            // Local variables must be renamed to avoid collisions with the caller.
            // The mappings are stored in local_arg_map
            local_arg_map_.emplace(std::make_pair(var.first, std::move(unique_decl.id)));

            auto e_tok = var.second;
            e_tok.spelling = unique_name;
            new_vars[unique_name] =  e_tok;
        }
        e->variables().swap(new_vars);
    }
    statements_.push_back(e->clone());
}

void ProcedureInliner::visit(UnaryExpression* e) {
    if (state_ != state::Running) return;

    auto sub = substitute(e->expression(), local_arg_map_);
    sub = substitute(sub, call_arg_map_);
    e->replace_expression(std::move(sub));
    e->semantic(scope_);
    check_errors(e);
}

void ProcedureInliner::visit(BinaryExpression* e) {
    if (state_ != state::Running) return;
    e->replace_lhs(substitute(substitute(e->lhs(), local_arg_map_), call_arg_map_));
    e->replace_rhs(substitute(substitute(e->rhs(), local_arg_map_), call_arg_map_));
    e->semantic(scope_);
    check_errors(e);
}


void ProcedureInliner::visit(AssignmentExpression* e) {
    // If we're inlining a call, take care of variable renaming
    if (state_ == state::Running) {
        if (e->lhs()->is_identifier()) {
            e->replace_lhs(substitute(e->lhs(), local_arg_map_));
        }
        if (e->rhs()->is_identifier()) {
            auto sub_rhs = substitute(e->rhs(), local_arg_map_);
            sub_rhs = substitute(sub_rhs, call_arg_map_);
            e->replace_rhs(std::move(sub_rhs));
        }
        else {
            e->rhs()->accept(this);
        }
    }
    statements_.push_back(e->clone());
}

void ProcedureInliner::visit(IfExpression* e) {
    expr_list_type outer;
    std::swap(outer, statements_);

    e->condition()->accept(this);
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

    statements_.clear();

    statements_ = std::move(outer);
    statements_.push_back(make_expression<IfExpression>(
            e->location(),
            e->condition()->clone(),
            std::move(true_branch),
            std::move(false_branch)));
}

void ProcedureInliner::visit(CallExpression* e) {
    if (state_ == state::Running) {
        if (e->is_procedure_call()) {
            auto nm = e->is_procedure_call()->name();
            if (nm == callee_ || nm == caller_) throw compiler_exception("recursive procedures not allowed", e->location());
        }
        auto& args = e->is_procedure_call()
                   ? e->is_procedure_call()->args()
                   : e->is_function_call()->args();
        for (auto& a: args) {
            if (a->is_identifier()) {
                a = substitute(substitute(a, local_arg_map_), call_arg_map_);
            }
            else {
                a->accept(this);
            }
        }
    }
    else if (state_ == state::Ready) {
        // If we are ready to do some inlining, check if we can indeed inline this statement
        if (auto call = e->is_procedure_call(); call != nullptr) {
            // fetch the procedure, its body and its formal args
            const auto& proc = call->procedure();
            const auto& body = proc->body()->clone();
            const auto& args = proc->args();

            // store the args we are actually called with to do replacement
            const auto& subs = call->args();
            for (unsigned i = 0; i < args.size(); ++i) {
                call_arg_map_.emplace(
                    std::make_pair(args[i]->is_argument()->spelling(),
                                   subs[i]->clone()));
            }

            scope_  = e->scope();
            callee_ = proc->name();
            state_  = state::Running;

            for (auto& s: body->is_block()->statements()) s->semantic(scope_);
            body->accept(this);

            state_ = state::Done;
        }
    }
    else if (e->is_procedure_call()) {
        statements_.push_back(e->clone());
    }
}
