#pragma once

#include <sstream>

#include "scope.hpp"
#include "visitor.hpp"
#include <libmodcc/export.hpp>

ARB_LIBMODCC_API expression_ptr inline_procedure_calls(std::string caller, BlockExpression* block);

struct ARB_LIBMODCC_API ProcedureInliner:
    public BlockRewriterBase
{
    enum class state {
       Done,
       Running,
       Ready,
    };

    using BlockRewriterBase::visit;
    ProcedureInliner(std::string caller) : BlockRewriterBase(), caller_(caller) {};
    ProcedureInliner(scope_ptr s): BlockRewriterBase(s) {}

    virtual void visit(Expression *e)            override;
    virtual void visit(CallExpression *e)        override;
    virtual void visit(ConserveExpression *e)    override;
    virtual void visit(CompartmentExpression *e) override;
    virtual void visit(LinearExpression *e)      override;
    virtual void visit(AssignmentExpression* e)  override;
    virtual void visit(BinaryExpression* e)      override;
    virtual void visit(UnaryExpression* e)       override;
    virtual void visit(IfExpression* e)          override;
    virtual void visit(LocalDeclaration* e)      override;
    virtual void visit(NumberExpression* e)      override {};
    virtual void visit(IdentifierExpression* e)  override {};

    std::string callee_, caller_;
    expression_ptr lhs_;
    std::map<std::string, expression_ptr> call_arg_map_;
    std::map<std::string, expression_ptr> local_arg_map_;
    scope_ptr scope_;

    state state_ = state::Ready;

protected:
    virtual void reset() override {
        state_ = state::Ready;
        callee_.clear();
        lhs_ = nullptr;
        call_arg_map_.clear();
        local_arg_map_.clear();
        scope_.reset();
        BlockRewriterBase::reset();
    }
};
