#pragma once

#include <sstream>

#include "expression.hpp"
#include "scope.hpp"
#include "visitor.hpp"

// Make a local declaration and assignment for the given expression,
// and insert at the front and back respectively of the statement list.
// Return the new unique local identifier.
expression_ptr insert_unique_local_assignment(expr_list_type& stmts, Expression* e);

// prototype for lowering function calls and arguments
expression_ptr lower_functions(BlockExpression* block);

class FunctionCallLowerer : public BlockRewriterBase {
public:
    using BlockRewriterBase::visit;

    FunctionCallLowerer(): BlockRewriterBase() {}
    FunctionCallLowerer(scope_ptr s): BlockRewriterBase(s) {}

    virtual void visit(CallExpression *e)        override;
    virtual void visit(ConserveExpression *e)    override;
    virtual void visit(CompartmentExpression *e) override;
    virtual void visit(LinearExpression *e)      override;
    virtual void visit(AssignmentExpression *e)  override;
    virtual void visit(BinaryExpression *e)      override;
    virtual void visit(UnaryExpression *e)       override;
    virtual void visit(IfExpression *e)          override;
    virtual void visit(NumberExpression *e)      override {};
    virtual void visit(IdentifierExpression *e)  override {};

private:
    template< typename F>
    void expand_call(CallExpression* func, F replacer) {
        auto id = insert_unique_local_assignment(statements_, func);
        // replace the function call in the original expression with the local
        // variable which holds the pre-computed value
        replacer(std::move(id));
    }
};

