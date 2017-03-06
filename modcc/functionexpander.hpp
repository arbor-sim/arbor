#pragma once

#include <sstream>

#include "expression.hpp"
#include "scope.hpp"
#include "visitor.hpp"

// Make a local declaration and assignment for the given expression,
// and insert at the front and back respectively of the statement list.
// Return the new unique local identifier.
expression_ptr insert_unique_local_assignment(expr_list_type& stmts, Expression* e);

// prototype for lowering function calls
expr_list_type lower_function_calls(Expression* e);

///////////////////////////////////////////////////////////////////////////////
// visitor that takes function call sites and lowers them to inline assignments
//
// e.g. if called on the following statement
//
// a = 3 + foo(x, y)
//
// the calls_ member will be
//
// LOCAL ll0_
// ll0_ = foo(x,y)
//
// and the original statment is modified to be
//
// a = 3 + ll0_
//
// If the calls_ data is spliced directly before the original statement
// the function call will have been fully lowered
///////////////////////////////////////////////////////////////////////////////
class FunctionCallLowerer : public Visitor {
public:
    FunctionCallLowerer(scope_ptr s)
    :   scope_(s)
    {}

    void visit(CallExpression *e)       override;
    void visit(Expression *e)           override;
    void visit(UnaryExpression *e)      override;
    void visit(BinaryExpression *e)     override;
    void visit(NumberExpression *e)     override {};
    void visit(IdentifierExpression *e) override {};

    expr_list_type& calls() {
        return calls_;
    }

    expr_list_type move_calls() {
        return std::move(calls_);
    }

    ~FunctionCallLowerer() {}

private:
    template< typename F>
    void expand_call(CallExpression* func, F replacer) {
        auto id = insert_unique_local_assignment(calls_, func);
        // replace the function call in the original expression with the local
        // variable which holds the pre-computed value
        replacer(std::move(id));
    }

    expr_list_type calls_;
    scope_ptr scope_;
};

///////////////////////////////////////////////////////////////////////////////
// visitor that takes function arguments that are not literals of identifiers
// and lowers them to inline assignments
//
// e.g. if called on the following statement
//
// a = foo(2+x, y)
//
// the calls_ member will be
//
// LOCAL ll0_
// ll0_ = 2+x
//
// and the original statment is modified to be
//
// a = foo(ll0_, y)
//
// If the calls_ data is spliced directly before the original statement
// the function arguments will have been fully lowered
///////////////////////////////////////////////////////////////////////////////
expr_list_type lower_function_arguments(std::vector<expression_ptr>& args);

