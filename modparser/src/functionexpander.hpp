#pragma once

#include <sstream>

#include "scope.hpp"
#include "visitor.hpp"

// storage for a list of expressions
using call_list_type = std::list<expression_ptr>;

// prototype for lowering function calls
call_list_type lower_function_calls(Expression* e);

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
    using scope_type = Scope<Symbol>;

    FunctionCallLowerer(std::shared_ptr<scope_type> s)
    :   scope_(s)
    {}

    void visit(CallExpression *e)       override;
    void visit(Expression *e)           override;
    void visit(UnaryExpression *e)      override;
    void visit(BinaryExpression *e)     override;
    void visit(NumberExpression *e)     override {};
    void visit(IdentifierExpression *e) override {};

    call_list_type& calls() {
        return calls_;
    }

    call_list_type move_calls() {
        return std::move(calls_);
    }

    ~FunctionCallLowerer() {}

private:
    Symbol* make_unique_local() {
        std::string name;
        auto i = 0;
        do {
            name = pprintf("ll%_", i);
            ++i;
        } while(scope_->find(name));

        auto sym =
            scope_->add_local_symbol(
                name,
                make_symbol<LocalVariable>(
                    Location(), name, localVariableKind::local
                )
            );

        return sym;
    }

    template< typename F>
    void expand_call(CallExpression* func, F replacer) {
        // use the source location of the original statement
        auto loc = func->location();

        // make an identifier for the new symbol which will store the result of
        // the function call
        auto id = make_expression<IdentifierExpression>
            (loc, make_unique_local()->name());
        id->semantic(scope_);
        // generate a LOCAL declaration for the variable
        calls_.push_front(
            make_expression<LocalDeclaration>(loc, id->is_identifier()->spelling())
        );
        calls_.front()->semantic(scope_);

        // make a binary expression which assigns the function to the variable
        auto ass = binary_expression(loc, tok::eq, id->clone(), func->clone());
        ass->semantic(scope_);
        calls_.push_back(std::move(ass));

        // replace the function call in the original expression with the local
        // variable which holds the pre-computed value
        replacer(std::move(id));
    }

    call_list_type calls_;
    std::shared_ptr<scope_type> scope_;
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
call_list_type lower_function_arguments(std::vector<expression_ptr>& args);

