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
expression_ptr lower_function_calls(BlockExpression* block);

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
//   virtual void visit(IfExpression *e)          override;
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

    void lower_call_arguments(std::vector<expression_ptr>& args) {
        for(auto it=args.begin(); it!=args.end(); ++it) {
            // get reference to the unique_ptr with the expression
            auto& e = *it;
#ifdef LOGGING
            std::cout << "inspecting argument @ " << e->location() << " : " << e->to_string() << std::endl;
#endif

            if(e->is_number() || e->is_identifier()) {
                // do nothing, because identifiers and literals are in the correct form
                // for lowering
                continue;
            }

            auto id = insert_unique_local_assignment(statements_, e.get());
#ifdef LOGGING
            std::cout << "  lowering to " << new_statements.back()->to_string() << "\n";
#endif
            // replace the function call in the original expression with the local
            // variable which holds the pre-computed value
            std::swap(e, id);
        }
#ifdef LOGGING
        std::cout << "\n";
#endif
    }

protected:
    virtual void reset() override {
        BlockRewriterBase::reset();
    }
};

