#include <iostream>
#include <map>
#include <string>
#include <list>

#include "astmanip.hpp"
#include "symdiff.hpp"
#include "visitor.hpp"

class LinearRewriter : public BlockRewriterBase {
public:
    using BlockRewriterBase::visit;

    LinearRewriter(std::vector<std::string> st_vars): state_vars(st_vars) {}
    LinearRewriter(scope_ptr enclosing_scope): BlockRewriterBase(enclosing_scope) {}

    virtual void visit(LinearExpression *e) override;

protected:
    virtual void reset() override {
        BlockRewriterBase::reset();
    }

private:
    std::vector<std::string> state_vars;
};

expression_ptr linear_rewrite(BlockExpression* block, std::vector<std::string> state_vars) {
    LinearRewriter visitor(state_vars);
    block->accept(&visitor);
    return visitor.as_block(false);
}

// LinearRewriter implementation follows.

// Factorize the linear expression in terms of the state variables and place
// the resulting sum of products on the lhs. Place everything else on the rhs
void LinearRewriter::visit(LinearExpression* e) {
    Location loc = e->location();
    scope_ptr scope = e->scope();

    expression_ptr lhs;
    for (const auto& state : state_vars) {
        // To factorize w.r.t state, differentiate the lhs and rhs
        auto ident = make_expression<IdentifierExpression>(loc, state);
        auto coeff = constant_simplify(make_expression<SubBinaryExpression>(loc,
                symbolic_pdiff(e->lhs(), state),
                symbolic_pdiff(e->rhs(), state)));

        if (expr_value(coeff) != 0) {
            auto local_coeff = make_unique_local_assign(scope, coeff, "l_");
            statements_.push_back(std::move(local_coeff.local_decl));
            statements_.push_back(std::move(local_coeff.assignment));

            auto pair = make_expression<MulBinaryExpression>(loc, std::move(local_coeff.id), std::move(ident));

            // Construct the lhs of the new linear expression
            if (!lhs) {
                lhs = std::move(pair);
            } else {
                lhs = make_expression<AddBinaryExpression>(loc, std::move(lhs), std::move(pair));
            }
        }
    }

    // To find the rhs of the new linear expression, simplify the old
    // linear expression with state variables set to zero
    auto rhs_0 = e->lhs()->clone();
    auto rhs_1 = e->rhs()->clone();

    for (auto state: state_vars) {
        auto zero_expr = make_expression<NumberExpression>(loc, 0.0);
        rhs_0 = substitute(rhs_0, state,  zero_expr);
        rhs_1 = substitute(rhs_1, state,  zero_expr);
    }
    rhs_0 = constant_simplify(rhs_0);
    rhs_1 = constant_simplify(rhs_1);

    auto rhs = constant_simplify(make_expression<SubBinaryExpression>(loc, std::move(rhs_1), std::move(rhs_0)));

    auto local_rhs = make_unique_local_assign(scope, rhs, "l_");
    statements_.push_back(std::move(local_rhs.local_decl));
    statements_.push_back(std::move(local_rhs.assignment));

    rhs = std::move(local_rhs.id);

    statements_.push_back(make_expression<LinearExpression>(loc, std::move(lhs), std::move(rhs)));
}
