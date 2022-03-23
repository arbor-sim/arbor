#include <iostream>
#include <map>
#include <string>
#include <list>

#include "astmanip.hpp"
#include "symdiff.hpp"
#include "visitor.hpp"

class KineticRewriter : public BlockRewriterBase {
public:
    using BlockRewriterBase::visit;

    KineticRewriter() {}
    KineticRewriter(scope_ptr enclosing_scope): BlockRewriterBase(enclosing_scope) {}

    virtual void visit(ConserveExpression *e) override;
    virtual void visit(ReactionExpression *e) override;

protected:
    virtual void reset() override {
        BlockRewriterBase::reset();
        dterms.clear();
    }

    virtual void finalize() override;

private:
    // Acccumulated terms for derivative expressions, keyed by id name.
    std::map<std::string, expression_ptr> dterms;
};

ARB_LIBMODCC_API expression_ptr kinetic_rewrite(BlockExpression* block) {
    KineticRewriter visitor;
    block->accept(&visitor);
    return visitor.as_block(false);
}

// KineticRewriter implementation follows.

void KineticRewriter::visit(ConserveExpression* e) {
    statements_.push_back(e->clone());
}

void KineticRewriter::visit(ReactionExpression* e) {
    Location loc = e->location();
    scope_ptr scope = e->scope();

    // Total forward rate is the specified forward reaction rate constant, multiplied
    // by the concentrations of species present in the left hand side.

    auto fwd = e->fwd_rate()->clone();
    auto lhs = e->lhs()->is_stoich();
    for (const auto& term: lhs->terms()) {
        auto& id = term->is_stoich_term()->ident();
        auto& coeff = term->is_stoich_term()->coeff();

        fwd = constant_simplify(make_expression<MulBinaryExpression>(loc,
            make_expression<PowBinaryExpression>(loc, id->clone(), coeff->clone()),
            std::move(fwd)));
    }

    // Similar for reverse rate.

    auto rev = e->rev_rate()->clone();
    auto rhs = e->rhs()->is_stoich();
    for (const auto& term: rhs->terms()) {
        auto& id = term->is_stoich_term()->ident();
        auto& coeff = term->is_stoich_term()->coeff();

        rev = constant_simplify(make_expression<MulBinaryExpression>(loc,
                make_expression<PowBinaryExpression>(loc, id->clone(), coeff->clone()),
                std::move(rev)));
    }

    auto net_rate = make_expression<SubBinaryExpression>(
            loc,
            std::move(fwd), std::move(rev));
    net_rate->semantic(scope);

    auto local_net_rate = make_unique_local_assign(scope, net_rate, "rate");
    statements_.push_back(std::move(local_net_rate.local_decl));
    statements_.push_back(std::move(local_net_rate.assignment));
    scope = local_net_rate.scope; // nop for now...

    const auto& net_rate_sym = local_net_rate.id;

    // Net change in quantity after forward reaction:
    // e.g.  A + ... <-> 3A + ...
    // has a net delta of 2 for A.

    std::map<std::string, long long int> net_delta;

    for (const auto& term: lhs->terms()) {
        auto sterm = term->is_stoich_term();
        auto name = sterm->ident()->is_identifier()->name();
        net_delta[name] -= sterm->coeff()->is_integer()->integer_value();
    }

    for (const auto& term: rhs->terms()) {
        auto sterm = term->is_stoich_term();
        auto name = sterm->ident()->is_identifier()->name();
        net_delta[name] += sterm->coeff()->is_integer()->integer_value();
    }

    // Contribution to final ODE for each species is given by
    // net_rate * net_delta.

    for (auto& p: net_delta) {
        if (p.second==0) continue;

        auto term = make_expression<MulBinaryExpression>(
            loc,
            make_expression<IntegerExpression>(loc, p.second),
            net_rate_sym->clone());
        term->semantic(scope);

        auto local_term = make_unique_local_assign(scope, term, p.first+"_rate");
        statements_.push_back(std::move(local_term.local_decl));
        statements_.push_back(std::move(local_term.assignment));
        scope = local_term.scope; // nop for now...

        auto& dterm = dterms[p.first];
        if (!dterm) {
            dterm = std::move(local_term.id);
        }
        else {
            dterm = make_expression<AddBinaryExpression>(
                loc,
                std::move(dterm),
                std::move(local_term.id));

            // don't actually want to overwrite scope of previous terms
            // in dterm sum, so set expression 'scope' directly.
            dterm->scope(scope);
        }
    }
}

void KineticRewriter::finalize() {
    // append new derivative assignments from saved terms
    for (auto& p: dterms) {
        auto loc = p.second->location();
        auto scope = p.second->scope();

        auto deriv = make_expression<DerivativeExpression>(
            loc,
            p.first);
        deriv->semantic(scope);

        auto assign = make_expression<AssignmentExpression>(
            loc,
            std::move(deriv),
            std::move(p.second));

        statements_.push_back(std::move(assign));
    }
}

