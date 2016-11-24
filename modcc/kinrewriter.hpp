#pragma once

#include <iostream>
#include <map>
#include <string>
#include <list>

#include "astmanip.hpp"
#include "visitor.hpp"

using stmt_list_type = std::list<expression_ptr>;

class KineticRewriter : public Visitor {
public:
    virtual void visit(Expression *) override;

    virtual void visit(UnaryExpression *e) override { visit((Expression*)e); }
    virtual void visit(BinaryExpression *e) override { visit((Expression*)e); }

    virtual void visit(ConserveExpression *e) override;
    virtual void visit(ReactionExpression *e) override;
    virtual void visit(BlockExpression *e) override;
    virtual void visit(ProcedureExpression* e) override;

    symbol_ptr as_procedure() {
        stmt_list_type body_stmts;
        for (const auto& s: statements) body_stmts.push_back(s->clone());

        auto body = make_expression<BlockExpression>(
            proc_loc,
            std::move(body_stmts),
            false);

        return make_symbol<ProcedureExpression>(
            proc_loc,
            proc_name,
            std::vector<expression_ptr>(),
            std::move(body));
    }

private:
    // Name and location of original kinetic procedure (used for `as_procedure` above).
    std::string proc_name;
    Location proc_loc;

    // Statements in replacement procedure body.
    stmt_list_type statements;

    // Acccumulated terms for derivative expressions, keyed by id name.
    std::map<std::string, expression_ptr> dterms;

    // Reset state (at e.g. start of kinetic proc).
    void reset() {
        proc_name = "";
        statements.clear();
        dterms.clear();
    }
};

// By default, copy statements across verbatim.
inline void KineticRewriter::visit(Expression* e) {
    statements.push_back(e->clone());
}

inline void KineticRewriter::visit(ConserveExpression*) {
    // Deliberately ignoring these for now!
}

inline void KineticRewriter::visit(ReactionExpression* e) {
    Location loc = e->location();
    scope_ptr scope = e->scope();

    // Total forward rate is the specified forward reaction rate constant, multiplied
    // by the concentrations of species present in the left hand side.

    auto fwd = e->fwd_rate()->clone();
    auto lhs = e->lhs()->is_stoich();
    for (const auto& term: lhs->terms()) {
        auto& id = term->is_stoich_term()->ident();
        auto& coeff = term->is_stoich_term()->coeff();

        fwd = make_expression<MulBinaryExpression>(
            loc,
            make_expression<PowBinaryExpression>(loc, id->clone(), coeff->clone()),
            std::move(fwd));
    }

    // Similar for reverse rate.

    auto rev = e->rev_rate()->clone();
    auto rhs = e->rhs()->is_stoich();
    for (const auto& term: rhs->terms()) {
        auto& id = term->is_stoich_term()->ident();
        auto& coeff = term->is_stoich_term()->coeff();

        rev = make_expression<MulBinaryExpression>(
            loc,
            make_expression<PowBinaryExpression>(loc, id->clone(), coeff->clone()),
            std::move(rev));
    }

    auto net_rate = make_expression<SubBinaryExpression>(
            loc,
            std::move(fwd), std::move(rev));
    net_rate->semantic(scope);

    auto local_net_rate = make_unique_local_assign(scope, net_rate, "rate");
    statements.push_back(std::move(local_net_rate.local_decl));
    statements.push_back(std::move(local_net_rate.assignment));
    scope = local_net_rate.scope; // nop for now...

    auto net_rate_sym = std::move(local_net_rate.id);

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
        statements.push_back(std::move(local_term.local_decl));
        statements.push_back(std::move(local_term.assignment));
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

inline void KineticRewriter::visit(ProcedureExpression* e) {
    reset();
    proc_name = e->name();
    proc_loc = e->location();
    e->body()->accept(this);

    // make new procedure from saved statements and terms
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

        assign->scope(scope); // don't re-do semantic analysis here
        statements.push_back(std::move(assign));
    }
}

inline void KineticRewriter::visit(BlockExpression* e) {
    for (auto& s: e->statements()) {
        s->accept(this);
    }
}
