#pragma once

// Transform derivative block into AST representing
// an integration step over the state variables, based on
// solver method.

#include <string>
#include <vector>

#include "expression.hpp"
#include "symdiff.hpp"
#include "symge.hpp"
#include "visitor.hpp"

expression_ptr remove_unused_locals(BlockExpression* block);

class SolverVisitorBase: public BlockRewriterBase {
protected:
    // list of identifier names appearing in derivatives on lhs
    std::vector<std::string> dvars_;

public:
    using BlockRewriterBase::visit;

    SolverVisitorBase() {}
    SolverVisitorBase(scope_ptr enclosing): BlockRewriterBase(enclosing) {}

    virtual std::vector<std::string> solved_identifiers() const {
        return dvars_;
    }

    virtual void reset() override {
        dvars_.clear();
        BlockRewriterBase::reset();
    }
};

class DirectSolverVisitor : public SolverVisitorBase {
public:
    using SolverVisitorBase::visit;

    DirectSolverVisitor() {}
    DirectSolverVisitor(scope_ptr enclosing): SolverVisitorBase(enclosing) {}

    virtual void visit(AssignmentExpression *e) override {
        // No solver method, so declare an error if lhs is a derivative.
        if(auto deriv = e->lhs()->is_derivative()) {
            error({"The DERIVATIVE block has a derivative expression"
                   " but no METHOD was specified in the SOLVE statement",
                   deriv->location()});
        }
        else {
            visit((Expression*)e);
        }
    }
};

class CnexpSolverVisitor : public SolverVisitorBase {
public:
    using SolverVisitorBase::visit;

    CnexpSolverVisitor() {}
    CnexpSolverVisitor(scope_ptr enclosing): SolverVisitorBase(enclosing) {}

    virtual void visit(BlockExpression* e) override;
    virtual void visit(AssignmentExpression *e) override;
};

class SparseSolverVisitor : public SolverVisitorBase {
protected:
    // 'Current' differential equation is for variable with this
    // index in `dvars`.
    unsigned deq_index_ = 0;

    // Expanded local assignments that need to be substituted in for derivative
    // calculations.
    substitute_map local_expr_;

    // Symbolic matrix for backwards Euler step.
    symge::sym_matrix A_;

    // 'Symbol table' for symbolic manipulation.
    symge::symbol_table symtbl_;

    // Flag to indicate whether conserve statements are part of the system
    bool conserve_ = false;

    // rhs of conserve statement
    std::vector<std::string> conserve_rhs_;
    std::vector<unsigned> conserve_idx_;
public:
    using SolverVisitorBase::visit;

    SparseSolverVisitor() {}
    SparseSolverVisitor(scope_ptr enclosing): SolverVisitorBase(enclosing) {}

    virtual void visit(BlockExpression* e) override;
    virtual void visit(AssignmentExpression *e) override;
    virtual void visit(ConserveExpression *e) override;
    virtual void finalize() override;
    virtual void reset() override {
        deq_index_ = 0;
        local_expr_.clear();
        symtbl_.clear();
        conserve_rhs_.clear();
        conserve_idx_.clear();
        conserve_ = false;
        SolverVisitorBase::reset();
    }
};

class LinearSolverVisitor : public SolverVisitorBase {
protected:
    // 'Current' differential equation is for variable with this
    // index in `dvars`.
    unsigned deq_index_ = 0;

    // Expanded local assignments that need to be substituted in for derivative
    // calculations.
    substitute_map local_expr_;

    // Symbolic matrix for backwards Euler step.
    symge::sym_matrix A_;

    // RHS
    std::vector<symge::symbol> rhs_;

    // 'Symbol table' for symbolic manipulation.
    symge::symbol_table symtbl_;

public:
    using SolverVisitorBase::visit;

    LinearSolverVisitor(std::vector<std::string> vars) {
        dvars_ = vars;
    }
    LinearSolverVisitor(scope_ptr enclosing): SolverVisitorBase(enclosing) {}

    virtual void visit(BlockExpression* e) override;
    virtual void visit(LinearExpression *e) override;
    virtual void visit(AssignmentExpression *e) override;
    virtual void finalize() override;
    virtual void reset() override {
        deq_index_ = 0;
        local_expr_.clear();
        symtbl_.clear();
        SolverVisitorBase::reset();
    }
};
