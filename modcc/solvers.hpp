#pragma once

// Transform derivative block into AST representing
// an integration step over the state variables, based on
// solver method.

#include <string>
#include <vector>

#include "astmanip.hpp"
#include "expression.hpp"
#include "symdiff.hpp"
#include "symge.hpp"
#include "visitor.hpp"
#include <libmodcc/export.hpp>

ARB_LIBMODCC_API expression_ptr remove_unused_locals(BlockExpression* block);

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

class ARB_LIBMODCC_API CnexpSolverVisitor : public SolverVisitorBase {
public:
    using SolverVisitorBase::visit;

    CnexpSolverVisitor() {}
    CnexpSolverVisitor(scope_ptr enclosing): SolverVisitorBase(enclosing) {}

    virtual void visit(BlockExpression* e) override;
    virtual void visit(AssignmentExpression *e) override;
};

class ARB_LIBMODCC_API SystemSolver {
protected:
    // Symbolic matrix for backwards Euler step.
    symge::sym_matrix A_;

    // 'Symbol table' for initial variables.
    symge::symbol_table symtbl_;

public:
    struct system_loc {
        unsigned row, col;
    };

    explicit SystemSolver() {}

    void reset() {
        A_.clear();
        symtbl_.clear();
    }
    unsigned size() const {
        return A_.size();
    }
    bool empty() const {
        return A_.empty();
    }
    bool empty_row(unsigned i) const {
        return A_[i].empty();
    }
    void clear_row(unsigned i) {
        A_[i].clear();
    }
    void create_square_matrix(unsigned n) {
        A_ = symge::sym_matrix(n, n);
    }
    void add_entry(system_loc loc, std::string name) {
        A_[loc.row].push_back({loc.col, symtbl_.define(name)});
    }
    void augment(std::vector<std::string> rhs) {
        std::vector<symge::symbol> rhs_sym;
        for (unsigned r = 0; r < rhs.size(); ++r) {
            rhs_sym.push_back(symtbl_.define(rhs[r]));
        }
        A_.augment(rhs_sym);
    }

    // Returns a vector of rows of symbols
    // Needed for normalization
    std::vector<std::vector<symge::symbol>> reduce() {
        return symge::gj_reduce(A_, symtbl_);
    }

    // Returns a vector of local assignments for row updates during system reduction
    std::vector<local_assignment> generate_row_updates(scope_ptr scope, std::vector<symge::symbol> row_sym);

    // Given a row of symbols, generates local assignment for a normalizing term
    local_assignment generate_normalizing_term(scope_ptr scope, std::vector<symge::symbol> row_sym);

    // Given a row of symbols, generates expressions normalizing row updates
    std::vector<expression_ptr> generate_normalizing_assignments(expression_ptr normalizer, std::vector<symge::symbol> row_sym);

    // Returns solution assignment of lhs_vars
    std::vector<expression_ptr> generate_solution_assignments(std::vector<std::string> lhs_vars);

};

class ARB_LIBMODCC_API SparseSolverVisitor : public SolverVisitorBase {
protected:
    solverVariant solve_variant_;

    // 'Current' differential equation is for variable with this
    // index in `dvars`.
    unsigned deq_index_ = 0;

    // Expanded local assignments that need to be substituted in for derivative
    // calculations.
    substitute_map local_expr_;

    // Flag to indicate whether conserve statements are part of the system
    bool conserve_ = false;

    // state variable multiplier/divider
    std::vector<expression_ptr> scale_factor_;

    // rhs of conserve statement
    std::vector<std::string> conserve_rhs_;
    std::vector<unsigned> conserve_idx_;

    // rhs of steadstate
    std::string steadystate_rhs_;

    // System Solver helper
    SystemSolver system_;

public:
    using SolverVisitorBase::visit;

    explicit SparseSolverVisitor(solverVariant s = solverVariant::regular) :
        solve_variant_(s) {}
    SparseSolverVisitor(scope_ptr enclosing): SolverVisitorBase(enclosing) {}

    virtual void visit(BlockExpression* e) override;
    virtual void visit(AssignmentExpression *e) override;
    virtual void visit(CompartmentExpression *e) override;
    virtual void visit(ConserveExpression *e) override;
    virtual void finalize() override;
    virtual void reset() override {
        deq_index_ = 0;
        local_expr_.clear();
        conserve_ = false;
        scale_factor_.clear();
        conserve_rhs_.clear();
        conserve_idx_.clear();
        steadystate_rhs_.clear();
        system_.reset();
        SolverVisitorBase::reset();
    }
};

class ARB_LIBMODCC_API SparseNonlinearSolverVisitor : public SolverVisitorBase {
protected:
    // 'Current' differential equation is for variable with this
    // index in `dvars`.
    unsigned deq_index_ = 0;

    // Expanded local assignments that need to be substituted in for derivative
    // calculations.
    substitute_map local_expr_;

    // F(x) and the Jacobian J(x) for every state variable
    // Needed for Newton's method
    std::vector<expression_ptr> F_;
    std::vector<expression_ptr> J_;

    std::vector<std::string> dvar_temp_;
    std::vector<std::string> dvar_init_;

    // State variable multiplier/divider
    std::vector<expression_ptr> scale_factor_;

    // System Solver helper
    SystemSolver system_;

public:
    using SolverVisitorBase::visit;

    SparseNonlinearSolverVisitor() {}
    SparseNonlinearSolverVisitor(scope_ptr enclosing): SolverVisitorBase(enclosing) {}

    virtual void visit(BlockExpression* e) override;
    virtual void visit(AssignmentExpression *e) override;
    virtual void visit(CompartmentExpression *e) override;
    virtual void visit(ConserveExpression *e) override {};
    virtual void finalize() override;
    virtual void reset() override {
        deq_index_ = 0;
        local_expr_.clear();
        F_.clear();
        J_.clear();
        scale_factor_.clear();
        system_.reset();
        SolverVisitorBase::reset();
    }
};

class ARB_LIBMODCC_API LinearSolverVisitor : public SolverVisitorBase {
protected:
    // 'Current' differential equation is for variable with this
    // index in `dvars`.
    unsigned deq_index_ = 0;

    // Expanded local assignments that need to be substituted in for derivative
    // calculations.
    substitute_map local_expr_;

    // Stores the rhs symbols of the linear system
    std::vector<std::string> rhs_;

    // System Solver helper
    SystemSolver system_;

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
        rhs_.clear();
        system_.reset();
        SolverVisitorBase::reset();
    }
};

class ARB_LIBMODCC_API EulerMaruyamaSolverVisitor : public SolverVisitorBase {
protected:
    // list of white noise names appearing in derivatives on lhs
    std::vector<std::string> wvars_;
    std::vector<std::string> wvars_id_;

    // dx = f(x,t) dt + Σ_k l_k dW_k
    std::vector<expression_ptr> f_;
    std::vector<std::vector<expression_ptr>> L_;
    
    std::string wscale_;

    // Expanded local assignments that need to be substituted in for derivative
    // calculations.
    substitute_map local_expr_;
public:
    using SolverVisitorBase::visit;

    EulerMaruyamaSolverVisitor() {}
    EulerMaruyamaSolverVisitor(scope_ptr enclosing): SolverVisitorBase(enclosing) {}
    EulerMaruyamaSolverVisitor(std::vector<std::string> white_noise_vars)
    :   wvars_{white_noise_vars},
        L_{wvars_.size()}
        //wvar_temp_{wvars_.size()}
    {}

    virtual void visit(BlockExpression* e) override;
    virtual void visit(AssignmentExpression *e) override;
    virtual void reset() override {
        dvars_.clear();
        f_.clear();
        L_.clear();
        local_expr_.clear();
        BlockRewriterBase::reset();
    }
protected:
    // Finalise statements list at end of top block visit.
    virtual void finalize() override;
};
