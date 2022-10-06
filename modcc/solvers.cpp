#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "expression.hpp"
#include "parser.hpp"
#include "solvers.hpp"
#include "symdiff.hpp"
#include "symge.hpp"
#include "visitor.hpp"

// Cnexp solver visitor implementation.

void CnexpSolverVisitor::visit(BlockExpression* e) {
    // Do a first pass to extract variables comprising ODE system
    // lhs; can't really trust 'STATE' block.

    for (auto& stmt: e->statements()) {
        if (stmt && stmt->is_assignment() && stmt->is_assignment()->lhs()->is_derivative()) {
            auto id = stmt->is_assignment()->lhs()->is_derivative();
            dvars_.push_back(id->name());
        }
    }

    BlockRewriterBase::visit(e);
}

void CnexpSolverVisitor::visit(AssignmentExpression *e) {
    auto loc = e->location();
    scope_ptr scope = e->scope();

    auto lhs = e->lhs();
    auto rhs = e->rhs();
    auto deriv = lhs->is_derivative();

    if (!deriv) {
        statements_.push_back(e->clone());
        return;
    }

    auto s = deriv->name();

    linear_test_result r = linear_test(rhs, dvars_);
    if (r.has_error()) {
        append_errors(r.errors());
        error({"CNExp: Could not determine linearity, maybe use a different solver?", loc});
        return;
    }

    if (!r.monolinear(s)) {
        error({"System not diagonal linear for cnexp", loc});
        return;
    }

    Expression* coef = r.coef[s].get();
    if (!coef || is_zero(coef)) {
        // s' = b becomes s = s + b*dt; use b_ as a local variable for
        // the constant term b.
        auto local_b_term = make_unique_local_assign(scope, r.constant.get(), "b_");
        statements_.push_back(std::move(local_b_term.local_decl));
        statements_.push_back(std::move(local_b_term.assignment));
        auto b_ = local_b_term.id->is_identifier()->spelling();

        std::string s_update = pprintf("% = %+%*dt", s, s, b_);
        statements_.push_back(Parser{s_update}.parse_line_expression());
        return;
    }
    else if (r.is_homogeneous) {
        // s' = a*s becomes s = s*exp(a*dt); use a_ as a local variable
        // for the coefficient.
        auto local_a_term = make_unique_local_assign(scope, coef, "a_");
        statements_.push_back(std::move(local_a_term.local_decl));
        statements_.push_back(std::move(local_a_term.assignment));
        auto a_ = local_a_term.id->is_identifier()->spelling();

        std::string s_update = pprintf("% = %*exp_pade_11(%*dt)", s, s, a_);
        statements_.push_back(Parser{s_update}.parse_line_expression());
        return;
    }
    else {
        // s' = a*s + b becomes s = -b/a + (s+b/a)*exp(a*dt); use
        // a_ as a local variable for the coefficient and ba_ for the
        // quotient.
        //
        // Note though this will be numerically bad for very small
        // (or zero) a. Perhaps re-implement as:
        //     s = s + exprel(a*dt)*(s*a+b)*dt
        // where exprel(x) = (exp(x)-1)/x and can be well approximated
        // by e.g. a Taylor expansion for small x.
        //
        // Special case ('gating variable') when s' = (b-s)/a; rather
        // than implement more general algebraic simplification, jump
        // straight to simplified update: s = b + (s-b)*exp(-dt/a).

        // Check for 'gating' case:
        if (rhs->is_binary() && rhs->is_binary()->op()==tok::divide) {
            auto denom = rhs->is_binary()->rhs();
            if (involves_identifier(denom, s)) {
                goto not_gating;
            }
            auto numer = rhs->is_binary()->lhs();
            linear_test_result r = linear_test(numer, {s});
            if (expr_value(r.coef[s]) != -1) {
                goto not_gating;
            }

            auto local_a_term = make_unique_local_assign(scope, denom, "a_");
            auto a_ = local_a_term.id->is_identifier()->spelling();
            auto local_b_term = make_unique_local_assign(scope, r.constant, "b_");
            auto b_ = local_b_term.id->is_identifier()->spelling();

            statements_.push_back(std::move(local_a_term.local_decl));
            statements_.push_back(std::move(local_a_term.assignment));
            statements_.push_back(std::move(local_b_term.local_decl));
            statements_.push_back(std::move(local_b_term.assignment));

            std::string s_update = pprintf("% = %+(%-%)*exp_pade_11(-dt/%)", s, b_, s, b_, a_);
            statements_.push_back(Parser{s_update}.parse_line_expression());
            return;
        }

not_gating:
        auto local_a_term = make_unique_local_assign(scope, coef, "a_");
        auto a_ = local_a_term.id->is_identifier()->spelling();

        auto ba_expr = make_expression<DivBinaryExpression>(loc,
                           r.constant->clone(), local_a_term.id->clone());
        auto local_ba_term = make_unique_local_assign(scope, ba_expr, "ba_");
        auto ba_ = local_ba_term.id->is_identifier()->spelling();

        statements_.push_back(std::move(local_a_term.local_decl));
        statements_.push_back(std::move(local_a_term.assignment));
        statements_.push_back(std::move(local_ba_term.local_decl));
        statements_.push_back(std::move(local_ba_term.assignment));

        std::string s_update = pprintf("% = -%+(%+%)*exp_pade_11(%*dt)", s, ba_, s, ba_, a_);
        statements_.push_back(Parser{s_update}.parse_line_expression());
        return;
    }
}


// Sparse solver visitor implementation.

static expression_ptr as_expression(symge::symbol_term term) {
    Location loc;
    if (term.is_zero()) {
        return make_expression<IntegerExpression>(loc, 0);
    }
    else {
        return make_expression<MulBinaryExpression>(loc,
            make_expression<IdentifierExpression>(loc, name(term.left)),
            make_expression<IdentifierExpression>(loc, name(term.right)));
    }
}

static expression_ptr as_expression(symge::symbol_term_diff diff) {
    Location loc;
    if (diff.left.is_zero() && diff.right.is_zero()) {
        return make_expression<IntegerExpression>(loc, 0);
    }
    else if (diff.right.is_zero()) {
        return as_expression(diff.left);
    }
    else if (diff.left.is_zero()) {
        return make_expression<NegUnaryExpression>(loc,
            as_expression(diff.right));
    }
    else {
        return make_expression<SubBinaryExpression>(loc,
            as_expression(diff.left),
            as_expression(diff.right));
    }
}

std::vector<local_assignment> SystemSolver::generate_row_updates(scope_ptr scope, std::vector<symge::symbol> row_sym) {
    std::vector<local_assignment> S_;
    for (const auto& s: row_sym) {
        if (primitive(s)) continue;

        auto expr = as_expression(definition(s));
        auto local_t_term = make_unique_local_assign(scope, expr.get(), "t_");
        auto t_ = local_t_term.id->is_identifier()->spelling();
        symtbl_.name(s, t_);
        S_.push_back(std::move(local_t_term));
    }
    return S_;
}

local_assignment SystemSolver::generate_normalizing_term(scope_ptr scope, std::vector<symge::symbol> row_sym) {
    // Get the max element in the row
    expression_ptr max;
    for (auto &s: row_sym) {
        auto elem = make_expression<IdentifierExpression>(Location(), symtbl_.name(s));
        auto abs = make_expression<AbsUnaryExpression>(elem->location(), elem->is_identifier()->clone());
        if (!max) {
            max = std::move(abs);
        } else {
            max = make_expression<MaxBinaryExpression>(elem->location(), max->clone(), std::move(abs));
        }
    }
    // Safely invert max
    auto inv = make_expression<SafeInvUnaryExpression>(max->location(), std::move(max));

    // Create a symbol for inv
    auto local_inv_term = make_unique_local_assign(scope, inv.get(), "inv_");

    return local_inv_term;
}

std::vector<expression_ptr> SystemSolver::generate_normalizing_assignments(expression_ptr normalizer, std::vector<symge::symbol> row_sym) {
    std::vector<expression_ptr> S_;
    for (auto &s: row_sym) {
        auto elem = make_expression<IdentifierExpression>(Location(), symtbl_.name(s));
        auto ratio = make_expression<MulBinaryExpression>(elem->location(), elem->clone(), normalizer->clone());
        auto assign = make_expression<AssignmentExpression>(elem->location(), elem->clone(), std::move(ratio));
        S_.push_back(std::move(assign));
    }
    return S_;
}

std::vector<expression_ptr> SystemSolver::generate_solution_assignments(std::vector<std::string> lhs_vars) {
    std::vector<expression_ptr> U_;

    // State variable updates given by rhs/diagonal for reduced matrix.
    Location loc;
    auto nrow = A_.nrow();
    for (unsigned i = 0; i < nrow; ++i) {
        const symge::sym_row& row = A_[i];
        unsigned rhs_col = A_.augcol();
        unsigned lhs_col = -1;
        for (unsigned r = 0; r < A_.nrow(); ++r) {
            if (row[r]) {
                lhs_col = r;
                break;
            }
        }

        if (lhs_col==unsigned(-1)) {
            throw std::logic_error("zero row in matrix solver");
        }
        auto expr = make_expression<AssignmentExpression>(loc,
                        make_expression<IdentifierExpression>(loc, lhs_vars[lhs_col]),
                        make_expression<DivBinaryExpression>(loc,
                            make_expression<IdentifierExpression>(loc, symge::name(A_[i][rhs_col])),
                            make_expression<IdentifierExpression>(loc, symge::name(A_[i][lhs_col]))));

        U_.push_back(std::move(expr));
    }
    return U_;
}


void SparseSolverVisitor::visit(BlockExpression* e) {
    // Do a first pass to extract variables comprising ODE system
    // lhs; can't really trust 'STATE' block.

    for (auto& stmt: e->statements()) {
        if (stmt && stmt->is_assignment() && stmt->is_assignment()->lhs()->is_derivative()) {
            auto id = stmt->is_assignment()->lhs()->is_derivative();
            dvars_.push_back(id->name());
        }
    }
    if (solve_variant_ == solverVariant::steadystate) {
        // create zero_epression local for the rhs
        auto zero_expr = make_expression<NumberExpression>(e->location(), 0.0);
        auto local_a_term = make_unique_local_assign(e->scope(), zero_expr.get(), "a_");
        auto a_ = local_a_term.id->is_identifier()->spelling();

        statements_.push_back(std::move(local_a_term.local_decl));
        statements_.push_back(std::move(local_a_term.assignment));
        steadystate_rhs_ = a_;
    }
    scale_factor_.resize(dvars_.size());

    BlockRewriterBase::visit(e);
}

void SparseSolverVisitor::visit(CompartmentExpression *e) {
    auto loc = e->location();

    for (auto& s: e->is_compartment()->state_vars()) {
        auto it = std::find(dvars_.begin(), dvars_.end(), s->is_identifier()->spelling());
        if (it == dvars_.end()) {
            error({"COMPARTMENT variable is not used", loc});
            return;
        }
        auto idx = it - dvars_.begin();
        scale_factor_[idx] = make_expression<DivBinaryExpression>(
                loc, make_expression<NumberExpression>(loc, 1.0), e->scale_factor()->clone());
    }
}

void SparseSolverVisitor::visit(AssignmentExpression *e) {
    if (system_.empty()) {
        system_.create_square_matrix(dvars_.size());
    }

    auto loc = e->location();
    scope_ptr scope = e->scope();

    auto lhs = e->lhs();
    auto rhs = e->rhs();
    auto deriv = lhs->is_derivative();

    if (!deriv) {
        statements_.push_back(e->clone());

        auto id = lhs->is_identifier();
        if (id) {
            auto expand = substitute(rhs, local_expr_);
            if (involves_identifier(expand, dvars_)) {
                local_expr_[id->spelling()] = std::move(expand);
            }
        }
        return;
    }

    if (conserve_ && !system_.empty_row(deq_index_)) {
        deq_index_++;
        return;
    }

    auto s = deriv->name();
    auto expanded_rhs = substitute(rhs, local_expr_);
    linear_test_result r = linear_test(expanded_rhs, dvars_);
    if (!r.is_homogeneous) {
        error({"System not homogeneous linear for sparse", loc});
        return;
    }

    // Populate sparse symbolic matrix for GE.
    if (s!=dvars_[deq_index_]) {
        error({"ICE: inconsistent ordering of derivative assignments", loc});
        return;
    }

    auto dt_expr = make_expression<IdentifierExpression>(loc, "dt");
    auto one_expr = make_expression<NumberExpression>(loc, 1.0);
    for (unsigned j = 0; j<dvars_.size(); ++j) {
        expression_ptr expr;

        // For regular solve:
        // For zero coefficient and diagonal element, the matrix entry is 1.
        // For non-zero coefficient c and diagonal element, the entry is 1-c*dt.
        // Otherwise, for non-zero coefficient c, the entry is -c*dt.
        // For steady state solve:
        // The entry is always the the coefficient.

        if (r.coef.count(dvars_[j])) {
            expr = solve_variant_ == solverVariant::steadystate ? r.coef[dvars_[j]]->clone() :
                    make_expression<MulBinaryExpression>(loc, r.coef[dvars_[j]]->clone(), dt_expr->clone());

            if (scale_factor_[j]) {
                expr =  make_expression<MulBinaryExpression>(loc, std::move(expr), scale_factor_[j]->clone());
            }
        }

        if (solve_variant_ != solverVariant::steadystate) {
            if (j == deq_index_) {
                if (expr) {
                    expr = make_expression<SubBinaryExpression>(loc,
                                                                one_expr->clone(),
                                                                std::move(expr));
                } else {
                    expr = one_expr->clone();
                }
            } else if (expr) {
                expr = make_expression<NegUnaryExpression>(loc, std::move(expr));
            }
        }

        if (!expr) continue;

        auto local_a_term = make_unique_local_assign(scope, expr.get(), "a_");
        auto a_ = local_a_term.id->is_identifier()->spelling();

        statements_.push_back(std::move(local_a_term.local_decl));
        statements_.push_back(std::move(local_a_term.assignment));

        system_.add_entry({deq_index_, j}, a_);
    }
    ++deq_index_;
}

void SparseSolverVisitor::visit(ConserveExpression *e) {
    if (system_.empty()) {
        system_.create_square_matrix(dvars_.size());
    }
    conserve_ = true;

    auto loc = e->location();
    scope_ptr scope = e->scope();

    int row_idx;

    // Find a row that contains one of the state variables in the conserve statement
    auto& l = e->lhs()->is_stoich()->terms().front();
    auto ident = l->is_stoich_term()->ident()->is_identifier();
    if (ident) {
        auto it = std::find(dvars_.begin(), dvars_.end(), ident->name());
        if (it!=dvars_.end()) {
            row_idx = it - dvars_.begin();
        } else {
            error({"CONSERVE statement unknown is not a state variable", loc});
            return;
        }
    }
    else {
        error({"ICE: coefficient in state variable is not an identifier", loc});
        return;
    }

    // Replace that row with the conserve statement
    system_.clear_row(row_idx);

    for (unsigned j = 0; j < dvars_.size(); ++j) {
        auto state = dvars_[j];

        auto& terms = e->lhs()->is_stoich()->terms();
        auto it = std::find_if(terms.begin(), terms.end(), [&state](expression_ptr& p)
            { return p->is_stoich_term()->ident()->is_identifier()->name() == state;});

        if (it != terms.end()) {
            auto expr = (*it)->is_stoich_term()->coeff()->clone();
            if (scale_factor_[j]) {
                expr =  make_expression<DivBinaryExpression>(loc, std::move(expr), scale_factor_[j]->clone());
            }

            auto local_a_term = make_unique_local_assign(scope, expr.get(), "a_");
            auto a_ = local_a_term.id->is_identifier()->spelling();

            statements_.push_back(std::move(local_a_term.local_decl));
            statements_.push_back(std::move(local_a_term.assignment));

            system_.add_entry({(unsigned)row_idx, j}, a_);
        }
    }


    expression_ptr expr = e->rhs()->clone();
    auto local_a_term = make_unique_local_assign(scope, expr.get(), "a_");
    auto a_ = local_a_term.id->is_identifier()->spelling();

    statements_.push_back(std::move(local_a_term.local_decl));
    statements_.push_back(std::move(local_a_term.assignment));

    conserve_rhs_.push_back(a_);
    conserve_idx_.push_back(row_idx);

}

void SparseSolverVisitor::finalize() {
    if (has_error()) return;

    if (solve_variant_ == solverVariant::steadystate && !conserve_) {
        error({"Conserve statement(s) missing in steady-state solver", {}});
    }

    std::vector<std::string> rhs;
    for (const auto& var: dvars_) {
        auto v = solve_variant_ == solverVariant::steadystate? steadystate_rhs_ : var;
        rhs.push_back(v);
    }
    if (conserve_) {
        for (unsigned i = 0; i < conserve_idx_.size(); ++i) {
            rhs[conserve_idx_[i]] = conserve_rhs_[i];
        }
    }
    system_.augment(rhs);

    // Reduce the system
    auto row_symbols = system_.reduce();

    // Row by row:
    // Generate entries of the system and declare and assign as local variables
    // Generate normalizing terms and normalize the row
    for (auto& row: row_symbols) {
        auto entries = system_.generate_row_updates(block_scope_, row);
        for (auto& l: entries) {
            statements_.push_back(std::move(l.local_decl));
            statements_.push_back(std::move(l.assignment));
        }

        // If size of system > 5 normalize the row updates
        if (system_.size() > 5) {
            auto norm_term = system_.generate_normalizing_term(block_scope_, row);
            auto norm_assigns = system_.generate_normalizing_assignments(norm_term.id->clone(), row);

            statements_.push_back(std::move(norm_term.local_decl));
            statements_.push_back(std::move(norm_term.assignment));
            std::move(std::begin(norm_assigns), std::end(norm_assigns), std::back_inserter(statements_));
        }
    }

    // Update the state variables
    auto updates = system_.generate_solution_assignments(dvars_);
    std::move(std::begin(updates), std::end(updates), std::back_inserter(statements_));

    BlockRewriterBase::finalize();
}

// EulerMaruyama solver visitor implementation

void EulerMaruyamaSolverVisitor::visit(AssignmentExpression *e) {
    auto lhs = e->lhs();
    auto rhs = e->rhs();
    auto deriv = lhs->is_derivative();

    if (!deriv) {
        // add to substitute map if regular assignment involves white noise
        auto orig = e->clone();
        auto id = lhs->is_identifier();
        if (id) {
            auto expand = substitute(rhs, local_expr_);
            if (involves_identifier(expand, wvars_))
                local_expr_[id->spelling()] = std::move(expand);
            else
                statements_.push_back(std::move(orig));
        }
        else {
            statements_.push_back(std::move(orig));
        }
    }
    else {
        // expand the rhs of the derivative with substitute map
        auto expanded_rhs = substitute(rhs, local_expr_);

        // get the determinsitc part: f
        Location loc = e->location();
        auto rhs_deterministic = expanded_rhs->clone();
        for (auto w : wvars_) {
            auto zero_expr = make_expression<NumberExpression>(loc, 0.0);
            rhs_deterministic = substitute(rhs_deterministic, w,  zero_expr);
        }
        rhs_deterministic = constant_simplify(rhs_deterministic);
        f_.push_back(std::move(rhs_deterministic));

        // get the white noise coefficients
        auto r = linear_test(expanded_rhs, wvars_);
        if (!r.is_linear) {
            error({"System is not a valid SDE", e->location()});
            return;
        }
        for (unsigned k=0; k<wvars_.size(); ++k) {
            auto const & w = wvars_[k];
            if (r.coef.count(w))
                L_[k].push_back(std::move(r.coef[w]));
            else
                L_[k].push_back(make_expression<IntegerExpression>(loc, 0));
        }
        // push back placeholder
        statements_.push_back(make_expression<IdentifierExpression>(loc, deriv->name()));
    }
}

void EulerMaruyamaSolverVisitor::visit(BlockExpression* e) {

    // scaling factor for white noise: s = sqrt(dt)
    auto dt_expr = make_expression<IdentifierExpression>(Location{}, "dt");
    auto half_expr = make_expression<NumberExpression>(Location{}, 0.5);
    auto wscale_expr = binary_expression(Location{}, tok::pow, std::move(dt_expr), std::move(half_expr));
    auto wscale = make_unique_local_assign(e->scope(), wscale_expr, "s_");
    wscale_ = wscale.id->is_identifier()->spelling();
    statements_.push_back(std::move(wscale.local_decl));
    statements_.push_back(std::move(wscale.assignment));

    // white noise w
    for (auto const& w : wvars_) {
        // check if symbol is present
        auto w_symbol = e->scope()->find(w);
        if (!w_symbol) {
            error({"couldn't find symbol", e->location()});
            return;
        }

        // scaled white noise w_i = s * w
        auto wscale = make_expression<IdentifierExpression>(Location{}, wscale_);
        auto ww = make_expression<IdentifierExpression>(Location{}, w);
        ww->is_identifier()->symbol(w_symbol);
        auto w_ = binary_expression(Location{}, tok::times, std::move(wscale), std::move(ww));
        auto temp_wvar_term = make_unique_local_assign(e->scope(), w_, "w_");
        auto temp_wvar = temp_wvar_term.id->is_identifier()->spelling();
        wvars_id_.push_back(temp_wvar);
        statements_.push_back(std::move(temp_wvar_term.local_decl));
        statements_.push_back(std::move(temp_wvar_term.assignment));
    }

    // Do a first pass to extract variables comprising ODE system
    // lhs; can't really trust 'STATE' block.
    for (auto& stmt: e->statements()) {
        if (stmt && stmt->is_assignment() && stmt->is_assignment()->lhs()->is_derivative()) {
            auto id = stmt->is_assignment()->lhs()->is_derivative();
            dvars_.push_back(id->name());
        }
    }

    BlockRewriterBase::visit(e);
}

void EulerMaruyamaSolverVisitor::finalize() {
    if (has_error()) return;

    // filter out unused white noise
    std::vector<std::string> wvars_new;
    std::vector<std::string> wvars_id_new;
    std::vector<std::vector<expression_ptr>> L_new;
    for (unsigned k=0; k<wvars_.size(); ++k) {
        bool used = false;
        for (auto& coef : L_[k]){
            if (!is_zero(coef)) {
                used = true;
                break;
            }
        }
        if (used) {
            wvars_new.push_back(std::move(wvars_[k]));
            wvars_id_new.push_back(std::move(wvars_id_[k]));
            L_new.push_back( std::move(L_[k]) );
        }
    }
    wvars_ = std::move(wvars_new);
    wvars_id_ = std::move(wvars_id_new);
    L_ = std::move(L_new);

    // update the state variables
    for (unsigned i = 0; i < dvars_.size(); ++i) {
        // find placeholder expression
        auto placeholder_expr_iter = std::find_if(statements_.begin(), statements_.end(),
            [name=dvars_[i]](expression_ptr const & e) {
                return e->is_identifier() && (e->is_identifier()->spelling() == name); });
        if (placeholder_expr_iter == statements_.end()) {
            error({"inconsistent ordering of derivative assignments"});
            return;
        }
        Location loc = (*placeholder_expr_iter)->location();

        // deterministic part: rhs = x + f dt
        auto rhs = binary_expression(loc, tok::plus, 
            make_expression<IdentifierExpression>(loc, dvars_[i]),
                binary_expression(loc, tok::times,
                    f_[i]->clone(),
                    make_expression<IdentifierExpression>(loc, "dt")));

        // stochastic part: rhs = rhs + Σ_k l_k dW_k
        for (unsigned k=0; k<wvars_.size(); ++k) {
            rhs = binary_expression(loc, tok::plus, rhs->clone(),
                constant_simplify(
                    binary_expression(loc, tok::times, L_[k][i]->clone(),
                        make_expression<IdentifierExpression>(loc, wvars_id_[k]))));
            rhs = constant_simplify(rhs);
        }

        // update state: x = rhs
        auto expr = make_expression<AssignmentExpression>(loc,
            make_expression<IdentifierExpression>(loc, dvars_[i]), std::move(rhs));

        // replace placeholder expression
        *placeholder_expr_iter = std::move(expr);
    }
}


// Linear solver vistior implementation

void LinearSolverVisitor::visit(BlockExpression* e) {
    BlockRewriterBase::visit(e);
}

void LinearSolverVisitor::visit(AssignmentExpression *e) {
    statements_.push_back(e->clone());
    return;
}

void LinearSolverVisitor::visit(LinearExpression *e) {
    auto loc = e->location();
    scope_ptr scope = e->scope();

    if (system_.empty()) {
        system_.create_square_matrix(dvars_.size());
    }

    linear_test_result r = linear_test(e->lhs(), dvars_);
    if (!r.is_homogeneous) {
        error({"System not homogeneous linear for sparse", loc});
        return;
    }

    for (unsigned j = 0; j<dvars_.size(); ++j) {
        expression_ptr expr;

        if (r.coef.count(dvars_[j])) {
            expr = r.coef[dvars_[j]]->clone();
        }

        if (!expr) continue;

        auto a_ = expr->is_identifier()->spelling();

        system_.add_entry({deq_index_, j}, a_);
    }
    rhs_.push_back(e->rhs()->is_identifier()->spelling());
    ++deq_index_;
}

void LinearSolverVisitor::finalize() {
    if (has_error()) return;

    system_.augment(rhs_);

    // Reduce the system
    auto row_symbols = system_.reduce();

    // Row by row:
    // Generate entries of the system and declare and assign as local variables
    // Generate normalizing terms and normalize the row
    for (auto& row: row_symbols) {
        auto entries = system_.generate_row_updates(block_scope_, row);
        for (auto& l: entries) {
            statements_.push_back(std::move(l.local_decl));
            statements_.push_back(std::move(l.assignment));
        }

        // If size of system > 5 normalize the row updates
        if (system_.size() > 5) {
            auto norm_term = system_.generate_normalizing_term(block_scope_, row);
            auto norm_assigns = system_.generate_normalizing_assignments(norm_term.id->clone(), row);

            statements_.push_back(std::move(norm_term.local_decl));
            statements_.push_back(std::move(norm_term.assignment));
            std::move(std::begin(norm_assigns), std::end(norm_assigns), std::back_inserter(statements_));
        }
    }

    // Update the state variables
    auto updates = system_.generate_solution_assignments(dvars_);
    std::move(std::begin(updates), std::end(updates), std::back_inserter(statements_));

    BlockRewriterBase::finalize();
}

void SparseNonlinearSolverVisitor::visit(BlockExpression* e) {
    // Do a first pass to initialize some local variables and extract state variables

    for (auto& stmt: e->statements()) {
        if (stmt && stmt->is_assignment() && stmt->is_assignment()->lhs()->is_derivative()) {
            auto id = stmt->is_assignment()->lhs()->is_derivative();

            // Save the state variables
            dvars_.push_back(id->name());

            // Create identifiers out of the state variables, they will be used to initialize some values
            auto dvar_ident = make_expression<IdentifierExpression>(e->location(), id->name());

            // Create two sets of local variables and assign them to the initial values of the state variables
            // The first set doesn't change across iterations
            auto init_dvar_term = make_unique_local_assign(e->scope(), dvar_ident.get(), "p_");
            auto init_dvar = init_dvar_term.id->is_identifier()->spelling();

            // The second set is updated in every iteration of Newton's method
            auto temp_dvar_term = make_unique_local_assign(e->scope(), dvar_ident.get(), "t_");
            auto temp_dvar = temp_dvar_term.id->is_identifier()->spelling();

            // Save the variable names
            dvar_init_.push_back(init_dvar);
            dvar_temp_.push_back(temp_dvar);

            // Add local declarations and assignment statements
            statements_.push_back(std::move(init_dvar_term.local_decl));
            statements_.push_back(std::move(init_dvar_term.assignment));

            statements_.push_back(std::move(temp_dvar_term.local_decl));
            statements_.push_back(std::move(temp_dvar_term.assignment));
        }
    }
    scale_factor_.resize(dvars_.size());

    BlockRewriterBase::visit(e);
}

void SparseNonlinearSolverVisitor::visit(CompartmentExpression *e) {
    auto loc = e->location();

    for (auto& s: e->is_compartment()->state_vars()) {
        auto it = std::find(dvars_.begin(), dvars_.end(), s->is_identifier()->spelling());
        if (it == dvars_.end()) {
            error({"COMPARTMENT variable is not used", loc});
            return;
        }
        auto idx = it - dvars_.begin();

        auto scale_inv = make_expression<DivBinaryExpression>(
                loc, make_expression<NumberExpression>(loc, 1.0), e->scale_factor()->clone());
        auto local_s_term = make_unique_local_assign(e->scope(), scale_inv.get(), "s_");

        statements_.push_back(std::move(local_s_term.local_decl));
        statements_.push_back(std::move(local_s_term.assignment));
        scale_factor_[idx] = std::move(local_s_term.id);
    }
}

void SparseNonlinearSolverVisitor::visit(AssignmentExpression *e) {
    if (system_.empty()) {
        system_.create_square_matrix(dvars_.size());
    }

    auto loc = e->location();
    scope_ptr scope = e->scope();

    auto lhs = e->lhs();
    auto rhs = e->rhs();
    auto deriv = lhs->is_derivative();

    if (!deriv) {
        statements_.push_back(e->clone());

        auto id = lhs->is_identifier();
        if (id) {
            auto expand = substitute(rhs, local_expr_);
            if (involves_identifier(expand, dvars_)) {
                local_expr_[id->spelling()] = std::move(expand);
            }
        }
        return;
    }

    auto s = deriv->name();
    auto expanded_rhs = substitute(rhs, local_expr_);

    // Populate sparse symbolic matrix for GE.
    if (s!=dvars_[deq_index_]) {
        error({"ICE: inconsistent ordering of derivative assignments", loc});
        return;
    }

    auto dt_expr = make_expression<IdentifierExpression>(loc, "dt");
    auto one_expr = make_expression<NumberExpression>(loc, 1.0);

    // Form and save F(y) = y - x(t) - dt G(y)
    // y    are stored in dvar_temp_ and are updated at every iteration of Newton's method
    // x(t) are stored in dvar_init_ and are constant across iterations of Newton's method
    // G(y) is the rhs of the derivative assignment expression
    // Newton's method is used to find x(t+1) = y s.t. F(y) = 0.

    // `scale_factors` multiply the lhs of a differential equation.
    // After applying Backward Euler and Newton's method, the new F(y) is
    // F(y) = y - x(t) - dt * diag(s^-1) * G(y)

    expression_ptr F_x;
    F_x = make_expression<MulBinaryExpression>(loc, expanded_rhs->clone(), dt_expr->clone());
    if (scale_factor_[deq_index_]) {
        F_x = make_expression<MulBinaryExpression>(loc, std::move(F_x), scale_factor_[deq_index_]->clone());
    }
    F_x = make_expression<AddBinaryExpression>(loc, make_expression<IdentifierExpression>(loc, dvar_init_[deq_index_]), std::move(F_x));
    F_x = make_expression<SubBinaryExpression>(loc, make_expression<IdentifierExpression>(loc, dvar_temp_[deq_index_]), std::move(F_x));

    for (unsigned k = 0; k < dvars_.size(); ++k) {
        F_x = substitute(F_x, dvars_[k], make_expression<IdentifierExpression>(loc, dvar_temp_[k]));
    }

    auto local_f_term = make_unique_local_assign(scope, F_x.get(), "f_");
    auto a_ = local_f_term.id->is_identifier()->spelling();

    statements_.push_back(std::move(local_f_term.local_decl));
    F_.push_back(std::move(local_f_term.assignment));

    // Form and save the Jacobian J(x) of F(x)
    // J(x) = I - dt * ∂G(x)/∂x
    // If scale_factor[x] exists
    // J(x) = I - dt * diag(s^-1) * ∂G(x)/∂x

    linear_test_result r = linear_test(expanded_rhs, dvars_);
    for (unsigned j = 0; j<dvars_.size(); ++j) {
        expression_ptr J_x;

        // For zero coefficient and diagonal element, the matrix entry is 1.
        // For non-zero coefficient c and diagonal element, the entry is 1-c*dt.
        // Otherwise, for non-zero coefficient c, the entry is -c*dt.

        if (r.coef.count(dvars_[j])) {
            J_x = make_expression<MulBinaryExpression>(loc,
                                                        r.coef[dvars_[j]]->clone(),
                                                        dt_expr->clone());

            if (scale_factor_[deq_index_]) {
                J_x =  make_expression<MulBinaryExpression>(loc, std::move(J_x), scale_factor_[deq_index_]->clone());
            }
        }

        if (j==deq_index_) {
            if (J_x) {
                J_x = make_expression<SubBinaryExpression>(loc, one_expr->clone(), std::move(J_x));
            }
            else {
                J_x = one_expr->clone();
            }
        }
        else if (J_x) {
            J_x = make_expression<NegUnaryExpression>(loc, std::move(J_x));
        }

        if (J_x) {
            for (unsigned k = 0; k < dvars_.size(); ++k) {
                J_x = substitute(J_x, dvars_[k], make_expression<IdentifierExpression>(loc, dvar_temp_[k]));
            }
        }

        if (!J_x) continue;

        auto local_j_term = make_unique_local_assign(scope, J_x.get(), "j_");
        auto j_ = local_j_term.id->is_identifier()->spelling();

        statements_.push_back(std::move(local_j_term.local_decl));
        J_.push_back(std::move(local_j_term.assignment));

        system_.add_entry({deq_index_, j}, j_);
    }
    ++deq_index_;
}

void SparseNonlinearSolverVisitor::finalize() {
    if (has_error()) return;

    // Create rhs of A_
    std::vector<std::string> rhs;
    for (const auto& var: F_) {
        auto id = var->is_assignment()->lhs()->is_identifier()->spelling();
        rhs.push_back(id);
    }

    system_.augment(rhs);

    // Reduce the system
    auto row_symbols =  system_.reduce();

    // Row by row:
    // Generate entries of the system and declare and assign as local variables
    // Generate normalizing terms and normalize the row
    std::vector<expression_ptr> S_;
    for (auto& row: row_symbols) {
        auto entries = system_.generate_row_updates(block_scope_, row);
        for (auto& l: entries) {
            statements_.push_back(std::move(l.local_decl));
            S_.push_back(std::move(l.assignment));
        }

        // If size of system > 5 normalize the row updates
        if (system_.size() > 5) {
            auto norm_term = system_.generate_normalizing_term(block_scope_, row);
            auto norm_assigns = system_.generate_normalizing_assignments(norm_term.id->clone(), row);

            statements_.push_back(std::move(norm_term.local_decl));
            S_.push_back(std::move(norm_term.assignment));
            std::move(std::begin(norm_assigns), std::end(norm_assigns), std::back_inserter(S_));
        }
    }

    // Update the state variables
    auto U_ = system_.generate_solution_assignments(dvar_temp_);

    // Create the statements that update the temporary state variables
    // (dvar_temp) after a Newton's iteration and save them in U_
    for (auto& u: U_) {
        auto lhs = u->is_assignment()->lhs();
        auto rhs = u->is_assignment()->rhs();
        u = make_expression<AssignmentExpression>(u->location(), lhs->clone(),
                make_expression<SubBinaryExpression>(u->location(), lhs->clone(), rhs->clone()));
    }

    // Do 3 Newton iterations
    for (unsigned n = 0; n < 3; n++) {
        // Print out the statements that calulate F(xn), J(xn), solve J(xn)^-1*F(xn), update xn -> xn+1
        for (auto &s: F_) {
            statements_.push_back(s->clone());
        }
        for (auto &s: J_) {
            statements_.push_back(s->clone());
        }
        for (auto &s: S_) {
            statements_.push_back(s->clone());
        }
        for (auto &s: U_) {
            statements_.push_back(s->clone());
        }
    }

    Location loc;
    // Finally update the state variables, dvars_ = dvar_temp_
    for (unsigned i = 0; i < dvars_.size(); ++i) {
        auto expr = make_expression<AssignmentExpression>(loc,
                        make_expression<IdentifierExpression>(loc, dvars_[i]),
                            make_expression<IdentifierExpression>(loc, dvar_temp_[i]));
        statements_.push_back(std::move(expr));
    }

    BlockRewriterBase::finalize();
}

// Implementation for `remove_unused_locals`: uses two visitors,
// `UnusedVisitor` and `RemoveVariableVisitor` below.

class UnusedVisitor : public Visitor {
protected:
    std::multimap<std::string, std::string> deps;
    std::set<std::string> unused_ids;
    std::set<std::string> used_ids;
    Symbol* lhs_sym = nullptr;

public:
    using Visitor::visit;

    UnusedVisitor() {}

    virtual void visit(Expression* e) override {}

    virtual void visit(BlockExpression* e) override {
        for (auto& s: e->statements()) {
            s->accept(this);
        }
    }

    virtual void visit(AssignmentExpression* e) override {
        auto lhs = e->lhs()->is_identifier();
        if (!lhs) return;

        lhs_sym = lhs->symbol();
        e->rhs()->accept(this);
        lhs_sym = nullptr;
    }

    virtual void visit(UnaryExpression* e) override {
        e->expression()->accept(this);
    }

    virtual void visit(BinaryExpression* e) override {
        e->lhs()->accept(this);
        e->rhs()->accept(this);
    }

    virtual void visit(CallExpression* e) override {
        for (auto& a: e->args()) {
            a->accept(this);
        }
    }

    virtual void visit(IfExpression* e) override {
        e->condition()->accept(this);
        e->true_branch()->accept(this);
        if (e->false_branch()) {
            e->false_branch()->accept(this);
        }
    }

    virtual void visit(IdentifierExpression* e) override {
        if (lhs_sym && lhs_sym->is_local_variable()) {
            deps.insert({lhs_sym->name(), e->name()});
        }
        else {
            used_ids.insert(e->name());
        }
    }

    virtual void visit(LocalDeclaration* e) override {
        for (auto& v: e->variables()) {
            unused_ids.insert(v.first);
        }
    }

    std::set<std::string> unused_locals() {
        if (!computed_) {
            for (auto& id: used_ids) {
                remove_deps_from_unused(id, {});
            }
            computed_ = true;
        }
        return unused_ids;
    }

    void reset() {
        deps.clear();
        unused_ids.clear();
        used_ids.clear();
        computed_ = false;
    }

private:
    bool computed_ = false;

    void remove_deps_from_unused(const std::string& id, std::set<std::string> visited) {
        auto range = deps.equal_range(id);
        for (auto i = range.first; i != range.second; ++i) {
            if (unused_ids.count(i->second) && visited.find(i->second) == visited.end()) {
                visited.insert(i->second);
                remove_deps_from_unused(i->second, visited);
            }
        }
        unused_ids.erase(id);
    }
};

class RemoveVariableVisitor: public BlockRewriterBase {
    std::set<std::string> remove_;

public:
    using BlockRewriterBase::visit;

    RemoveVariableVisitor(std::set<std::string> ids):
        remove_(std::move(ids)) {}

    RemoveVariableVisitor(std::set<std::string> ids, scope_ptr enclosing):
        BlockRewriterBase(enclosing), remove_(std::move(ids)) {}

    virtual void visit(LocalDeclaration* e) override {
        auto replacement = e->clone();
        auto& vars = replacement->is_local_declaration()->variables();

        for (const auto& id: remove_) {
            vars.erase(id);
        }
        if (!vars.empty()) {
            statements_.push_back(std::move(replacement));
        }
    }

    virtual void visit(AssignmentExpression* e) override {
        std::string lhs_id = e->lhs()->is_identifier()->name();
        if (!remove_.count(lhs_id)) {
            statements_.push_back(e->clone());
        }
    }
};

ARB_LIBMODCC_API expression_ptr remove_unused_locals(BlockExpression* block) {
    UnusedVisitor unused_visitor;
    block->accept(&unused_visitor);

    RemoveVariableVisitor remove_visitor(unused_visitor.unused_locals());
    block->accept(&remove_visitor);
    return remove_visitor.as_block(false);
}
