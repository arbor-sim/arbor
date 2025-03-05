#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <unordered_set>

#include "errorvisitor.hpp"
#include "functionexpander.hpp"
#include "functioninliner.hpp"
#include "procinliner.hpp"
#include "kineticrewriter.hpp"
#include "linearrewriter.hpp"
#include "module.hpp"
#include "parser.hpp"
#include "solvers.hpp"
#include "symdiff.hpp"
#include "visitor.hpp"

class NrnCurrentRewriter: public BlockRewriterBase {
    expression_ptr id(const std::string& name, Location loc) {
        return make_expression<IdentifierExpression>(loc, name);
    }

    expression_ptr id(const std::string& name) {
        return id(name, loc_);
    }

    static sourceKind current_update(Expression* e) {
        if(auto a = e->is_assignment()) {
            if(auto sym = a->lhs()->is_identifier()->symbol()) {
                if(auto var = sym->is_local_variable()) {
                    if(auto ext = var->external_variable()) {
                        sourceKind src = ext->data_source();
                        if (src==sourceKind::current_density ||
                            src==sourceKind::current ||
                            src==sourceKind::ion_current_density ||
                            src==sourceKind::ion_current)
                        {
                            return src;
                        }
                    }
                }
            }
        }
        return sourceKind::no_source;
    }

    bool has_current_update_ = false;
    std::set<std::string> current_vars_;
    std::map<std::string, expression_ptr> conductivity_exps_;

public:

    std::string non_specific_current = "";
    using BlockRewriterBase::visit;

    virtual void finalize() override {
        if (has_current_update_) {
            expression_ptr current_sum, conductivity_sum;
            for (auto& curr: current_vars_) {
                auto curr_id = make_expression<IdentifierExpression>(Location{}, curr);
                if (!current_sum) {
                    current_sum = std::move(curr_id);
                } else {
                    current_sum = make_expression<AddBinaryExpression>(
                            Location{}, std::move(current_sum), std::move(curr_id));
                }
            }
            for (auto& [name, cond]: conductivity_exps_) {
                if (!conductivity_sum) {
                    conductivity_sum = cond->clone();
                } else {
                    conductivity_sum = make_expression<AddBinaryExpression>(
                            Location{}, std::move(conductivity_sum), cond->clone());
                }
            }
            if (current_sum) {
                statements_.push_back(make_expression<AssignmentExpression>(loc_,
                        id("current_"), std::move(current_sum)));
            }
            if (conductivity_sum) {
                statements_.push_back(make_expression<AssignmentExpression>(loc_,
                        id("conductivity_"), std::move(conductivity_sum)));
            }
        }
    }

    virtual void visit(SolveExpression *e) override {}
    virtual void visit(ConductanceExpression *e) override {}
    virtual void visit(AssignmentExpression *e) override {
        statements_.push_back(e->clone());

        sourceKind current_source = current_update(e);
        if (current_source != sourceKind::no_source) {
            has_current_update_ = true;
            auto name = e->lhs()->is_identifier()->name();
            auto visited_current = current_vars_.count(name);
            current_vars_.insert(name);

            linear_test_result L = linear_test(e->rhs(), {"v"});
            if (L.coef.count("v") && !visited_current) {
                conductivity_exps_[name] = L.coef.at("v")->clone();
            }
        }
    }
};

std::string Module::error_string() const {
    std::string str;
    for (const error_entry& entry: errors()) {
        if (!str.empty()) str += '\n';
        str += red("  * ");
        str += white(pprintf("%:% ", source_name(), entry.location));
        str += entry.message;
    }
    return str;
}

std::string Module::warning_string() const {
    std::string str;
    for (auto& entry: warnings()) {
        if (!str.empty()) str += '\n';
        str += purple("  * ");
        str += white(pprintf("%:% ", source_name(), entry.location));
        str += entry.message;
    }
    return str;
}

void Module::add_callable(symbol_ptr callable) {
    callables_.push_back(std::move(callable));
}

bool Module::semantic() {
    ////////////////////////////////////////////////////////////////////////////
    // create the symbol table
    // there are three types of symbol to look up
    //  1. variables
    //  2. function calls
    //  3. procedure calls
    // the symbol table is generated, then we can traverse the AST and verify
    // that all symbols are correctly used
    ////////////////////////////////////////////////////////////////////////////

    // first add variables defined in the NEURON, ASSIGNED, WHITE_NOISE and PARAMETER
    // blocks these symbols have "global" scope, i.e. they are visible to all
    // functions and procedures in the mechanism
    if (!add_variables_to_symbols()) return false;

    // Helper which iterates over a vector of Symbols, moving them into the
    // symbol table.
    // Returns false if a symbol name clashes with the name of a symbol that
    // is already in the symbol table.
    bool linear_homogeneous = true;
    std::vector<std::string> state_vars;
    std::vector<std::string> white_noise_vars;
    auto move_symbols = [this] (std::vector<symbol_ptr>& symbol_list) {
        for(auto& symbol: symbol_list) {
            bool is_found = (symbols_.find(symbol->name()) != symbols_.end());
            if(is_found) {
                error(
                    pprintf("'%' clashes with previously defined symbol",
                            symbol->name()),
                    symbol->location()
                );
                return false;
            }
            // move symbol to table
            symbols_[symbol->name()] = std::move(symbol);
        }
        return true;
    };

    // Add built in function that approximate exp use pade polynomials
    callables_.push_back(
        Parser{"FUNCTION exp_pade_11(z) { exp_pade_11=(1+0.5*z)/(1-0.5*z) }"}.parse_function());
    callables_.push_back(
        Parser{
            "FUNCTION exp_pade_22(z)"
            "{ exp_pade_22=(1+0.5*z+0.08333333333333333*z*z)/(1-0.5*z+0.08333333333333333*z*z) }"
        }.parse_function());

    // move functions and procedures to the symbol table
    if(!move_symbols(callables_))  return false;

    // Before starting the inlining process, look for the BREAKPOINT block:
    // if it includes a SOLVE statement, check that it is the first statement
    // in the block.
    if (has_symbol("breakpoint", symbolKind::procedure)) {
        bool found_non_solve = false;
        auto breakpoint = symbols_["breakpoint"]->is_procedure();
        for (const auto& s: breakpoint->body()->statements()) {
            if(!s->is_solve_statement()) {
                found_non_solve = true;
            }
            else if (found_non_solve) {
                error("SOLVE statements must come first in BREAKPOINT block", s->location());
                return false;
            }
        }
    }

    // perform semantic analysis and inlining on function and procedure bodies
    if(auto errors = semantic_func_proc()) {
        error("There were "+std::to_string(errors)+" errors in the semantic analysis");
        return false;
    }

    // All API methods are generated from statements in one of the special procedures
    // defined in NMODL, e.g. the init() API call is based on the INITIAL block.
    // When creating an API method, the first task is to look up the source procedure,
    // i.e. the INITIAL block for init(). This lambda takes care of this repetative
    // lookup work, with error checking.
    auto make_empty_api_method = [this]
            (std::string const& name, std::string const& source_name)
            -> std::pair<APIMethod*, ProcedureExpression*>
    {
        if( !has_symbol(source_name, symbolKind::procedure) ) {
            error(pprintf("unable to find symbol '%'", yellow(source_name)),
                   Location());
            return std::make_pair(nullptr, nullptr);
        }

        auto source = symbols_[source_name]->is_procedure();
        auto loc = source->location();

        if( symbols_.find(name)!=symbols_.end() ) {
            error(pprintf("'%' clashes with reserved name, please rename it",
                          yellow(name)),
                  symbols_.find(name)->second->location());
            return std::make_pair(nullptr, source);
        }
        std::vector<expression_ptr> args;
        for (auto& a: source->args()) {
            args.push_back(a->clone());
        }
        symbols_[name] = make_symbol<APIMethod>(
                          loc, name,
                          std::move(args),
                          make_expression<BlockExpression>
                            (loc, expr_list_type(), false)
                         );

        auto proc = symbols_[name]->is_api_method();
        return std::make_pair(proc, source);
    };

    // ... except for write_ions(), which we construct by hand here.

    expr_list_type ion_assignments;

    auto sym_to_id = [](Symbol* sym) -> expression_ptr {
        auto id = make_expression<IdentifierExpression>(sym->location(), sym->name());
        id->is_identifier()->symbol(sym);
        return id;
    };

    for (auto& sym: symbols_) {
        Location loc;

        // get white noise variable names
        auto wn = sym.second->is_white_noise();
        if (wn) {
            white_noise_vars.push_back(wn->name());
            continue;
        }

        auto state = sym.second->is_variable();
        if (!state || !state->is_state()) continue;
        state_vars.push_back(state->name());

        auto shadowed = state->shadows();
        if (!shadowed) continue;

        auto ionvar = shadowed->is_indexed_variable();
        if (!ionvar || !ionvar->is_ion() || !ionvar->is_write()) continue;

        ion_assignments.push_back(
            make_expression<AssignmentExpression>(loc,
                sym_to_id(ionvar), sym_to_id(state)));
    }

    symbols_["write_ions"] = make_symbol<APIMethod>(Location{}, "write_ions",
        std::vector<expression_ptr>(),
        make_expression<BlockExpression>(Location{}, std::move(ion_assignments), false));

    //.........................................................................
    // init : based on the INITIAL block (i.e. the 'initial' procedure
    //.........................................................................

    // insert an empty INITIAL block if none was defined in the .mod file.
    if( !has_symbol("initial", symbolKind::procedure) ) {
        symbols_["initial"] = make_symbol<ProcedureExpression>(
                Location(), "initial",
                std::vector<expression_ptr>(),
                make_expression<BlockExpression>(Location(), expr_list_type(), false),
                procedureKind::initial
        );
    }
    auto initial_api = make_empty_api_method("init", "initial");
    auto api_init  = initial_api.first;
    auto proc_init = initial_api.second;

    auto& init_body = api_init->body()->statements();

    api_init->semantic(symbols_);
    scope_ptr init_scope = api_init->scope();

    for(auto& e : *proc_init->body()) {
        auto solve_expression = e->is_solve_statement();
        if (!white_noise_vars.empty() && involves_identifier(e, white_noise_vars)) {
            error("An error occured while compiling the INITIAL block. "
                  "White noise is not allowed.", e->location());
            return false;
        }
        if (solve_expression) {
            // Grab SOLVE statements, put them in `body` after translation.
            std::set<std::string> solved_ids;
            std::unique_ptr<SolverVisitorBase> solver;

            // The solve expression inside an initial block can only refer to a linear block
            auto solve_proc = solve_expression->procedure();

            if (solve_proc->kind() == procedureKind::linear) {
                solver = std::make_unique<LinearSolverVisitor>(state_vars);
                auto rewrite_body = linear_rewrite(solve_proc->body(), state_vars);
                if (!rewrite_body) {
                    error("An error occured while compiling the LINEAR block. "
                          "Check whether the statements are in fact linear.");
                    return false;
                }

                rewrite_body->semantic(init_scope);
                rewrite_body->accept(solver.get());
            } else if (solve_proc->kind() == procedureKind::kinetic &&
                       solve_expression->variant() == solverVariant::steadystate) {
                solver = std::make_unique<SparseSolverVisitor>(solverVariant::steadystate);
                auto rewrite_body = kinetic_rewrite(solve_proc->body());

                rewrite_body->semantic(init_scope);
                rewrite_body->accept(solver.get());
            } else {
                error("A SOLVE expression in an INITIAL block can only be used to solve a "
                      "LINEAR block or a KINETIC block at steadystate and " +
                      solve_expression->name() + " is neither.", solve_expression->location());
                return false;
            }

            if (auto solve_block = solver->as_block(false)) {
                // Check that we didn't solve an already solved variable.
                for (const auto &id: solver->solved_identifiers()) {
                    if (solved_ids.count(id) > 0) {
                        error("Variable " + id + " solved twice!", solve_expression->location());
                        return false;
                    }
                    solved_ids.insert(id);
                }

                solve_block = remove_unused_locals(solve_block->is_block());

                // Copy body into init.
                for (auto &stmt: solve_block->is_block()->statements()) {
                    init_body.emplace_back(stmt->clone());
                }
            } else {
                // Something went wrong: copy errors across.
                append_errors(solver->errors());
                return false;
            }
        } else {
            init_body.emplace_back(e->clone());
        }
    }

    api_init->semantic(symbols_);

    // Look in the symbol table for a procedure with the name "breakpoint".
    // This symbol corresponds to the BREAKPOINT block in the .mod file
    // There are two APIMethods generated from BREAKPOINT.
    // The first is advance_state, which is the first case handled below.
    // The second is compute_currents, which is handled after this block
    auto state_api  = make_empty_api_method("advance_state", "breakpoint");
    auto api_state  = state_api.first;
    auto breakpoint = state_api.second; // implies we are building the `advance_state()` method.

    if(!breakpoint) {
        error("a BREAKPOINT block is required");
        return false;
    }

    api_state->semantic(symbols_);
    scope_ptr advance_state_scope = api_state->scope();

    // Grab SOLVE statements, put them in `advance_state` after translation.
    bool found_solve = false;
    std::set<std::string> solved_ids;

    for(auto& e: (breakpoint->body()->statements())) {
        SolveExpression* solve_expression = e->is_solve_statement();
        if(!solve_expression) {
            continue;
        }
        found_solve = true;

        // If the derivative block is a kinetic block, perform kinetic rewrite first.
        auto deriv = solve_expression->procedure();
        auto solve_body = deriv->body()->clone();
        if (deriv->kind()==procedureKind::kinetic) {
            solve_body = kinetic_rewrite(deriv->body());
        }
        else if (deriv->kind()==procedureKind::linear) {
            solve_body = linear_rewrite(deriv->body(), state_vars);
        }

        // Calculate linearity, homogeneity and stochasticity of the statements in the derivative block.
        bool linear = true;
        bool homogeneous = true;
        std::vector<expression_ptr> substitution;
        for (auto& s: solve_body->is_block()->statements()) {
            // loop over declared white noise variables
            if (!white_noise_vars.empty()) {
                for (auto const & w : white_noise_vars) {
                    // check whether a statement contains an expression involving white noise
                    if (involves_identifier(s, w)) {
                        // mark the white noise variable as used, and set its index if we see this
                        // variable for the first time
                        auto it = white_noise_block_.used.insert(std::make_pair(w,0u));
                        if (it.second) {
                            // set white noise lookup index
                            const unsigned int idx = white_noise_block_.used.size()-1;
                            symbols_.find(w)->second->is_white_noise()->set_index(idx);
                            it.first->second = idx;
                        }
                    }
                }
            }
            if(s->is_assignment() && !state_vars.empty()) {
                linear_test_result r = linear_test(s->is_assignment()->rhs(), state_vars);
                if (!s->is_assignment()->lhs()->is_derivative() && !r.is_constant) {
                    substitution.push_back(s->clone());
                }
                linear &= r.is_linear;
                homogeneous &= r.is_homogeneous;
            }
        }
        linear_homogeneous &= (linear & homogeneous);

        // filter out unused white noise variables from the white noise vector
        white_noise_vars.clear();
        for (auto const & w : white_noise_block_.used) {
            white_noise_vars.push_back(w.first);
        }
        bool stochastic = (white_noise_vars.size() > 0u);

        if (stochastic && (solve_expression->method() != solverMethod::stochastic)) {
            error("SOLVE expression '" + solve_expression->name() + "' involves white noise and can "
                  "only be solved using the stochastic method", solve_expression->location());
            return false;
        }

        // Construct solver based on system kind, linearity and solver method.
        std::unique_ptr<SolverVisitorBase> solver;
        switch(solve_expression->method()) {
        case solverMethod::cnexp:
            for (const auto& s: substitution) {
                warning("Assignments to local variable containing state variables will not be integrated in time: " + s->to_string(), s->location());
            }
            solver = std::make_unique<CnexpSolverVisitor>();
            break;
        case solverMethod::sparse: {
            if (linear) {
                solver = std::make_unique<SparseSolverVisitor>(solve_expression->variant());
            }
            else {
                solver = std::make_unique<SparseNonlinearSolverVisitor>();
            }
            break;
        }
        case solverMethod::stochastic:
                solver = std::make_unique<EulerMaruyamaSolverVisitor>(white_noise_vars);
            break;
        case solverMethod::none:
            if (deriv->kind()==procedureKind::linear) {
                solver = std::make_unique<LinearSolverVisitor>(state_vars);
            }
            else {
                solver = std::make_unique<DirectSolverVisitor>();
            }
            break;
        }
        // Perform semantic analysis on the solve block statements and solve them.
        solve_body->semantic(advance_state_scope);
        solve_body->accept(solver.get());

        if (auto solve_block = solver->as_block(false)) {
            // Check that we didn't solve an already solved variable.
            for (const auto& id: solver->solved_identifiers()) {
                if (solved_ids.count(id)>0) {
                    error("Variable "+id+" solved twice!", e->location());
                    return false;
                }
                solved_ids.insert(id);
            }

            // May have now redundant local variables; remove these first.
            solve_block->semantic(advance_state_scope);
            solve_block = remove_unused_locals(solve_block->is_block());

            // Copy body into advance_state.
            for (auto& stmt: solve_block->is_block()->statements()) {
                api_state->body()->statements().push_back(std::move(stmt));
            }
        }
        else {
            // Something went wrong: copy errors across.
            append_errors(solver->errors());
            return false;
        }
    }

    // handle the case where there is a SOLVE in BREAKPOINT (which is the typical case)
    if (found_solve) {
        // Redo semantic pass in order to eliminate any removed local symbols.
        api_state->semantic(symbols_);
    }

    // Run remove locals pass again on the whole body in case `dt` was never used.
    api_state->body(remove_unused_locals(api_state->body()));
    api_state->semantic(symbols_);

    //..........................................................
    // compute_currents : update contributions to currents
    //..........................................................
    NrnCurrentRewriter compute_currents_rewriter;
    // Register non-specific current name
    if (neuron_block_.has_nonspecific_current()) compute_currents_rewriter.non_specific_current = neuron_block_.nonspecific_current.spelling;

    breakpoint->accept(&compute_currents_rewriter);

    for (auto& s: breakpoint->body()->statements()) {
        if(s->is_assignment() && !state_vars.empty()) {
            linear_test_result r = linear_test(s->is_assignment()->rhs(), state_vars);
            linear_homogeneous &= r.is_linear;
            linear_homogeneous &= r.is_homogeneous;
        }
    }

    auto compute_currents_block = compute_currents_rewriter.as_block();
    if (!compute_currents_block) {
        append_errors(compute_currents_rewriter.errors());
        return false;
    }

    symbols_["compute_currents"] =
        make_symbol<APIMethod>(
                breakpoint->location(), "compute_currents",
                std::vector<expression_ptr>(),
                constant_simplify(compute_currents_block));
    symbols_["compute_currents"]->semantic(symbols_);

    if (has_symbol("net_receive", symbolKind::procedure)) {
        auto net_rec_api = make_empty_api_method("net_rec_api", "net_receive");
        // handle Arbor specifics
        if (net_rec_api.second->args().size() > 1) {
            error(pprintf("NET_RECEIVE takes at most one argument (Arbor limitation!)"), net_rec_api.first->location());
        }
        net_rec_api.first->body(net_rec_api.second->body()->clone());
        if (net_rec_api.second) {
            for (auto &s: net_rec_api.second->body()->statements()) {
                if (s->is_assignment()) {
                    for (const auto &id: state_vars) {
                        auto coef = symbolic_pdiff(s->is_assignment()->rhs(), id);
                        if(!coef) {
                            linear_homogeneous = false;
                            continue;
                        }
                        if(coef->is_number()) {
                            if (!s->is_assignment()->lhs()->is_identifier()) {
                                error(pprintf("Left hand side of assignment is not an identifier"));
                                return false;
                            }
                            linear_homogeneous &= s->is_assignment()->lhs()->is_identifier()->name() == id ?
                                                  coef->is_number()->value() == 1 :
                                                  coef->is_number()->value() == 0;
                        }
                        else {
                            linear_homogeneous = false;
                        }
                    }
                }
            }
        }
    }
    linear_ = linear_homogeneous;

    post_events_ = has_symbol("post_event", symbolKind::procedure);
    if (post_events_) {
        auto post_events_api = make_empty_api_method("post_event_api", "post_event");
        post_events_api.first->body(post_events_api.second->body()->clone());
    }

    // check voltage mechanisms before rev pot ... otherwise we are in trouble
    check_voltage_mechanism();

    // Are we writing an ionic reversal potential? If so, change the moduleKind to
    // `revpot` and assert that the mechanism is 'pure': it has no state variables;
    // it contributes to no currents, ionic or otherwise; it isn't a point mechanism;
    // and it writes to only one ionic reversal potential.
    check_revpot_mechanism();

    // Perform semantic analysis and inlining on function and procedure bodies
    // in order to inline calls inside the newly crated API methods.
    semantic_func_proc();

    return !has_error();
}

struct builtin_info {
    std::string name;
    sourceKind source;
    accessKind access;
    std::string channel;
    Location location;
};

auto make_builtin_variables(moduleKind mod_kind) {
    std::vector<builtin_info> result = {
        {"v",       sourceKind::voltage,      mod_kind == moduleKind::voltage ? accessKind::write : accessKind::read, "", {}},
        {"v_peer",  sourceKind::peer_voltage, accessKind::read,                                                       "", {}},
        {"celsius", sourceKind::temperature,  accessKind::read,                                                       "", {}},
        {"diam",    sourceKind::diameter,     accessKind::read,                                                       "", {}},
        {"area",    sourceKind::area,         accessKind::read,                                                       "", {}},
    };
    return result;
}

auto make_ion_variables(moduleKind mod_kind, const std::vector<IonDep>& ions) {
    std::vector<builtin_info> result;
    for(auto const& ion: ions) {
        auto channel = ion.name;
        for(auto const& var: ion.write) {
            auto name = var.spelling;
            result.push_back({name, ion_source(channel, name, mod_kind), accessKind::write, channel, var.location});
        }
        for(auto const& var: ion.read) {
            auto name = var.spelling;
            result.push_back({name, ion_source(channel, name, mod_kind), accessKind::read, channel, var.location});
        }
    }
    return result;
}

/// populate the symbol table with class scope variables
bool Module::add_variables_to_symbols() {
    auto builtins = make_builtin_variables(kind_);
    auto ion_vars = make_ion_variables(kind_, neuron_block_.ions);

    auto is_builtin = [&](auto& nm) -> bool {
        return std::count_if(builtins.begin(), builtins.end(),
                            [&](auto& v) { return v.name == nm; });
    };

    auto is_ion_var = [&](auto& nm) -> bool {
        return std::count_if(ion_vars.begin(), ion_vars.end(),
                            [&](auto& v) { return v.name == nm; });
    };


    auto create_variable =
        [this](const Token& token, accessKind a, visibilityKind v, linkageKind l,
               rangeKind r, bool is_state = false) -> symbol_ptr& {
            auto var = new VariableExpression(token.location, token.spelling);
            var->access(a);
            var->visibility(v);
            var->linkage(l);
            var->range(r);
            var->state(is_state);
            return symbols_[var->name()] = symbol_ptr{var};
        };

    // add indexed variables to the table
    auto create_indexed_variable =
        [this](std::string const& name, sourceKind data_source,
               accessKind acc, std::string ch, Location loc) -> symbol_ptr& {
        if (symbols_.count(name)) {
            throw compiler_exception(
                pprintf("the symbol % already exists", yellow(name)), loc);
        }
        return symbols_[name] =
            make_symbol<IndexedVariable>(loc, name, data_source, acc, ch);
    };

    auto create_white_noise = [this](Token const & token) -> symbol_ptr& {
        if (symbols_.count(token.spelling)) {
            throw compiler_exception(
                pprintf("the symbol % already exists", yellow(token.spelling)), token.location);
        }
        return symbols_[token.spelling] = make_symbol<WhiteNoise>(token.location, token.spelling);
    };

    auto current_kind     = kind_ == moduleKind::density ? sourceKind::current_density : sourceKind::current;
    auto conductance_kind = kind_ == moduleKind::density ? sourceKind::conductivity    : sourceKind::conductance;

    create_indexed_variable("current_",      current_kind,     accessKind::write, "", Location());
    create_indexed_variable("conductivity_", conductance_kind, accessKind::write, "", Location());
    create_variable(Token{tok::identifier, "dt", Location()}, accessKind::read, visibilityKind::global, linkageKind::local, rangeKind::scalar);

    for (const auto& [name, source, access, channel, location]: builtins) {
        create_indexed_variable(name, source, access, channel, location);
    }
    parameter_block_.parameters.erase(
        std::remove_if(parameter_block_.begin(), parameter_block_.end(),
                       [&](const Id& id) { return is_builtin(id.name()); }),
        parameter_block_.end()
    );

    // If we put back support for accessing cell time again from NMODL code,
    // add indexed_variable also for "time" with appropriate cell-index based
    // indirection in printers.

    // Add state variables.
    for (const Id& id: state_block_) {
        create_variable(id.token, accessKind::readwrite, visibilityKind::local, linkageKind::local, rangeKind::range, true);
    }

    // Add parameters, ignoring built-in voltage variables "v" and "v_peer".
    for (const Id& id: parameter_block_) {
        if (is_builtin(id.name())) continue;
        if (is_ion_var(id.name())) continue;

        // Parameters are scalar by default, but may later be changed to range.
        auto& sym = create_variable(id.token, accessKind::read, visibilityKind::global, linkageKind::local, rangeKind::scalar);

        // Set default value if one was specified.
        if (id.has_value()) sym->is_variable()->value(std::stod(id.value));
    }


    // Add 'assigned' variables, ignoring built-in voltage variables "v" and "v_peer".
    for (const Id& id: assigned_block_) {
        auto builtin = std::find_if(builtins.begin(), builtins.end(),
                                    [&] (auto& v) { return v.name == id.name(); });
        if (builtin != builtins.end()) {
                error(pprintf("the symbol '%' is defined as a builtin and cannot be used as an ASSIGNED here %.",
                              yellow(id.name()), id.token.location),
                      id.token.location);
                return false;
        }

        auto ion_var = std::find_if(ion_vars.begin(), ion_vars.end(),
                                    [&] (auto& v) { return v.name == id.name(); });
        if (ion_var != ion_vars.end()) {
                error(pprintf("the symbol '%' is defined as a ion variable and cannot be used as an ASSIGNED here %.",
                              yellow(id.name()), id.token.location),
                      id.token.location);
                return false;
        }

        create_variable(id.token, accessKind::readwrite, visibilityKind::local, linkageKind::local, rangeKind::range);
    }

    ////////////////////////////////////////////////////
    // parse the NEURON block data, and use it to update
    // the variables in symbols_
    ////////////////////////////////////////////////////
    // first the ION channels
    // add ion channel variables
    auto update_ion_symbols = [this, create_indexed_variable]
            (Token const& tkn, accessKind acc, const std::string& channel) {
        std::string name = tkn.spelling;
        sourceKind data_source = ion_source(channel, name, kind_);

        // If the symbol already exists and is not a state variable,
        // it is an error.
        //
        // Otherwise create an indexed variable and associate it
        // with the state variable if present (via a different name)
        // for ion state updates.
        VariableExpression* state = nullptr;
        if (has_symbol(name)) {
            state = symbols_[name].get()->is_variable();
            if (!state) {
                error(pprintf("the symbol defined % can't be redeclared", yellow(name)), tkn.location);
                return false;
            }
            if (!state->is_state()) {
                error(pprintf("the symbol defined % at % can't be redeclared",
                    state->location(), yellow(name)), tkn.location);
                return false;
            }
            name += "_shadowed_";
        }

        auto& sym = create_indexed_variable(name, data_source, acc, channel, tkn.location);

        if (state) {
            state->shadows(sym.get());
        }

        return true;
    };

    // Nonspecific current variables are represented by an indexed variable
    // with a 'current' data source. Assignments in the NrnCurrent block will
    // later be rewritten so that these contributions are accumulated in `current_`
    // (potentially saving some weight multiplications);

    if( neuron_block_.has_nonspecific_current() ) {
        auto const& i = neuron_block_.nonspecific_current;
        create_indexed_variable(i.spelling, current_kind, accessKind::noaccess, "", i.location);
    }

    std::set<std::string> cond;
    for(auto const& ion: neuron_block_.ions) {
        for(auto const& var: ion.write) {
            update_ion_symbols(var, accessKind::write, ion.name);
        }

        for(auto const& var: ion.read) {
            // Skip vars we have already processed as WRITE, since those can be read as well.
            if (std::count_if(ion.write.begin(),
                              ion.write.end(),
                              [&var](const auto& it) { return var.spelling == it.spelling; })) {
                continue;
            }
            if (!update_ion_symbols(var, accessKind::read, ion.name)) return false;
        }

        if(ion.uses_valence()) {
            Token valence_var = ion.valence_var;
            create_indexed_variable(valence_var.spelling, sourceKind::ion_valence,
                    accessKind::read, ion.name, valence_var.location);
        }
    }

    // then GLOBAL variables
    for(auto const& var : neuron_block_.globals) {
        if(!symbols_.count(var.spelling)) {
            error( yellow(var.spelling) +
                   " is declared as GLOBAL, but has not been declared in the" +
                   " ASSIGNED block",
                   var.location);
            return false;
        }
        auto& sym = symbols_[var.spelling];
        if(auto id = sym->is_variable()) {
            id->visibility(visibilityKind::global);
        }
        else if (!sym->is_indexed_variable()){
            throw compiler_exception(
                "unable to find symbol " + yellow(var.spelling) + " in symbols",
                Location());
        }
    }

    // then RANGE variables
    for(auto const& var : neuron_block_.ranges) {
        const auto& name = var.spelling;
        if(!symbols_.count(name)) {
            error( yellow(name) +
                   " is declared as RANGE, but has not been declared in the" +
                   " ASSIGNED or PARAMETER block",
                   var.location);
            return false;
        }
        if (is_ion_var(name)) {
            warning( yellow(name) + " is declared as RANGE, but is an ion variable", var.location);

        }
        if (is_builtin(name)) {
            warning( yellow(name) + " is declared as RANGE, but is a builtin variable", var.location);
        }
        auto& sym = symbols_[name];
        if(auto id = sym->is_variable()) {
            if (id->is_state()) {
                warning( yellow(name) + " is declared as RANGE, but is a STATE variable", var.location);
            }
            id->range(rangeKind::range);
        }
        else if (!sym->is_indexed_variable()){
            throw compiler_exception(
                "unable to find symbol " + yellow(name) + " in symbols",
                var.location);
        }
    }

    for (const Id& id: white_noise_block_) {
        create_white_noise(id.token);
    }
    return true;
}

int Module::semantic_func_proc() {
    ////////////////////////////////////////////////////////////////////////////
    // now iterate over the functions and procedures and perform semantic
    // analysis on each. This includes
    //  -   variable, function and procedure lookup
    //  -   generate local variable table for each function/procedure
    //  -   inlining function calls
    ////////////////////////////////////////////////////////////////////////////

    // Before, make sure there are no errors
    int errors = 0;
    for(auto& e : symbols_) {
        auto &s = e.second;
        if(s->kind() == symbolKind::procedure || s->kind() == symbolKind::function) {
            s->semantic(symbols_);
            ErrorVisitor v(source_name());
            s->accept(&v);
            errors += v.num_errors();
        }
    }
    if (errors > 0) {
        return errors;
    }

#ifdef LOGGING
    std::cout << white("===================================\n");
        std::cout << cyan("        Function Inlining\n");
        std::cout << white("===================================\n");
#endif
    for (auto& e: symbols_) {
        auto& s = e.second;
        if(s->kind() == symbolKind::procedure || s->kind() == symbolKind::function) {
            // perform semantic analysis
            s->semantic(symbols_);
#ifdef LOGGING
            std::cout << "\nfunction lowering for " << s->location() << "\n"
                      << s->to_string() << "\n\n";
#endif

            if (s->kind() == symbolKind::function) {
                auto rewritten = lower_functions(s->is_function()->body());
                s->is_function()->body(std::move(rewritten));
            } else {
                auto rewritten = lower_functions(s->is_procedure()->body());
                s->is_procedure()->body(std::move(rewritten));
            }
#ifdef LOGGING
            std::cout << "body after function lowering\n"
                      << s->to_string() << "\n\n";
#endif
        }
    }

    auto inline_and_simplify = [&](auto&& caller) {
        {
            auto rewritten = inline_function_calls(caller->name(), caller->body());
            caller->body(std::move(rewritten));
        }
        {
            auto rewritten = inline_procedure_calls(caller->name(), caller->body());
            caller->body(std::move(rewritten));
        }
        caller->body(constant_simplify(caller->body()));
    };

    // First, inline all function calls inside the bodies of each function
    // This catches recursions
    for(auto& e : symbols_) {
        auto& s = e.second;

        if (s->kind() == symbolKind::function) {
            // perform semantic analysis
            s->semantic(symbols_);
#ifdef LOGGING
            std::cout << "function inlining for " << s->location() << "\n"
                      << s->to_string() << "\n\n";
#endif
            inline_and_simplify(s->is_function());
            s->semantic(symbols_);
#ifdef LOGGING
            std::cout << "body after inlining\n"
                      << s->to_string() << "\n\n";
#endif
        }
    }

    // Once all functions are inlined internally; we can inline
    // function calls in the bodies of procedures
    for(auto& e : symbols_) {
        auto& s = e.second;

        if(s->kind() == symbolKind::procedure) {
            // perform semantic analysis
            s->semantic(symbols_);
#ifdef LOGGING
            std::cout << "function inlining for " << s->location() << "\n"
                      << s->to_string() << "\n\n";
#endif
            inline_and_simplify(s->is_procedure());
            s->semantic(symbols_);
#ifdef LOGGING
            std::cout << "body after inlining\n"
                      << s->to_string() << "\n\n";
#endif
        }
    }

    errors = 0;
    for(auto& e : symbols_) {
        auto& s = e.second;
        if(s->kind() == symbolKind::procedure) {
            s->semantic(symbols_);
            ErrorVisitor v(source_name());
            s->accept(&v);
            errors += v.num_errors();
        }
    }
    return errors;
}

void Module::check_revpot_mechanism() {
    int n_write_revpot = 0;
    for (auto& iondep: neuron_block_.ions) {
        if (iondep.writes_rev_potential()) ++n_write_revpot;
    }

    if (n_write_revpot==0) return;

    bool pure = true;

    // Are we marked as a point mechanism?
    if (kind()==moduleKind::point) {
        error("Cannot write reversal potential from a point mechanism.");
        return;
    }

    // Do we write any other ionic variables or a nonspecific current?
    for (auto& iondep: neuron_block_.ions) {
        pure &= !iondep.writes_concentration_int();
        pure &= !iondep.writes_concentration_ext();
        pure &= !iondep.writes_current();
    }
    pure &= !neuron_block_.has_nonspecific_current();

    if (!pure) {
        error("Cannot write reversal potential and also to other currents or ionic state.");
        return;
    }

    // Do we have any state variables?
    pure &= state_block_.state_variables.size()==0;
    if (!pure) {
        error("Cannot write reversal potential and also maintain state variables.");
        return;
    }

    kind_ = moduleKind::revpot;
}

void Module::check_voltage_mechanism() {
    if (kind_ != moduleKind::voltage) return;

    auto impure = neuron_block_.has_nonspecific_current();
    for (const auto& ion: neuron_block_.ions) {
        impure |= ion.writes_concentration_int();
        impure |= ion.writes_concentration_ext();
        impure |= ion.writes_current();
        impure |= ion.writes_rev_potential();
    }

    if (impure) {
        error("Voltage mechanisms may not write ionic quantities..");
        return;
    }
}
