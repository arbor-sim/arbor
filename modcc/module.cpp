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
#include "kineticrewriter.hpp"
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
    std::set<std::string> ion_current_vars_;

public:
    using BlockRewriterBase::visit;

    virtual void finalize() override {
        if (has_current_update_) {
            // Initialize conductivity_ as first statement.
            statements_.push_front(make_expression<AssignmentExpression>(loc_,
                    id("conductivity_"),
                    make_expression<NumberExpression>(loc_, 0.0)));
            statements_.push_front(make_expression<AssignmentExpression>(loc_,
                    id("current_"),
                    make_expression<NumberExpression>(loc_, 0.0)));
        }
    }

    virtual void visit(SolveExpression *e) override {}
    virtual void visit(ConductanceExpression *e) override {}
    virtual void visit(AssignmentExpression *e) override {
        statements_.push_back(e->clone());
        auto loc = e->location();

        sourceKind current_source = current_update(e);
        if (current_source != sourceKind::no_source) {
            has_current_update_ = true;

            if (current_source==sourceKind::ion_current_density || current_source==sourceKind::ion_current) {
                ion_current_vars_.insert(e->lhs()->is_identifier()->name());
            }
            else {
                // A 'nonspecific' current contribution.
                // Remove data source; currents accumulated into `current_` instead.

                e->lhs()->is_identifier()->symbol()->is_local_variable()
                    ->external_variable()->data_source(sourceKind::no_source);
            }

            linear_test_result L = linear_test(e->rhs(), {"v"});
            if (!L.is_linear) {
                error({"current update expressions must be linear in v: "+e->rhs()->to_string(),
                       e->location()});
                return;
            }
            else {
                statements_.push_back(make_expression<AssignmentExpression>(loc,
                    id("current_", loc),
                    make_expression<AddBinaryExpression>(loc,
                        id("current_", loc),
                        e->lhs()->clone())));
                if (L.coef.count("v")) {
                    statements_.push_back(make_expression<AssignmentExpression>(loc,
                        id("conductivity_", loc),
                        make_expression<AddBinaryExpression>(loc,
                            id("conductivity_", loc),
                            L.coef.at("v")->clone())));
                }
            }
        }
    }
};

std::string Module::error_string() const {
    std::string str;
    for (const error_entry& entry: errors()) {
        if (!str.empty()) str += '\n';
        str += red("error   ");
        str += white(pprintf("%:% ", source_name(), entry.location));
        str += entry.message;
    }
    return str;
}

std::string Module::warning_string() const {
    std::string str;
    for (auto& entry: warnings()) {
        if (!str.empty()) str += '\n';
        str += purple("warning   ");
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

    // first add variables defined in the NEURON, ASSIGNED and PARAMETER
    // blocks these symbols have "global" scope, i.e. they are visible to all
    // functions and procedurs in the mechanism
    add_variables_to_symbols();

    // Helper which iterates over a vector of Symbols, moving them into the
    // symbol table.
    // Returns false if a symbol name clashes with the name of a symbol that
    // is already in the symbol table.
    bool linear = true;
    std::vector<std::string> state_vars;
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

    // perform semantic analysis and inlining on function and procedure bodies
    if(auto errors = semantic_func_proc()) {
        error("There were "+std::to_string(errors)+" errors in the semantic analysis");
        return false;
    }

    // All API methods are generated from statements in one of the special procedures
    // defined in NMODL, e.g. the nrn_init() API call is based on the INITIAL block.
    // When creating an API method, the first task is to look up the source procedure,
    // i.e. the INITIAL block for nrn_init(). This lambda takes care of this repetative
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

        symbols_[name] = make_symbol<APIMethod>(
                          loc, name,
                          std::vector<expression_ptr>(), // no arguments
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
    // nrn_init : based on the INITIAL block (i.e. the 'initial' procedure
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
    auto initial_api = make_empty_api_method("nrn_init", "initial");
    auto api_init  = initial_api.first;
    auto proc_init = initial_api.second;
    auto& init_body = api_init->body()->statements();

    for(auto& e : *proc_init->body()) {
        init_body.emplace_back(e->clone());
    }

    api_init->semantic(symbols_);

    // Look in the symbol table for a procedure with the name "breakpoint".
    // This symbol corresponds to the BREAKPOINT block in the .mod file
    // There are two APIMethods generated from BREAKPOINT.
    // The first is nrn_state, which is the first case handled below.
    // The second is nrn_current, which is handled after this block
    auto state_api  = make_empty_api_method("nrn_state", "breakpoint");
    auto api_state  = state_api.first;
    auto breakpoint = state_api.second; // implies we are building the `nrn_state()` method.

    api_state->semantic(symbols_);
    scope_ptr nrn_state_scope = api_state->scope();

    if(breakpoint) {
        // Grab SOLVE statements, put them in `nrn_state` after translation.
        bool found_solve = false;
        bool found_non_solve = false;
        std::set<std::string> solved_ids;

        for(auto& e: (breakpoint->body()->statements())) {
            SolveExpression* solve_expression = e->is_solve_statement();
            if(!solve_expression) {
                found_non_solve = true;
                continue;
            }
            if(found_non_solve) {
                error("SOLVE statements must come first in BREAKPOINT block",
                    e->location());
                return false;
            }

            found_solve = true;
            std::unique_ptr<SolverVisitorBase> solver;

            switch(solve_expression->method()) {
            case solverMethod::cnexp:
                solver = std::make_unique<CnexpSolverVisitor>();
                break;
            case solverMethod::sparse:
                solver = std::make_unique<SparseSolverVisitor>();
                break;
            case solverMethod::none:
                solver = std::make_unique<DirectSolverVisitor>();
                break;
            }

            // If the derivative block is a kinetic block, perform kinetic
            // rewrite first.

            auto deriv = solve_expression->procedure();

            if (deriv->kind()==procedureKind::kinetic) {
                kinetic_rewrite(deriv->body())->accept(solver.get());
            }
            else {
                deriv->body()->accept(solver.get());
                for (auto& s: deriv->body()->statements()) {
                    if(s->is_assignment() && !state_vars.empty()) {
                        linear_test_result r = linear_test(s->is_assignment()->rhs(), state_vars);
                        linear &= r.is_linear;
                        linear &= r.is_homogeneous;
                    }
                }
            }

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
                solve_block->semantic(nrn_state_scope);
                solve_block = remove_unused_locals(solve_block->is_block());

                // Copy body into nrn_state.
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

        // handle the case where there is no SOLVE in BREAKPOINT
        if(!found_solve) {
            warning(" there is no SOLVE statement, required to update the"
                    " state variables, in the BREAKPOINT block",
                    breakpoint->location());
        }
        else {
            // Redo semantic pass in order to elimate any removed local symbols.
            api_state->semantic(symbols_);
        }

        // Run remove locals pass again on the whole body in case `dt` was never used.
        api_state->body(remove_unused_locals(api_state->body()));
        api_state->semantic(symbols_);

        //..........................................................
        // nrn_current : update contributions to currents
        //..........................................................
        NrnCurrentRewriter nrn_current_rewriter;
        breakpoint->accept(&nrn_current_rewriter);

        for (auto& s: breakpoint->body()->statements()) {
            if(s->is_assignment() && !state_vars.empty()) {
                linear_test_result r = linear_test(s->is_assignment()->rhs(), state_vars);
                linear &= r.is_linear;
                linear &= r.is_homogeneous;
            }
        }

        auto nrn_current_block = nrn_current_rewriter.as_block();
        if (!nrn_current_block) {
            append_errors(nrn_current_rewriter.errors());
            return false;
        }

        symbols_["nrn_current"] =
            make_symbol<APIMethod>(
                    breakpoint->location(), "nrn_current",
                    std::vector<expression_ptr>(),
                    constant_simplify(nrn_current_block));
        symbols_["nrn_current"]->semantic(symbols_);
    }
    else {
        error("a BREAKPOINT block is required");
        return false;
    }

    if (has_symbol("net_receive", symbolKind::procedure)) {
        auto net_rec_api = make_empty_api_method("net_rec_api", "net_receive");
        if (net_rec_api.second) {
            for (auto &s: net_rec_api.second->body()->statements()) {
                if (s->is_assignment()) {
                    for (const auto &id: state_vars) {
                        auto coef = symbolic_pdiff(s->is_assignment()->rhs(), id);
                        if(coef->is_number()) {
                            if (!s->is_assignment()->lhs()->is_identifier()) {
                                error(pprintf("Left hand side of assignment is not an identifier"));
                                return false;
                            }
                            linear &= s->is_assignment()->lhs()->is_identifier()->name() == id ?
                                      coef->is_number()->value() == 1 :
                                      coef->is_number()->value() == 0;
                        }
                        else {
                            linear = false;
                        }
                    }
                }
            }
        }
    }
    linear_ = linear;

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

/// populate the symbol table with class scope variables
void Module::add_variables_to_symbols() {
    auto create_variable =
        [this](const Token& token, accessKind a, visibilityKind v, linkageKind l,
               rangeKind r, bool is_state = false) -> symbol_ptr&
        {
            auto var = new VariableExpression(token.location, token.spelling);
            var->access(a);
            var->visibility(v);
            var->linkage(l);
            var->range(r);
            var->state(is_state);
            return symbols_[var->name()] = symbol_ptr{var};
        };

    // add indexed variables to the table
    auto create_indexed_variable = [this]
        (std::string const& name, sourceKind data_source,
         accessKind acc, std::string ch, Location loc) -> symbol_ptr&
    {
        if(symbols_.count(name)) {
            throw compiler_exception(
                pprintf("the symbol % already exists", yellow(name)), loc);
        }
        return symbols_[name] =
            make_symbol<IndexedVariable>(loc, name, data_source, acc, ch);
    };

    sourceKind current_kind = kind_==moduleKind::point? sourceKind::current: sourceKind::current_density;
    sourceKind conductance_kind = kind_==moduleKind::point? sourceKind::conductance: sourceKind::conductivity;

    create_indexed_variable("current_", current_kind, accessKind::write, "", Location());
    create_indexed_variable("conductivity_", conductance_kind, accessKind::write, "", Location());
    create_indexed_variable("v", sourceKind::voltage, accessKind::read,  "", Location());
    create_indexed_variable("dt", sourceKind::dt, accessKind::read,  "", Location());

    // If we put back support for accessing cell time again from NMODL code,
    // add indexed_variable also for "time" with appropriate cell-index based
    // indirection in printers.

    // Add state variables.
    for (const Id& id: state_block_) {
        create_variable(id.token,
            accessKind::readwrite, visibilityKind::local, linkageKind::local, rangeKind::range, true);
    }

    // Add parameters, ignoring built-in voltage variable "v".
    for (const Id& id: parameter_block_) {
        if (id.name() == "v") {
            continue;
        }

        // Special case: 'celsius' is an external indexed-variable with a special
        // data source. Retrieval of value is handled especially by printers.

        if (id.name() == "celsius") {
            create_indexed_variable("celsius", sourceKind::temperature, accessKind::read, "", Location());
        }
        else {
            // Parameters are scalar by default, but may later be changed to range.
            auto& sym = create_variable(id.token,
                accessKind::read, visibilityKind::global, linkageKind::local, rangeKind::scalar);

            // Set default value if one was specified.
            if (id.has_value()) {
                sym->is_variable()->value(std::stod(id.value));
            }
        }
    }

    // Remove `celsius` from the parameter block, as it is not a true parameter anymore.
    parameter_block_.parameters.erase(
        std::remove_if(parameter_block_.begin(), parameter_block_.end(),
            [](const Id& id) { return id.name() == "celsius"; }),
        parameter_block_.end()
    );

    // Add 'assigned' variables, ignoring built-in voltage variable "v".
    for (const Id& id: assigned_block_) {
        if (id.name() == "v") {
            continue;
        }

        create_variable(id.token,
            accessKind::readwrite, visibilityKind::local, linkageKind::local, rangeKind::range);
    }

    ////////////////////////////////////////////////////
    // parse the NEURON block data, and use it to update
    // the variables in symbols_
    ////////////////////////////////////////////////////
    // first the ION channels
    // add ion channel variables
    auto update_ion_symbols = [this, create_indexed_variable]
            (Token const& tkn, accessKind acc, const std::string& channel)
    {
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
                return;
            }
            if (!state->is_state()) {
                error(pprintf("the symbol defined % at % can't be redeclared",
                    state->location(), yellow(name)), tkn.location);
                return;
            }
            name += "_shadowed_";
        }

        auto& sym = create_indexed_variable(name, data_source, acc, channel, tkn.location);

        if (state) {
            state->shadows(sym.get());
        }
    };

    // Nonspecific current variables are represented by an indexed variable
    // with a 'current' data source. Assignments in the NrnCurrent block will
    // later be rewritten so that these contributions are accumulated in `current_`
    // (potentially saving some weight multiplications); at that point the
    // data source for the nonspecific current variable will be reset to 'no_source'.

    if( neuron_block_.has_nonspecific_current() ) {
        auto const& i = neuron_block_.nonspecific_current;
        create_indexed_variable(i.spelling, sourceKind::current, accessKind::write, "", i.location);
    }

    for(auto const& ion : neuron_block_.ions) {
        for(auto const& var : ion.read) {
            update_ion_symbols(var, accessKind::read, ion.name);
        }
        for(auto const& var : ion.write) {
            update_ion_symbols(var, accessKind::write, ion.name);
        }

        if(ion.uses_valence()) {
            Token valence_var = ion.valence_var;
            create_indexed_variable(valence_var.spelling, sourceKind::ion_valence,
                    accessKind::read, ion.name, valence_var.location);
        }
    }

    // then GLOBAL variables
    for(auto const& var : neuron_block_.globals) {
        if(!symbols_[var.spelling]) {
            error( yellow(var.spelling) +
                   " is declared as GLOBAL, but has not been declared in the" +
                   " ASSIGNED block",
                   var.location);
            return;
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
        if(!symbols_[var.spelling]) {
            error( yellow(var.spelling) +
                   " is declared as RANGE, but has not been declared in the" +
                   " ASSIGNED or PARAMETER block",
                   var.location);
            return;
        }
        auto& sym = symbols_[var.spelling];
        if(auto id = sym->is_variable()) {
            id->range(rangeKind::range);
        }
        else if (!sym->is_indexed_variable()){
            throw compiler_exception(
                "unable to find symbol " + yellow(var.spelling) + " in symbols",
                var.location);
        }
    }
}

int Module::semantic_func_proc() {
    ////////////////////////////////////////////////////////////////////////////
    // now iterate over the functions and procedures and perform semantic
    // analysis on each. This includes
    //  -   variable, function and procedure lookup
    //  -   generate local variable table for each function/procedure
    //  -   inlining function calls
    ////////////////////////////////////////////////////////////////////////////
#ifdef LOGGING
    std::cout << white("===================================\n");
    std::cout << cyan("        Function Inlining\n");
    std::cout << white("===================================\n");
#endif
    int errors = 0;
    for(auto& e : symbols_) {
        auto& s = e.second;

        if(    s->kind() == symbolKind::function
            || s->kind() == symbolKind::procedure)
        {
#ifdef LOGGING
            std::cout << "\nfunction inlining for " << s->location() << "\n"
                      << s->to_string() << "\n"
                      << green("\n-call site lowering-\n\n");
#endif
            // first perform semantic analysis
            s->semantic(symbols_);

            // then use an error visitor to print out all the semantic errors
            ErrorVisitor v(source_name());
            s->accept(&v);
            errors += v.num_errors();

            // inline function calls
            // this requires that the symbol table has already been built
            if(v.num_errors()==0) {
                auto &b = s->kind()==symbolKind::function ?
                    s->is_function()->body()->statements() :
                    s->is_procedure()->body()->statements();

                // lower function call sites so that all function calls are of
                // the form : variable = call(<args>)
                // e.g.
                //      a = 2 + foo(2+x, y, 1)
                // becomes
                //      ll0_ = foo(2+x, y, 1)
                //      a = 2 + ll0_
                for(auto e=b.begin(); e!=b.end(); ++e) {
                    b.splice(e, lower_function_calls((*e).get()));
                }
#ifdef LOGGING
                std::cout << "body after call site lowering\n";
                for(auto& l : b) std::cout << "  " << l->to_string() << " @ " << l->location() << "\n";
                std::cout << green("\n-argument lowering-\n\n");
#endif

                // lower function arguments that are not identifiers or literals
                // e.g.
                //      ll0_ = foo(2+x, y, 1)
                //      a = 2 + ll0_
                // becomes
                //      ll1_ = 2+x
                //      ll0_ = foo(ll1_, y, 1)
                //      a = 2 + ll0_
                for(auto e=b.begin(); e!=b.end(); ++e) {
                    if(auto be = (*e)->is_binary()) {
                        // only apply to assignment expressions where rhs is a
                        // function call because the function call lowering step
                        // above ensures that all function calls are of this form
                        if(auto rhs = be->rhs()->is_function_call()) {
                            b.splice(e, lower_function_arguments(rhs->args()));
                        }
                    }
                }

#ifdef LOGGING
                std::cout << "body after argument lowering\n";
                for(auto& l : b) std::cout << "  " << l->to_string() << " @ " << l->location() << "\n";
                std::cout << green("\n-inlining-\n\n");
#endif

                // Do the inlining, which currently only works for functions
                // that have a single statement in their body
                // e.g. if the function foo in the examples above is defined as follows
                //
                //  function foo(a, b, c) {
                //      foo = a*(b + c)
                //  }
                //
                // the full inlined example is
                //      ll1_ = 2+x
                //      ll0_ = ll1_*(y + 1)
                //      a = 2 + ll0_
                for(auto e=b.begin(); e!=b.end(); ++e) {
                    if(auto ass = (*e)->is_assignment()) {
                        if(ass->rhs()->is_function_call()) {
                            ass->replace_rhs(inline_function_call(ass->rhs()));
                        }
                    }
                }

#ifdef LOGGING
                std::cout << "body after inlining\n";
                for(auto& l : b) std::cout << "  " << l->to_string() << " @ " << l->location() << "\n";
#endif
                // Finally, run a constant simplification pass.
                if (auto proc = s->is_procedure()) {
                    proc->body(constant_simplify(proc->body()));
                    s->semantic(symbols_);
                }
            }
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


