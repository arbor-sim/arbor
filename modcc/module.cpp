#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <set>

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

    static ionKind is_ion_update(Expression* e) {
        if(auto a = e->is_assignment()) {
            if(auto sym = a->lhs()->is_identifier()->symbol()) {
                if(auto var = sym->is_local_variable()) {
                    return var->ion_channel();
                }
            }
        }
        return ionKind::none;
    }

    bool has_current_update_ = false;
    std::set<std::string> ion_current_vars_;

public:
    using BlockRewriterBase::visit;

    virtual void finalize() override {
        if (has_current_update_) {
            // Initialize current_ as first statement.
            statements_.push_front(make_expression<AssignmentExpression>(loc_,
                    id("current_"),
                    make_expression<NumberExpression>(loc_, 0.0)));

            statements_.push_back(make_expression<AssignmentExpression>(loc_,
                id("current_"),
                make_expression<MulBinaryExpression>(loc_,
                    id("weights_"),
                    id("current_"))));

            for (auto& v: ion_current_vars_) {
                statements_.push_back(make_expression<AssignmentExpression>(loc_,
                    id(v),
                    make_expression<MulBinaryExpression>(loc_,
                        id("weights_"),
                        id(v))));
            }
        }
    }

    virtual void visit(SolveExpression *e) override {}
    virtual void visit(ConductanceExpression *e) override {}
    virtual void visit(AssignmentExpression *e) override {
        statements_.push_back(e->clone());
        auto loc = e->location();

        auto update_kind = is_ion_update(e);
        if (update_kind!=ionKind::none) {
            if (update_kind!=ionKind::nonspecific) {
                ion_current_vars_.insert(e->lhs()->is_identifier()->name());
            }
            has_current_update_ = true;

            if (!linear_test(e->rhs(), {"v"}).is_linear) {
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
    functions_.push_back(
        Parser{"FUNCTION exp_pade_11(z) { exp_pade_11=(1+0.5*z)/(1-0.5*z) }"}.parse_function());
    functions_.push_back(
        Parser{
            "FUNCTION exp_pade_22(z)"
            "{ exp_pade_22=(1+0.5*z+0.08333333333333333*z*z)/(1-0.5*z+0.08333333333333333*z*z) }"
        }.parse_function());

    // move functions and procedures to the symbol table
    if(!move_symbols(functions_))  return false;
    if(!move_symbols(procedures_)) return false;

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
                solver = make_unique<CnexpSolverVisitor>();
                break;
            case solverMethod::sparse:
                solver = make_unique<SparseSolverVisitor>();
                break;
            case solverMethod::none:
                solver = make_unique<DirectSolverVisitor>();
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

    // Perform semantic analysis and inlining on function and procedure bodies
    // in order to inline calls inside the newly crated API methods.
    semantic_func_proc();

    return !has_error();
}

/// populate the symbol table with class scope variables
void Module::add_variables_to_symbols() {
    // add reserved symbols (not v, because for some reason it has to be added
    // by the user)
    auto create_variable = [this] (const char* name, rangeKind rng, accessKind acc) {
        auto t = new VariableExpression(Location(), name);
        t->state(false);
        t->linkage(linkageKind::local);
        t->ion_channel(ionKind::none);
        t->range(rng);
        t->access(acc);
        t->visibility(visibilityKind::global);
        symbols_[name] = symbol_ptr{t};
    };

    // mechanisms use a vector of weights to:
    //  density mechs:
    //      - convert current densities from 10.A.m^-2 to A.m^-2
    //      - density or proportion of a CV's area affected by the mechansim
    //  point procs:
    //      - convert current in nA to current densities in A.m^-2
    create_variable("weights_", rangeKind::range, accessKind::read);

    // add indexed variables to the table
    auto create_indexed_variable = [this]
        (std::string const& name, std::string const& indexed_name,
         tok op, accessKind acc, ionKind ch, Location loc)
    {
        if(symbols_.count(name)) {
            throw compiler_exception(
                pprintf("the symbol % already exists", yellow(name)),
                loc);
        }
        symbols_[name] =
            make_symbol<IndexedVariable>(loc, name, indexed_name, acc, op, ch);
    };

    create_indexed_variable("current_", "vec_i", tok::plus,
                            accessKind::write, ionKind::none, Location());
    create_indexed_variable("v", "vec_v", tok::eq,
                            accessKind::read,  ionKind::none, Location());
    create_indexed_variable("dt", "vec_dt", tok::eq,
                            accessKind::read,  ionKind::none, Location());

    // add cell-indexed variables to the table
    auto create_cell_indexed_variable = [this]
        (std::string const& name, std::string const& indexed_name, Location loc = Location())
    {
        if(symbols_.count(name)) {
            throw compiler_exception(
                "trying to insert a symbol that already exists",
                loc);
        }
        symbols_[name] = make_symbol<CellIndexedVariable>(loc, name, indexed_name);
    };

    create_cell_indexed_variable("t", "vec_t");
    create_cell_indexed_variable("t_to", "vec_t_to");

    // add state variables
    for(auto const &var : state_block()) {
        VariableExpression *id = new VariableExpression(Location(), var.name());

        id->state(true);    // set state to true
        // state variables are private
        //      what about if the state variables is an ion concentration?
        id->linkage(linkageKind::local);
        id->visibility(visibilityKind::local);
        id->ion_channel(ionKind::none);    // no ion channel
        id->range(rangeKind::range);       // always a range
        id->access(accessKind::readwrite);

        symbols_[var.name()] = symbol_ptr{id};
    }

    // add the parameters
    for(auto const& var : parameter_block()) {
        auto name = var.name();
        if(name == "v") { // global voltage values
            // ignore voltage, which is added as an indexed variable by default
            continue;
        }
        VariableExpression *id = new VariableExpression(Location(), name);

        id->state(false);           // never a state variable
        id->linkage(linkageKind::local);
        // parameters are visible to Neuron
        id->visibility(visibilityKind::global);
        id->ion_channel(ionKind::none);
        // scalar by default, may later be upgraded to range
        id->range(rangeKind::scalar);
        id->access(accessKind::read);

        // check for 'special' variables
        if(name == "celcius") { // global celcius parameter
            id->linkage(linkageKind::external);
        }

        // set default value if one was specified
        if(var.value.size()) {
            id->value(std::stod(var.value));
        }

        symbols_[name] = symbol_ptr{id};
    }

    // add the assigned variables
    for(auto const& var : assigned_block()) {
        auto name = var.name();
        if(name == "v") { // global voltage values
            // ignore voltage, which is added as an indexed variable by default
            continue;
        }
        VariableExpression *id = new VariableExpression(var.token.location, name);

        id->state(false);           // never a state variable
        id->linkage(linkageKind::local);
        // local visibility by default
        id->visibility(visibilityKind::local);
        id->ion_channel(ionKind::none); // can change later
        // ranges because these are assigned to in loop
        id->range(rangeKind::range);
        id->access(accessKind::readwrite);

        symbols_[name] = symbol_ptr{id};
    }

    ////////////////////////////////////////////////////
    // parse the NEURON block data, and use it to update
    // the variables in symbols_
    ////////////////////////////////////////////////////
    // first the ION channels
    // add ion channel variables
    auto update_ion_symbols = [this, create_indexed_variable]
            (Token const& tkn, accessKind acc, ionKind channel)
    {
        auto const& name = tkn.spelling;

        if(has_symbol(name)) {
            auto sym = symbols_[name].get();

            //  if sym is an indexed_variable: error
            //  else if sym is a state variable: register a writeback call
            //  else if sym is a range (non parameter) variable: error
            //  else if sym is a parameter variable: error
            //  else it does not exist so make an indexed variable

            // If an indexed variable has already been created with the same name
            // throw an error.
            if(sym->kind()==symbolKind::indexed_variable) {
                error(pprintf("the symbol defined % at % can't be redeclared",
                              sym->location(), yellow(name)),
                      tkn.location);
                return;
            }
            else if(sym->kind()==symbolKind::variable) {
                auto var = sym->is_variable();

                // state variable: register writeback
                if(var->is_state()) {
                    // create writeback
                    write_backs_.push_back(WriteBack(name, "ion_"+name, channel));
                    return;
                }

                // error: a normal range variable or parameter can't have the same
                // name as an indexed ion variable
                error(pprintf("the ion channel variable % at % can't be redeclared",
                              yellow(name), sym->location()),
                      tkn.location);
                return;
            }
        }

        // add the ion variable's indexed shadow
        create_indexed_variable(name, "ion_"+name,
                                acc==accessKind::read ? tok::eq : tok::plus,
acc, channel, tkn.location);
    };

    // check for nonspecific current
    if( neuron_block().has_nonspecific_current() ) {
        auto const& i = neuron_block().nonspecific_current;
        update_ion_symbols(i, accessKind::write, ionKind::nonspecific);
    }


    for(auto const& ion : neuron_block().ions) {
        for(auto const& var : ion.read) {
            update_ion_symbols(var, accessKind::read, ion.kind());
        }
        for(auto const& var : ion.write) {
            update_ion_symbols(var, accessKind::write, ion.kind());
        }
    }

    // then GLOBAL variables
    for(auto const& var : neuron_block().globals) {
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
    for(auto const& var : neuron_block().ranges) {
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
