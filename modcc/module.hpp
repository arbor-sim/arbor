#pragma once

#include <iterator>
#include <string>
#include <utility>
#include <vector>

#include "blocks.hpp"
#include "error.hpp"
#include "expression.hpp"

// wrapper around a .mod file
class Module: public error_stack {
public:
    using symbol_map = scope_type::symbol_map;
    using symbol_ptr = scope_type::symbol_ptr;

    template <typename Iter>
    Module(Iter b, Iter e, std::string source_name):
        source_name_(std::move(source_name))
    {
        buffer_.assign(b, e);
        buffer_.push_back('\0');
    }

    template <typename Container>
    explicit Module(const Container& text, std::string source_name):
        Module(std::begin(text), std::end(text), std::move(source_name)) {}

    std::vector<char> const& buffer() const {
        return buffer_;
    }

    bool empty() const {
        return buffer_.empty() || buffer_.front()=='\0';
    }

    std::string module_name() const {
        return module_name_.empty()? neuron_block_.name: module_name_;
    }
    void module_name(std::string name) { module_name_ = std::move(name); }

    const std::string& source_name() const { return source_name_; }

    void title(const std::string& t) { title_ = t; }
    const std::string& title() const { return title_; }

    moduleKind kind() const { return kind_; }
    void kind(moduleKind k) { kind_ = k; }

    // only used for ion access - this will be done differently ...
    NeuronBlock const& neuron_block() const {return neuron_block_;}

    // Retrieve list of state variable ids.
    StateBlock const&  state_block()  const {return state_block_;}

    // Retrieve list of parameter variable ids.
    ParameterBlock const&  parameter_block()  const {return parameter_block_;}

    // Retrieve list of ion dependencies.
    const std::vector<IonDep>& ion_deps() const { return neuron_block_.ions; }

    // Set top-level blocks (called from Parser).
    void neuron_block(const NeuronBlock& n) { neuron_block_ = n; }
    void state_block(const StateBlock& s) { state_block_ = s; }
    void units_block(const UnitsBlock& u) { units_block_ = u; }
    void parameter_block(const ParameterBlock& p) { parameter_block_ = p; }
    void assigned_block(const AssignedBlock& a) { assigned_block_ = a; }

    // Add global procedure or function, before semantic pass (called from Parser).
    void add_callable(symbol_ptr callable);

    // Raw access to AST data.
    const symbol_map& symbols() const { return symbols_; }

    // Error and warning handling.
    using error_stack::error;
    void error(std::string const& msg, Location loc = Location{}) {
        error({msg, loc});
    }

    std::string error_string() const;

    using error_stack::warning;
    void warning(std::string const& msg, Location loc = Location{}) {
        warning({msg, loc});
    }

    std::string warning_string() const;

    // Perform semantic analysis pass.
    bool semantic();

    auto find_ion(const std::string& ion_name) -> decltype(ion_deps().begin()) {
        auto& ions = neuron_block().ions;
        return std::find_if(
            ions.begin(), ions.end(),
            [&ion_name](IonDep const& d) {return d.name==ion_name;}
        );
    };

    bool has_ion(const std::string& ion_name) {
        return find_ion(ion_name) != neuron_block().ions.end();
    };

    bool is_linear() const { return linear_; }

private:
    moduleKind kind_;
    std::string title_;
    std::string module_name_;
    std::string source_name_;
    std::vector<char> buffer_; // Holds module source, zero terminated.

    NeuronBlock neuron_block_;
    StateBlock state_block_;
    UnitsBlock units_block_;
    ParameterBlock parameter_block_;
    AssignedBlock assigned_block_;
    bool linear_;

    // AST storage.
    std::vector<symbol_ptr> callables_;

    // Symbol name to symbol_ptr map.
    symbol_map symbols_;

    bool generate_initial_api();
    bool generate_current_api();
    bool generate_state_api();
    void add_variables_to_symbols();

    bool has_symbol(const std::string& name) {
        return symbols_.find(name) != symbols_.end();
    }

    bool has_symbol(const std::string& name, symbolKind kind) {
        auto s = symbols_.find(name);
        return s == symbols_.end() ? false : s->second->kind() == kind;
    }

    // Check requirements for reversal potential setters.
    void check_revpot_mechanism();

    // Perform semantic analysis on functions and procedures.
    // Returns the number of errors that were encountered.
    int semantic_func_proc();
};
