#pragma once

#include <string>
#include <vector>

#include "blocks.hpp"
#include "error.hpp"
#include "expression.hpp"
#include "writeback.hpp"

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

// TODO: are const and non-const methods necessary? check usage.
    NeuronBlock &      neuron_block() {return neuron_block_;}
    NeuronBlock const& neuron_block() const {return neuron_block_;}

    StateBlock &       state_block()  {return state_block_;}
    StateBlock const&  state_block()  const {return state_block_;}

    UnitsBlock &       units_block()  {return units_block_;}
    UnitsBlock const&  units_block()  const {return units_block_;}

    ParameterBlock &       parameter_block()        {return parameter_block_;}
    ParameterBlock const&  parameter_block()  const {return parameter_block_;}

    AssignedBlock &       assigned_block()        {return assigned_block_;}
    AssignedBlock const&  assigned_block()  const {return assigned_block_;}

    void neuron_block(const NeuronBlock& n) { neuron_block_ = n; }
    void state_block(const StateBlock& s) { state_block_ = s; }
    void units_block(const UnitsBlock& u) { units_block_ = u; }
    void parameter_block(const ParameterBlock& p) { parameter_block_ = p; }
    void assigned_block(const AssignedBlock& a) { assigned_block_ = a; }

    // access to the AST
    std::vector<symbol_ptr>& procedures() { return procedures_; }
    const std::vector<symbol_ptr>& procedures() const { return procedures_; }

    std::vector<symbol_ptr>& functions() { return functions_; }
    const std::vector<symbol_ptr>& functions() const { return functions_; }

    symbol_map& symbols() { return symbols_; }
    const symbol_map& symbols() const { return symbols_; }

    // error handling
    using error_stack::error;
    void error(std::string const& msg, Location loc = Location{}) {
        error({msg, loc});
    }

    std::string error_string() const;

    // warnings
    using error_stack::warning;
    void warning(std::string const& msg, Location loc = Location{}) {
        warning({msg, loc});
    }

    std::string warning_string() const;

    moduleKind kind() const { return kind_; }
    void kind(moduleKind k) { kind_ = k; }

    // perform semantic analysis
    void add_variables_to_symbols();
    bool semantic();

    const std::vector<WriteBack>& write_backs() const {
        return write_backs_;
    }

    auto find_ion(ionKind k) -> decltype(neuron_block().ions.begin()) {
        auto& ions = neuron_block().ions;
        return std::find_if(
            ions.begin(), ions.end(),
            [k](IonDep const& d) {return d.kind()==k;}
        );
    };

    bool has_ion(ionKind k) {
        return find_ion(k) != neuron_block().ions.end();
    };


private:
    moduleKind kind_;
    std::string title_;
    std::string module_name_;
    std::string source_name_;
    std::vector<char> buffer_; // character buffer loaded from file

    bool generate_initial_api();
    bool generate_current_api();
    bool generate_state_api();

    // AST storage
    std::vector<symbol_ptr> procedures_;
    std::vector<symbol_ptr> functions_;

    // hash table for lookup of variable and call names
    symbol_map symbols_;

    /// tests if symbol is defined
    bool has_symbol(const std::string& name) {
        return symbols_.find(name) != symbols_.end();
    }
    /// tests if symbol is defined
    bool has_symbol(const std::string& name, symbolKind kind) {
        auto s = symbols_.find(name);
        return s == symbols_.end() ? false : s->second->kind() == kind;
    }

    // Perform semantic analysis on functions and procedures.
    // Returns the number of errors that were encountered.
    int semantic_func_proc();

    // blocks
    NeuronBlock neuron_block_;
    StateBlock  state_block_;
    UnitsBlock  units_block_;
    ParameterBlock parameter_block_;
    AssignedBlock assigned_block_;

    std::vector<WriteBack> write_backs_;
};
