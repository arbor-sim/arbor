#pragma once

#include <string>
#include <vector>

#include "blocks.hpp"
#include "expression.hpp"

// wrapper around a .mod file
class Module {
public :
    using scope_type = Expression::scope_type;
    using symbol_map = scope_type::symbol_map;
    using symbol_ptr = scope_type::symbol_ptr;

    Module(std::string const& fname);
    Module(std::vector<char> const& buffer);

    std::vector<char> const& buffer() const {
        return buffer_;
    }

    std::string const& file_name()  const {return fname_;}
    std::string const& name()  const {return neuron_block_.name;}

    void               title(const std::string& t) {title_ = t;}
    std::string const& title() const          {return title_;}

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

    void neuron_block(NeuronBlock const &n) {neuron_block_ = n;}
    void state_block (StateBlock  const &s) {state_block_  = s;}
    void units_block (UnitsBlock  const &u) {units_block_  = u;}
    void parameter_block (ParameterBlock  const &p) {parameter_block_  = p;}
    void assigned_block (AssignedBlock  const &a) {assigned_block_  = a;}

    // access to the AST
    std::vector<symbol_ptr>&      procedures();
    std::vector<symbol_ptr>const& procedures() const;

    std::vector<symbol_ptr>&      functions();
    std::vector<symbol_ptr>const& functions() const;

    symbol_map &      symbols();
    symbol_map const& symbols() const;

    // error handling
    void error(std::string const& msg, Location loc);
    std::string const& error_string() {
        return error_string_;
    }

    lexerStatus status() const {
        return status_;
    }

    // warnings
    void warning(std::string const& msg, Location loc);
    bool has_warning() const {
        return has_warning_;
    }
    bool has_error() const {
        return status()==lexerStatus::error;
    }

    moduleKind kind() const {
        return kind_;
    }
    void kind(moduleKind k) {
        kind_ = k;
    }

    // perform semantic analysis
    void add_variables_to_symbols();
    bool semantic();
    bool optimize();
private :
    moduleKind kind_;
    std::string title_;
    std::string fname_;
    std::vector<char> buffer_; // character buffer loaded from file

    bool generate_initial_api();
    bool generate_current_api();
    bool generate_state_api();

    // error handling
    std::string error_string_;
    lexerStatus status_ = lexerStatus::happy;
    bool has_warning_ = false;

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

    // blocks
    NeuronBlock neuron_block_;
    StateBlock  state_block_;
    UnitsBlock  units_block_;
    ParameterBlock parameter_block_;
    AssignedBlock assigned_block_;
};
