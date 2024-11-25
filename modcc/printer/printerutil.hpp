#pragma once

// Convenience routines/helpers for source printers.

#include <ostream>
#include <string>
#include <vector>

#include "io/ostream_wrappers.hpp"
#include "blocks.hpp"
#include "error.hpp"
#include "expression.hpp"
#include "module.hpp"
#include <libmodcc/export.hpp>

ARB_LIBMODCC_API std::vector<std::string> namespace_components(const std::string& qualified_namespace);

// Can use this in a namespace. No __ allowed anywhere, neither _[A-Z], and in _global namespace_ _ followed by anything is verboten.
const static std::string pp_var_pfx = "_pp_var_";

inline const char* arb_header_prefix() {
    static const char* prefix = "arbor/";
    return prefix;
}

// TODO: this function will be obsoleted once arbor private/public headers are
// properly split.

inline const char* arb_private_header_prefix() {
    static const char* prefix = "";
    return prefix;
}

struct namespace_declaration_open {
    const std::vector<std::string>& ids;
    namespace_declaration_open(const std::vector<std::string>& ids): ids(ids) {}

    friend std::ostream& operator<<(std::ostream& o, const namespace_declaration_open& n) {
        for (auto& id: n.ids) {
            o << "namespace " << id << " {\n";
        }
        return o;
    }
};

struct namespace_declaration_close {
    const std::vector<std::string>& ids;
    namespace_declaration_close(const std::vector<std::string>& ids): ids(ids) {}

    friend std::ostream& operator<<(std::ostream& o, const namespace_declaration_close& n) {
        for (auto i = n.ids.rbegin(); i!=n.ids.rend(); ++i) {
            o << "} // namespace " << *i << "\n";
        }
        return o;
    }
};

// Enum representation:

inline const char* module_kind_str(const Module& m) {
    switch (m.kind()) {
    case moduleKind::density:   return "arb_mechanism_kind_density";
    case moduleKind::voltage:   return "arb_mechanism_kind_voltage";
    case moduleKind::point:     return "arb_mechanism_kind_point";
    case moduleKind::revpot:    return "arb_mechanism_kind_reversal_potential";
    case moduleKind::junction:  return "arb_mechanism_kind_gap_junction";
    default: throw compiler_exception("Unknown module kind " + std::to_string((int)m.kind()));
    }
}

// Check expression non-null and scoped, or else throw.

inline void assert_has_scope(Expression* expr, const std::string& context) {
    return
        !expr? throw compiler_exception("missing expression for "+context):
        !expr->scope()? throw compiler_exception("printer invoked before semantic pass for "+context):
        void();
}


// Scope query functions:

// All local variables in scope with `is_indexed()` true.
ARB_LIBMODCC_API std::vector<LocalVariable*> indexed_locals(scope_ptr scope);

// All local variables in scope with `is_arg()` and `is_indexed()` false.
ARB_LIBMODCC_API std::vector<LocalVariable*> pure_locals(scope_ptr scope);


// Module state query functions:

// Normal (not API, net_receive) procedures in module:

ARB_LIBMODCC_API std::vector<ProcedureExpression*> normal_procedures(const Module&);

struct public_variable_ids_t {
    std::vector<Id> state_ids;
    std::vector<Id> global_parameter_ids;
    std::vector<Id> range_parameter_ids;
    std::vector<std::pair<Id,unsigned int>> white_noise_ids;
};

// Public module variables by role.

ARB_LIBMODCC_API public_variable_ids_t public_variable_ids(const Module&);

struct module_variables_t {
    std::vector<VariableExpression*> scalars;
    std::vector<VariableExpression*> arrays;
};

// Scalar and array variables with local linkage.

ARB_LIBMODCC_API module_variables_t local_module_variables(const Module&);

// "normal" procedures in a module.
// A normal procedure is one that has been declared with the
// PROCEDURE keyword in NMODL.

ARB_LIBMODCC_API std::vector<ProcedureExpression*> module_normal_procedures(const Module& m);

// Extract key procedures from module.

ARB_LIBMODCC_API APIMethod* find_api_method(const Module& m, const char* which);

ARB_LIBMODCC_API NetReceiveExpression* find_net_receive(const Module& m);

ARB_LIBMODCC_API PostEventExpression* find_post_event(const Module& m);

// For generating vectorized code for reading and writing data sources.
// node: The data source uses the CV index which is categorized into
//       one of four constraints to optimize memory accesses.
// cell: The data source uses the cell index, which is in turn indexed
//       according to the CV index.
// other: The data source is indexed according to some other index.
//        Vector optimizations should be skipped.
// none: The data source is scalar.
enum class index_kind {
    node,
    cell,
    other,
    none
};

struct ARB_LIBMODCC_API indexed_variable_info {
    std::string data_var;
    std::string node_index_var;
    std::string cell_index_var;
    std::string other_index_var;
    index_kind  index_var_kind;

    bool accumulate = true; // true => add with weight_ factor on assignment
    bool readonly = false;  // true => can never be assigned to by a mechanism
    bool additive = false;  // only additive contributions allowed?
    bool always_use_weight = false; // can disable weighting?

    // Scale is the conversion factor from the data variable
    // to the NMODL value.
    double scale = 1;
    bool scalar() const;
    std::string inner_index_var() const;
    std::string outer_index_var() const;
};

ARB_LIBMODCC_API indexed_variable_info decode_indexed_variable(IndexedVariable* sym);

template<typename C>
size_t emit_array(std::ostream& out, const C& vars) {
    auto n = 0ul;
    io::separator sep("", ", ");
    out << "{ ";
    for (const auto& var: vars) {
        out << sep << var;
        ++n;
    }
    out << " }";
    return n;
}
