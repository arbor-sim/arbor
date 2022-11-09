#pragma once

#include <any>
#include <stdexcept>
#include <string>

#include <arbor/common_types.hpp>
#include <arbor/export.hpp>
#include <arbor/mechanism_abi.h>

// Arbor-specific exception hierarchy.

namespace arb {

// Arbor internal logic error (if these are thrown,
// there is a bug in the library.)

struct ARB_SYMBOL_VISIBLE arbor_internal_error: std::logic_error {
    arbor_internal_error(const std::string&);
    std::string where;
};

// Common base-class for arbor run-time errors.

struct ARB_SYMBOL_VISIBLE arbor_exception: std::runtime_error {
    arbor_exception(const std::string&);
    std::string where;
};

// Logic errors

// Argument violates domain constraints, eg ln(-1)
struct ARB_SYMBOL_VISIBLE domain_error: arbor_exception {
    domain_error(const std::string&);
};

// Recipe errors:

struct ARB_SYMBOL_VISIBLE bad_cell_probe: arbor_exception {
    bad_cell_probe(cell_kind kind, cell_gid_type gid);
    cell_gid_type gid;
    cell_kind kind;
};

struct ARB_SYMBOL_VISIBLE invalid_mechanism_kind: arbor_exception {
    invalid_mechanism_kind(arb_mechanism_kind);
    arb_mechanism_kind kind;
};

struct ARB_SYMBOL_VISIBLE bad_cell_description: arbor_exception {
    bad_cell_description(cell_kind kind, cell_gid_type gid);
    cell_gid_type gid;
    cell_kind kind;
};

struct ARB_SYMBOL_VISIBLE bad_connection_source_gid: arbor_exception {
    bad_connection_source_gid(cell_gid_type gid, cell_gid_type src_gid, cell_size_type num_cells);
    cell_gid_type gid, src_gid;
    cell_size_type num_cells;
};

struct ARB_SYMBOL_VISIBLE bad_connection_label: arbor_exception {
    bad_connection_label(cell_gid_type gid, const cell_tag_type& label, const std::string& msg);
    cell_gid_type gid;
    cell_tag_type label;
};

struct ARB_SYMBOL_VISIBLE bad_global_property: arbor_exception {
    explicit bad_global_property(cell_kind kind);
    cell_kind kind;
};

struct ARB_SYMBOL_VISIBLE bad_probeset_id: arbor_exception {
    explicit bad_probeset_id(cell_member_type id);
    cell_member_type probeset_id;
};

struct ARB_SYMBOL_VISIBLE gj_kind_mismatch: arbor_exception {
    gj_kind_mismatch(cell_gid_type gid_0, cell_gid_type gid_1);
    cell_gid_type gid_0, gid_1;
};

struct ARB_SYMBOL_VISIBLE gj_unsupported_lid_selection_policy: arbor_exception {
    gj_unsupported_lid_selection_policy(cell_gid_type gid, cell_tag_type label);
    cell_gid_type gid;
    cell_tag_type label;
};

// Context errors:

struct ARB_SYMBOL_VISIBLE zero_thread_requested_error: arbor_exception {
    zero_thread_requested_error(unsigned nbt);
    unsigned nbt;
};

// Domain decomposition errors:

struct ARB_SYMBOL_VISIBLE gj_unsupported_domain_decomposition: arbor_exception {
    gj_unsupported_domain_decomposition(cell_gid_type gid_0, cell_gid_type gid_1);
    cell_gid_type gid_0, gid_1;
};

// Simulation errors:

struct ARB_SYMBOL_VISIBLE bad_event_time: arbor_exception {
    explicit bad_event_time(time_type event_time, time_type sim_time);
    time_type event_time;
    time_type sim_time;
};

// Mechanism catalogue errors:

struct ARB_SYMBOL_VISIBLE no_such_mechanism: arbor_exception {
    explicit no_such_mechanism(const std::string& mech_name);
    std::string mech_name;
};

struct ARB_SYMBOL_VISIBLE duplicate_mechanism: arbor_exception {
    explicit duplicate_mechanism(const std::string& mech_name);
    std::string mech_name;
};

struct ARB_SYMBOL_VISIBLE fingerprint_mismatch: arbor_exception {
    explicit fingerprint_mismatch(const std::string& mech_name);
    std::string mech_name;
};

struct ARB_SYMBOL_VISIBLE no_such_parameter: arbor_exception {
    no_such_parameter(const std::string& mech_name, const std::string& param_name);
    std::string mech_name;
    std::string param_name;
};

struct ARB_SYMBOL_VISIBLE illegal_diffusive_mechanism: arbor_exception {
    explicit illegal_diffusive_mechanism(const std::string& mech, const std::string& ion);
    std::string mech;
    std::string ion;
};

struct ARB_SYMBOL_VISIBLE invalid_parameter_value: arbor_exception {
    invalid_parameter_value(const std::string& mech_name, const std::string& param_name, const std::string& value_str);
    invalid_parameter_value(const std::string& mech_name, const std::string& param_name, double value);
    std::string mech_name;
    std::string param_name;
    std::string value_str;
    double value;
};

struct ARB_SYMBOL_VISIBLE invalid_ion_remap: arbor_exception {
    explicit invalid_ion_remap(const std::string& mech_name);
    invalid_ion_remap(const std::string& mech_name, const std::string& from_ion, const std::string& to_ion);
    std::string from_ion;
    std::string to_ion;
};

struct ARB_SYMBOL_VISIBLE no_such_implementation: arbor_exception {
    explicit no_such_implementation(const std::string& mech_name);
    std::string mech_name;
};

// Run-time value bounds check:

struct ARB_SYMBOL_VISIBLE range_check_failure: arbor_exception {
    explicit range_check_failure(const std::string& whatstr, double value);
    double value;
};

struct ARB_SYMBOL_VISIBLE file_not_found_error: arbor_exception {
    file_not_found_error(const std::string& fn);
    std::string filename;
};

//
struct ARB_SYMBOL_VISIBLE bad_catalogue_error: arbor_exception {
    bad_catalogue_error(const std::string&);
    bad_catalogue_error(const std::string&, const std::any&);
    std::any platform_error;
};

// ABI errors

struct ARB_SYMBOL_VISIBLE bad_alignment: arbor_exception {
    bad_alignment(size_t);
    size_t alignment;
};

struct ARB_SYMBOL_VISIBLE unsupported_abi_error: arbor_exception {
    unsupported_abi_error(size_t);
    size_t version;
};

} // namespace arb
