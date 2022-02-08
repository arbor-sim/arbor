#pragma once

#include <any>
#include <stdexcept>
#include <string>

#include <arbor/common_types.hpp>
#include <arbor/export.hpp>

// Arbor-specific exception hierarchy.

namespace arb {

// Arbor internal logic error (if these are thrown,
// there is a bug in the library.)

struct ARB_ARBOR_API arbor_internal_error: std::logic_error {
    arbor_internal_error(const std::string& what_arg):
        std::logic_error(what_arg)
    {}
};


// Common base-class for arbor run-time errors.

struct ARB_ARBOR_API arbor_exception: std::runtime_error {
    arbor_exception(const std::string& what_arg):
        std::runtime_error(what_arg)
    {}
};

// Logic errors

// Argument violates domain constraints, eg ln(-1)
struct domain_error: arbor_exception {
    domain_error(const std::string&);
};

// Recipe errors:

struct ARB_ARBOR_API bad_cell_probe: arbor_exception {
    bad_cell_probe(cell_kind kind, cell_gid_type gid);
    cell_gid_type gid;
    cell_kind kind;
};

struct ARB_ARBOR_API bad_cell_description: arbor_exception {
    bad_cell_description(cell_kind kind, cell_gid_type gid);
    cell_gid_type gid;
    cell_kind kind;
};

struct ARB_ARBOR_API bad_connection_source_gid: arbor_exception {
    bad_connection_source_gid(cell_gid_type gid, cell_gid_type src_gid, cell_size_type num_cells);
    cell_gid_type gid, src_gid;
    cell_size_type num_cells;
};

struct ARB_ARBOR_API bad_connection_label: arbor_exception {
    bad_connection_label(cell_gid_type gid, const cell_tag_type& label, const std::string& msg);
    cell_gid_type gid;
    cell_tag_type label;
};

struct ARB_ARBOR_API bad_global_property: arbor_exception {
    explicit bad_global_property(cell_kind kind);
    cell_kind kind;
};

struct ARB_ARBOR_API bad_probe_id: arbor_exception {
    explicit bad_probe_id(cell_member_type id);
    cell_member_type probe_id;
};

struct ARB_ARBOR_API gj_kind_mismatch: arbor_exception {
    gj_kind_mismatch(cell_gid_type gid_0, cell_gid_type gid_1);
    cell_gid_type gid_0, gid_1;
};

struct ARB_ARBOR_API gj_unsupported_lid_selection_policy: arbor_exception {
    gj_unsupported_lid_selection_policy(cell_gid_type gid, cell_tag_type label);
    cell_gid_type gid;
    cell_tag_type label;
};

// Context errors:

struct ARB_ARBOR_API zero_thread_requested_error: arbor_exception {
    zero_thread_requested_error(unsigned nbt);
    unsigned nbt;
};

// Domain decomposition errors:

struct ARB_ARBOR_API gj_unsupported_domain_decomposition: arbor_exception {
    gj_unsupported_domain_decomposition(cell_gid_type gid_0, cell_gid_type gid_1);
    cell_gid_type gid_0, gid_1;
};

// Simulation errors:

struct ARB_ARBOR_API bad_event_time: arbor_exception {
    explicit bad_event_time(time_type event_time, time_type sim_time);
    time_type event_time;
    time_type sim_time;
};

// Mechanism catalogue errors:

struct ARB_ARBOR_API no_such_mechanism: arbor_exception {
    explicit no_such_mechanism(const std::string& mech_name);
    std::string mech_name;
};

struct ARB_ARBOR_API duplicate_mechanism: arbor_exception {
    explicit duplicate_mechanism(const std::string& mech_name);
    std::string mech_name;
};

struct ARB_ARBOR_API fingerprint_mismatch: arbor_exception {
    explicit fingerprint_mismatch(const std::string& mech_name);
    std::string mech_name;
};

struct ARB_ARBOR_API no_such_parameter: arbor_exception {
    no_such_parameter(const std::string& mech_name, const std::string& param_name);
    std::string mech_name;
    std::string param_name;
};

struct ARB_ARBOR_API invalid_parameter_value: arbor_exception {
    invalid_parameter_value(const std::string& mech_name, const std::string& param_name, const std::string& value_str);
    invalid_parameter_value(const std::string& mech_name, const std::string& param_name, double value);
    std::string mech_name;
    std::string param_name;
    std::string value_str;
    double value;
};

struct ARB_ARBOR_API invalid_ion_remap: arbor_exception {
    explicit invalid_ion_remap(const std::string& mech_name);
    invalid_ion_remap(const std::string& mech_name, const std::string& from_ion, const std::string& to_ion);
    std::string from_ion;
    std::string to_ion;
};

struct ARB_ARBOR_API no_such_implementation: arbor_exception {
    explicit no_such_implementation(const std::string& mech_name);
    std::string mech_name;
};

// Run-time value bounds check:

struct ARB_ARBOR_API range_check_failure: arbor_exception {
    explicit range_check_failure(const std::string& whatstr, double value);
    double value;
};

struct ARB_ARBOR_API file_not_found_error: arbor_exception {
    file_not_found_error(const std::string& fn);
    std::string filename;
};

//
struct ARB_ARBOR_API bad_catalogue_error: arbor_exception {
    bad_catalogue_error(const std::string&);
    bad_catalogue_error(const std::string&, const std::any&);
    std::any platform_error;
};

// ABI errors

struct ARB_ARBOR_API bad_alignment: arbor_exception {
    bad_alignment(size_t);
    size_t alignment;
};

struct ARB_ARBOR_API unsupported_abi_error: arbor_exception {
    unsupported_abi_error(size_t);
    size_t version;
};

} // namespace arb
