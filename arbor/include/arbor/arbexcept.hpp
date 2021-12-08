#pragma once

#include <any>
#include <stdexcept>
#include <string>

#include <arbor/common_types.hpp>

// Arbor-specific exception hierarchy.

namespace arb {

// Arbor internal logic error (if these are thrown,
// there is a bug in the library.)

struct arbor_internal_error: std::logic_error {
    arbor_internal_error(const std::string& what_arg):
        std::logic_error(what_arg)
    {}
};


// Common base-class for arbor run-time errors.

struct arbor_exception: std::runtime_error {
    arbor_exception(const std::string& what_arg):
        std::runtime_error(what_arg)
    {}
};

// Recipe errors:

struct bad_cell_probe: arbor_exception {
    bad_cell_probe(cell_kind kind, cell_gid_type gid);
    cell_gid_type gid;
    cell_kind kind;
};

struct bad_cell_description: arbor_exception {
    bad_cell_description(cell_kind kind, cell_gid_type gid);
    cell_gid_type gid;
    cell_kind kind;
};

struct bad_connection_source_gid: arbor_exception {
    bad_connection_source_gid(cell_gid_type gid, cell_gid_type src_gid, cell_size_type num_cells);
    cell_gid_type gid, src_gid;
    cell_size_type num_cells;
};

struct bad_connection_label: arbor_exception {
    bad_connection_label(cell_gid_type gid, const cell_tag_type& label, const std::string& msg);
    cell_gid_type gid;
    cell_tag_type label;
};

struct bad_global_property: arbor_exception {
    explicit bad_global_property(cell_kind kind);
    cell_kind kind;
};

struct bad_probe_id: arbor_exception {
    explicit bad_probe_id(cell_member_type id);
    cell_member_type probe_id;
};

struct gj_kind_mismatch: arbor_exception {
    gj_kind_mismatch(cell_gid_type gid_0, cell_gid_type gid_1);
    cell_gid_type gid_0, gid_1;
};

struct gj_unsupported_lid_selection_policy: arbor_exception {
    gj_unsupported_lid_selection_policy(cell_gid_type gid, cell_tag_type label);
    cell_gid_type gid;
    cell_tag_type label;
};

// Domain decomposition errors:

struct dom_dec_invalid_gj_cell_group: arbor_exception {
    dom_dec_invalid_gj_cell_group(cell_gid_type gid_0, cell_gid_type gid_1);
    cell_gid_type gid_0, gid_1;
};

struct dom_dec_invalid_num_domains: arbor_exception {
    dom_dec_invalid_num_domains(int domains_wrong, int domains_right);
    int domains_wrong, domains_right;
};

struct dom_dec_invalid_domain_id: arbor_exception {
    dom_dec_invalid_domain_id(int id_wrong, int id_right);
    int id_wrong, id_right;
};

struct dom_dec_invalid_num_local_cells: arbor_exception {
    dom_dec_invalid_num_local_cells(int rank, unsigned lc_wrong, unsigned lc_right);
    int rank;
    unsigned lc_wrong, lc_right;
};

struct dom_dec_invalid_num_global_cells: arbor_exception {
    dom_dec_invalid_num_global_cells(unsigned gc_wrong, unsigned gc_right);
    unsigned gc_wrong, gc_right;
};

struct dom_dec_invalid_sum_local_cells: arbor_exception {
    dom_dec_invalid_sum_local_cells(unsigned gc_wrong, unsigned gc_right);
    unsigned gc_wrong, gc_right;
};

struct dom_dec_duplicate_gid: arbor_exception {
    dom_dec_duplicate_gid(cell_gid_type gid);
    cell_gid_type gid;
};

struct dom_dec_non_existent_rank: arbor_exception {
    dom_dec_non_existent_rank(cell_gid_type gid, int rank);
    cell_gid_type gid;
    int rank;
};

struct dom_dec_out_of_bounds: arbor_exception {
    dom_dec_out_of_bounds(cell_gid_type gid, unsigned num_cells);
    cell_gid_type gid;
    unsigned num_cells;
};

struct dom_dec_invalid_backend: arbor_exception {
    dom_dec_invalid_backend(int rank);
    int rank;
};

struct dom_dec_incompatible_backend: arbor_exception {
    dom_dec_incompatible_backend(int rank, cell_kind kind);
    int rank;
    cell_kind kind;
};

// Simulation errors:

struct bad_event_time: arbor_exception {
    explicit bad_event_time(time_type event_time, time_type sim_time);
    time_type event_time;
    time_type sim_time;
};

// Mechanism catalogue errors:

struct no_such_mechanism: arbor_exception {
    explicit no_such_mechanism(const std::string& mech_name);
    std::string mech_name;
};

struct duplicate_mechanism: arbor_exception {
    explicit duplicate_mechanism(const std::string& mech_name);
    std::string mech_name;
};

struct fingerprint_mismatch: arbor_exception {
    explicit fingerprint_mismatch(const std::string& mech_name);
    std::string mech_name;
};

struct no_such_parameter: arbor_exception {
    no_such_parameter(const std::string& mech_name, const std::string& param_name);
    std::string mech_name;
    std::string param_name;
};

struct invalid_parameter_value: arbor_exception {
    invalid_parameter_value(const std::string& mech_name, const std::string& param_name, const std::string& value_str);
    invalid_parameter_value(const std::string& mech_name, const std::string& param_name, double value);
    std::string mech_name;
    std::string param_name;
    std::string value_str;
    double value;
};

struct invalid_ion_remap: arbor_exception {
    explicit invalid_ion_remap(const std::string& mech_name);
    invalid_ion_remap(const std::string& mech_name, const std::string& from_ion, const std::string& to_ion);
    std::string from_ion;
    std::string to_ion;
};

struct no_such_implementation: arbor_exception {
    explicit no_such_implementation(const std::string& mech_name);
    std::string mech_name;
};

// Run-time value bounds check:

struct range_check_failure: arbor_exception {
    explicit range_check_failure(const std::string& whatstr, double value);
    double value;
};

struct file_not_found_error: arbor_exception {
    file_not_found_error(const std::string& fn);
    std::string filename;
};

//
struct bad_catalogue_error: arbor_exception {
    bad_catalogue_error(const std::string&);
    bad_catalogue_error(const std::string&, const std::any&);
    std::any platform_error;
};

// ABI errors

struct bad_alignment: arbor_exception {
    bad_alignment(size_t);
    size_t alignment;
};

struct unsupported_abi_error: arbor_exception {
    unsupported_abi_error(size_t);
    size_t version;
};

} // namespace arb
