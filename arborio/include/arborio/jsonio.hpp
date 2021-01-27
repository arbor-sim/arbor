#pragma once

#include <iostream>

#include <arbor/arbexcept.hpp>
#include <arbor/cable_cell.hpp>

#include <nlohmann/json.hpp>

#define JSONIO_VERSION 1

namespace arborio {

struct jsonio_error: public arb::arbor_exception {
    jsonio_error(const std::string& msg);
};

// Input in JSON not used
struct jsonio_unused_input: jsonio_error {
    explicit jsonio_unused_input(const std::string& key);
};

// Error loading decor global parameters
struct jsonio_decor_global_load_error: jsonio_error {
    explicit jsonio_decor_global_load_error(const std::string& err);
};

// Error setting decor global parameters
struct jsonio_decor_global_set_error: jsonio_error {
    explicit jsonio_decor_global_set_error(const std::string& err);
};

// Missing region label in decor local parameters
struct jsonio_decor_local_missing_region: jsonio_error {
    explicit jsonio_decor_local_missing_region();
};

// Cannot set regional revpot method in decor local parameters
struct jsonio_decor_local_revpot_mech: jsonio_error {
    explicit jsonio_decor_local_revpot_mech();
};

// Error loading decor local parameters
struct jsonio_decor_local_load_error: jsonio_error {
    explicit jsonio_decor_local_load_error(const std::string& err);
};

// Error setting decor local parameters
struct jsonio_decor_local_set_error: jsonio_error {
    explicit jsonio_decor_local_set_error(const std::string& err);
};

// Missing region label in mechanism desc
struct jsonio_decor_mech_missing_region: jsonio_error {
    explicit jsonio_decor_mech_missing_region();
};

// Missing mechanism name in mechanism desc
struct jsonio_decor_mech_missing_name: jsonio_error {
    explicit jsonio_decor_mech_missing_name();
};

// Error painting mechanism on region
struct jsonio_decor_mech_set_error: jsonio_error {
    explicit jsonio_decor_mech_set_error(const std::string& reg, const std::string& mech, const std::string& err);
};

struct jsonio_missing_field: jsonio_error {
    explicit jsonio_missing_field(const std::string& field);
};

struct jsonio_version_error: jsonio_error {
    explicit jsonio_version_error(const unsigned version);
};

struct jsonio_type_error: jsonio_error {
    explicit jsonio_type_error(const std::string& type);
};

// Load/store cable_cell_parameter_set and decor from/to stream
//arb::cable_cell_parameter_set load_cable_cell_parameter_set(std::istream&);
std::variant<arb::decor, arb::cable_cell_parameter_set> load_json(const nlohmann::json&);
nlohmann::json write_json(const arb::cable_cell_parameter_set&);
nlohmann::json write_json(const arb::decor&);

} // namespace arborio