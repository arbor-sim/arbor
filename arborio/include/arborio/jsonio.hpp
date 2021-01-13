#pragma once

#include <iostream>

#include <arbor/arbexcept.hpp>
#include <arbor/cable_cell.hpp>

namespace arborio {

struct jsonio_error: public arb::arbor_exception {
    jsonio_error(const std::string& msg);
};

// Error parsing json
struct jsonio_json_parse_error: jsonio_error {
    explicit jsonio_json_parse_error(const std::string err);
};

// Input in JSON not used
struct jsonio_unused_input: jsonio_error {
    explicit jsonio_unused_input(const std::string& key);
};

// Error in loading decor global parameters
struct jsonio_decor_global_load_error: jsonio_error {
    explicit jsonio_decor_global_load_error(const std::string err);
};

// Error in setting decor global parameters
struct jsonio_decor_global_set_error: jsonio_error {
    explicit jsonio_decor_global_set_error(const std::string err);
};

// Missing region label in decor local parameters
struct jsonio_decor_local_missing_region: jsonio_error {
    explicit jsonio_decor_local_missing_region();
};

// Cannot set regional revpot method in decor local parameters
struct jsonio_decor_local_revpot_mech: jsonio_error {
    explicit jsonio_decor_local_revpot_mech(const std::string& reg, const std::string& ion, const std::string& mech);
};

// Error in loading decor local parameters
struct jsonio_decor_local_load_error: jsonio_error {
    explicit jsonio_decor_local_load_error(const std::string err);
};

// Error in setting decor local parameters
struct jsonio_decor_local_set_error: jsonio_error {
    explicit jsonio_decor_local_set_error(const std::string err);
};

// Missing region label in mechanism desc
struct jsonio_decor_mech_missing_region: jsonio_error {
    explicit jsonio_decor_mech_missing_region();
};

// Missing mechanism name in mechanism desc
struct jsonio_decor_mech_missing_name: jsonio_error {
    explicit jsonio_decor_mech_missing_name();
};

// Error in painting mechanism on region
struct jsonio_decor_mech_set_error: jsonio_error {
    explicit jsonio_decor_mech_set_error(const std::string& reg, const std::string& mech, const std::string& err);
};

// Load/store cable_cell_parameter_set and decor from/to file
arb::cable_cell_parameter_set load_cable_cell_parameter_set(std::istream&);
arb::decor load_decor(std::istream&);
void store_cable_cell_parameter_set(const arb::cable_cell_parameter_set&, std::ostream&);
void store_decor(const arb::decor&, std::ostream&);

} // namespace arborio