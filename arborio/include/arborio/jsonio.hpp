#pragma once

#include <arbor/arbexcept.hpp>
#include <arbor/cable_cell.hpp>

namespace arborio {

struct jsonio_error: public arb::arbor_exception {
    jsonio_error(const std::string& msg);
};

// Load/store cable_cell_parameter_set and decor from/to file
arb::cable_cell_parameter_set load_cable_cell_parameter_set(std::string fname);
arb::decor load_decor(std::string fname);
void store_cable_cell_parameter_set(const arb::cable_cell_parameter_set& set, std::string fname);
void store_decor(const arb::decor& decor, std::string fname);

} // namespace arborio