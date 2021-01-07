#pragma once

#include <iomanip>

#include <arbor/cable_cell.hpp>

namespace arborio {

arb::cable_cell_parameter_set load_cable_cell_parameter_set(std::string fname);
arb::decor load_decor(std::string fname);
void write_cable_cell_parameter_set(const arb::cable_cell_parameter_set& set, std::string fname);
void write_decor(const arb::decor& decor, std::string fname);

} // namespace arborio