#pragma once

#include <ostream>

#include <arbor/cable_cell.hpp>

namespace arb {

std::ostream& write_s_expr(std::ostream&, const cable_cell&);
std::ostream& write_s_expr(std::ostream&, const label_dict&);
std::ostream& write_s_expr(std::ostream&, const decor&);

} // namespace arb
