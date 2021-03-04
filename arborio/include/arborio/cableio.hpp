#pragma once

#include <arbor/cable_cell.hpp>
#include <arbor/s_expr.hpp>

#include <arborio/cableio_error.hpp>

namespace arborio {

template <typename T>
using parse_hopefully = arb::util::expected<T, cableio_parse_error>;
using cable_cell_component = std::variant<arb::morphology, arb::label_dict, arb::decor, arb::cable_cell>;

std::ostream& write_s_expr(std::ostream&, const arb::label_dict&);
std::ostream& write_s_expr(std::ostream&, const arb::decor&);
std::ostream& write_s_expr(std::ostream&, const arb::morphology&);
std::ostream& write_s_expr(std::ostream&, const arb::cable_cell&);

parse_hopefully<cable_cell_component> parse_component(const std::string&);

} // namespace arborio