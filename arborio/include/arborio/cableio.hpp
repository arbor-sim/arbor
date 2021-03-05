#pragma once

#include <arbor/cable_cell.hpp>
#include <arbor/s_expr.hpp>

namespace arborio {

struct cableio_parse_error: arb::arbor_exception {
    explicit cableio_parse_error(const std::string& msg, const arb::src_location& loc);
};

template <typename T>
using parse_hopefully = arb::util::expected<T, cableio_parse_error>;
using cable_cell_component = std::variant<arb::morphology, arb::label_dict, arb::decor, arb::cable_cell>;

std::ostream& write_s_expr(std::ostream&, const arb::label_dict&);
std::ostream& write_s_expr(std::ostream&, const arb::decor&);
std::ostream& write_s_expr(std::ostream&, const arb::morphology&);
std::ostream& write_s_expr(std::ostream&, const arb::cable_cell&);

parse_hopefully<cable_cell_component> parse_component(const std::string&);

} // namespace arborio