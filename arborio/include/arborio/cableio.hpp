#pragma once

#include <ostream>

#include <arbor/cable_cell.hpp>

namespace arborio {

struct format_parse_error: arb::arbor_exception {
    format_parse_error(const std::string& msg);
};

template <typename T>
using parse_hopefully = arb::util::expected<T, format_parse_error>;

std::ostream& write_s_expr(std::ostream&, const arb::label_dict&);
std::ostream& write_s_expr(std::ostream&, const arb::decor&);
std::ostream& write_s_expr(std::ostream&, const arb::morphology&);
std::ostream& write_s_expr(std::ostream&, const arb::cable_cell&);

parse_hopefully<arb::label_dict> parse_label_dict(const std::string&);

} // namespace arb
