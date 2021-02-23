#pragma once

#include <ostream>

#include <arbor/cable_cell.hpp>
#include <arbor/morph/label_parse.hpp>

namespace arborio {

struct cableio_parse_error: arb::arbor_exception {
    explicit cableio_parse_error(const std::string& msg, const arb::src_location& loc);
};
struct cableio_unexpected_symbol: cableio_parse_error {
    explicit cableio_unexpected_symbol(const std::string& sym, const arb::src_location& loc);
};


template <typename T>
using parse_hopefully = arb::util::expected<T, cableio_parse_error>;

std::ostream& write_s_expr(std::ostream&, const arb::label_dict&);
std::ostream& write_s_expr(std::ostream&, const arb::decor&);
std::ostream& write_s_expr(std::ostream&, const arb::morphology&);
std::ostream& write_s_expr(std::ostream&, const arb::cable_cell&);

parse_hopefully<arb::decor> parse_decor(const std::string& str);
parse_hopefully<arb::label_dict> parse_label_dict(const std::string& str);
parse_hopefully<arb::morphology> parse_morphology(const std::string& str);
parse_hopefully<arb::cable_cell> parse_cable_cell(const std::string& str);

std::optional<std::variant<arb::morphology, arb::label_dict, arb::decor, arb::cable_cell>> parse_component(const arb::s_expr&);

} // namespace arb
