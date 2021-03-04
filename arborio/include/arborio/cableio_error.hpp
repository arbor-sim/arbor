#pragma once

#include <ostream>

#include <arbor/morph/label_parse.hpp>

namespace arborio {
struct cableio_parse_error: arb::arbor_exception {
    explicit cableio_parse_error(const std::string& msg, const arb::src_location& loc);
};
struct cableio_unexpected_symbol: cableio_parse_error {
    explicit cableio_unexpected_symbol(const std::string& sym, const arb::src_location& loc);
};

} // namespace arb
