#pragma once

#include <any>

#include <arbor/arbexcept.hpp>
#include <arbor/morph/region.hpp>
#include <arbor/s_expr.hpp>
#include <arbor/util/expected.hpp>

namespace arb {

struct label_parse_error: arb::arbor_exception {
    label_parse_error(const std::string& msg);
};

template <typename T>
using parse_hopefully = arb::util::expected<T, label_parse_error>;

parse_hopefully<std::any> parse_label_expression(const std::string&);
parse_hopefully<std::any> parse_label_expression(const s_expr&);

parse_hopefully<arb::region> parse_region_expression(const std::string& s);
parse_hopefully<arb::locset> parse_locset_expression(const std::string& s);

} // namespace arb
