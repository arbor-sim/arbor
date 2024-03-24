#pragma once

#include <any>
#include <string>

#include <arbor/arbexcept.hpp>
#include <arbor/network.hpp>
#include <arbor/util/expected.hpp>

#include <arbor/s_expr.hpp>
#include <arborio/export.hpp>

namespace arborio {

struct ARB_SYMBOL_VISIBLE network_parse_error: arb::arbor_exception {
    explicit network_parse_error(const std::string& msg, const arb::src_location& loc);
    explicit network_parse_error(const std::string& msg): arb::arbor_exception(msg) {}
};

template <typename T>
using parse_network_hopefully = arb::util::expected<T, network_parse_error>;

ARB_ARBORIO_API parse_network_hopefully<arb::network_selection> parse_network_selection_expression(
    const std::string& s);
ARB_ARBORIO_API parse_network_hopefully<arb::network_value> parse_network_value_expression(
    const std::string& s);

namespace literals {
inline arb::network_selection operator"" _ns(const char* s, std::size_t) {
    if (auto r = parse_network_selection_expression(s))
        return *r;
    else
        throw r.error();
}

inline arb::network_value operator"" _nv(const char* s, std::size_t) {
    if (auto r = parse_network_value_expression(s))
        return *r;
    else
        throw r.error();
}

}  // namespace literals
}  // namespace arborio
