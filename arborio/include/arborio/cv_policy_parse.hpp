#pragma once

#include <any>
#include <string>

#include <arbor/cv_policy.hpp>
#include <arbor/arbexcept.hpp>
#include <arbor/util/expected.hpp>
#include <arbor/s_expr.hpp>
#include <arborio/export.hpp>

namespace arborio {

struct ARB_SYMBOL_VISIBLE cv_policy_parse_error: arb::arbor_exception {
    explicit cv_policy_parse_error(const std::string& msg, const arb::src_location& loc);
    explicit cv_policy_parse_error(const std::string& msg);
};

using parse_cv_policy_hopefully = arb::util::expected<arb::cv_policy, cv_policy_parse_error>;

ARB_ARBORIO_API parse_cv_policy_hopefully parse_cv_policy_expression(const std::string& s);
ARB_ARBORIO_API parse_cv_policy_hopefully parse_cv_policy_expression(const arb::s_expr& s);

namespace literals {

inline
arb::cv_policy operator "" _cvp(const char* s, std::size_t) {
    if (auto r = parse_cv_policy_expression(s)) return *r;
    else throw r.error();
}

} // namespace literals

} // namespace arborio
