#pragma once

#include <any>

#include <arbor/arbexcept.hpp>
#include <arbor/cv_policy.hpp>
#include <arbor/s_expr.hpp>
#include <arbor/util/expected.hpp>

namespace arb {
namespace cv {

struct parse_error: arb::arbor_exception {
    parse_error(const std::string& msg);
};

template <typename T> using parse_hopefully = arb::util::expected<T, parse_error>;

parse_hopefully<cv_policy> parse_expression(const std::string& s);

} // namespace cv
} // namespace arb
