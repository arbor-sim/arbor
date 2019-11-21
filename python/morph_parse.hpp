#pragma once

#include "error.hpp"
#include "s_expr.hpp"

#include <arbor/util/any.hpp>

namespace pyarb {

struct parse_error_state {
    std::string message;
    int location;
};

template <typename T>
using parse_hopefully = hopefully<T, parse_error_state>;

parse_hopefully<arb::util::any> eval(const s_expr&);

} // namespace pyarb
