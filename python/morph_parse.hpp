#pragma once

#include <any>

#include "error.hpp"
#include "s_expr.hpp"

namespace pyarb {

struct parse_error_state {
    std::string message;
    int location;
};

template <typename T>
using parse_hopefully = hopefully<T, parse_error_state>;

parse_hopefully<std::any> eval(const s_expr&);

} // namespace pyarb
