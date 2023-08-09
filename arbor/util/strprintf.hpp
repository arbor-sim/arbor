#pragma once

// printf-like routines that return std::string.
//
// TODO: Consolidate with a single routine that provides a consistent interface
// along the lines of the PO645R2 text formatting proposal.

#include <cstdio>
#include <memory>
#include <string>
#include <sstream>
#include <system_error>
#include <type_traits>
#include <utility>
#include <vector>

#include "util/meta.hpp"

#include <fmt/format.h>
#include <fmt/compile.h>
#include <fmt/ranges.h>
#include <fmt/std.h>

namespace arb {
namespace util {

// Use ADL to_string or std::to_string, falling back to ostream formatting:

namespace impl_to_string {
    using std::to_string;

    template <typename T, typename = void>
    struct select {
        static std::string str(const T& value) {
            std::ostringstream o;
            o << value;
            return o.str();
        }
    };

    template <typename T>
    struct select<T, std::void_t<decltype(to_string(std::declval<T>()))>> {
        static std::string str(const T& v) {
            return to_string(v);
        }
    };
}

template <typename T>
std::string to_string(const T& value) {
    return impl_to_string::select<T>::str(value);
}

} // namespace util
} // namespace arb
