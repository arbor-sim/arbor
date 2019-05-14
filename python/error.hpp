#pragma once

#include <stdexcept>
#include <string>

#include <arbor/util/optional.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

// from https://pybind11.readthedocs.io/en/stable/advanced/cast/stl.html?highlight=boost%3A%3Aoptional#c-17-library-containers
namespace pybind11 { namespace detail {
    template <typename T>
    struct type_caster<arb::util::optional<T>> : optional_caster<arb::util::optional<T>> {};
}}

namespace pyarb {

// Python wrapper errors

struct pyarb_error: std::runtime_error {
    pyarb_error(const std::string& what_msg):
        std::runtime_error(what_msg) {}
};

static
void assert_throw(bool pred, const char* msg) {
    if (!pred) throw pyarb_error(msg);
}

} // namespace pyarb
