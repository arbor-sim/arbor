#pragma once

#include <stdexcept>
#include <string>

#include <arbor/arbexcept.hpp>
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

using arb::arbor_exception;

// Python wrapper errors

struct python_error: arbor_exception {
    explicit python_error(const std::string& message);
};

template <typename T, typename F>
T&& assert_predicate(T&& t, F&& f, const char* msg) {
    if (!f(t)) throw std::runtime_error(msg);
    return std::forward<T>(t);
}

} // namespace pyarb
