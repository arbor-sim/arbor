#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include <arbor/util/optional.hpp>

#include "error.hpp"

// from https://pybind11.readthedocs.io/en/stable/advanced/cast/stl.html?highlight=boost%3A%3Aoptional#c-17-library-containers
namespace pybind11 { namespace detail {
    template <typename T>
    struct type_caster<arb::util::optional<T>>: optional_caster<arb::util::optional<T>> {};
}}

namespace pyarb {

struct is_nonneg {
    template<typename T>
    constexpr
    bool operator()(const T& v) {
        return v>=T(0);
    }
};

// A helper function for converting from a Python object to a C++ optional wrapper.
// Throws an runtime_error exception with msg if either the Python object
// can't be converted to type T, or if the predicate is false for the value.
template <typename T, typename F>
arb::util::optional<T> py2optional(pybind11::object o, const char* msg, F&& pred) {
    bool ok = true;
    T value;

    if (!o.is_none()) {
        try {
            value = o.cast<T>();
            ok = pred(value);
        }
        catch (...) {
            ok = false;
        }
    }

    if (!ok) {
        throw pyarb_error(msg);
    }

    return o.is_none()? arb::util::nullopt: arb::util::optional<T>(std::move(value));
}

template <typename T>
arb::util::optional<T> py2optional(pybind11::object o, const char* msg) {
    T value;

    if (!o.is_none()) {
        try {
            value = o.cast<T>();
        }
        catch (...) {
            throw pyarb_error(msg);
        }
    }

    return o.is_none()? arb::util::nullopt: arb::util::optional<T>(std::move(value));
}

}
