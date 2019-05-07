#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

#include "error.hpp"

namespace pyarb {

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
