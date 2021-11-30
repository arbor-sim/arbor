#pragma once

#include <optional>

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include "error.hpp"

namespace pyarb {

struct is_nonneg {
    template<typename T>
    constexpr bool operator()(const T& v) {
        return v>=T(0);
    }
};

struct is_positive {
    template<typename T>
    constexpr
    bool operator()(const T& v) {
        return v>T(0);
    }
};

// A helper function for converting from a Python object to a C++ optional wrapper.
// Throws an runtime_error exception with msg if either the Python object
// can't be converted to type T, or if the predicate is false for the value.
template <typename T, typename F>
std::optional<T> py2optional(pybind11::object o, const char* msg, F&& pred) {
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

    return o.is_none()? std::nullopt: std::optional<T>(std::move(value));
}

template <typename T>
std::optional<T> py2optional(pybind11::object o, const char* msg) {
    T value;

    if (!o.is_none()) {
        try {
            value = o.cast<T>();
        }
        catch (...) {
            throw pyarb_error(msg);
        }
    }

    return o.is_none()? std::nullopt: std::optional<T>(std::move(value));
}

// Attempt to cast a Python object to a C++ type T.
// Returns an optional that is set if o could be cast
// to T, otherwise it is empty. Hence not being able
// to cast is not an error.
template <typename T>
std::optional<T> try_cast(pybind11::object o) {
    if (o.is_none()) return std::nullopt;

    try {
        return o.cast<T>();
    }
    // Ignore cast_error: if unable to perform cast.
    catch (pybind11::cast_error& e) {}

    return std::nullopt;
}

} // namespace arb
