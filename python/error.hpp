#pragma once

#include <mutex>
#include <stdexcept>
#include <string>

#include <pybind11/pybind11.h>

#include <arbor/arbexcept.hpp>
#include <arbor/util/either.hpp>

#include "strprintf.hpp"

namespace pyarb {

extern std::exception_ptr py_exception;
extern std::mutex py_callback_mutex;

// Python wrapper errors

struct pyarb_error: std::runtime_error {
    pyarb_error(const std::string& what_msg):
        std::runtime_error(what_msg) {}
    pyarb_error(const char* what_msg):
        std::runtime_error(what_msg) {}
};

inline
void assert_throw(bool pred, const char* msg) {
    if (!pred) throw pyarb_error(msg);
}

// This function resets a python exception to nullptr
// and rethrows a copy of the set python exception.
// It should be used in serial code
// just before handing control back to Python. 
void py_reset_and_throw();

template <typename L>
auto try_catch_pyexception(L func, const char* msg){
    std::lock_guard<std::mutex> g(py_callback_mutex);
    try {
        if(!py_exception) {
            return func();
        }
        else {
            throw pyarb_error(msg);
        }
    }
    catch (pybind11::error_already_set& e) {
        py_exception = std::current_exception();
        throw;
    }
}

template <typename T, typename E>
struct hopefully {
    using value_type = T;
    using error_type = E;
    arb::util::either<value_type, error_type> state;

    hopefully(const hopefully&) = default;

    hopefully(value_type x): state(std::move(x)) {}
    hopefully(error_type x): state(std::move(x)) {}

    const value_type& operator*() const {
        return try_get();
    }
    value_type& operator*() {
        return try_get();
    }
    const value_type* operator->() const {
        return &try_get();
    }
    value_type* operator->() {
        return &try_get();
    }

    operator bool() const {
        return (bool)state;
    }

    const error_type& error() const {
        try {
            return state.template get<1>();
        }
        catch(arb::util::either_invalid_access& e) {
            throw arb::arbor_internal_error(
                "Attempt to get an error from a valid hopefully wrapper.");
        }
    }

private:

    const value_type& try_get() const {
        try {
            return state.template get<0>();
        }
        catch(arb::util::either_invalid_access& e) {
            throw arbor_internal_error(util::pprintf(
                "Attempt to unwrap a hopefully with error state '{}'",
                error().message));
        }
    }
    value_type& try_get() {
        try {
            return state.template get<0>();
        }
        catch(arb::util::either_invalid_access& e) {
            throw arb::arbor_internal_error(util::pprintf(
                "Attempt to unwrap a hopefully with error state '{}'",
                error().message));
        }
    }
};

} // namespace pyarb
