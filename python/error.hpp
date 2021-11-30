#pragma once

#include <mutex>
#include <stdexcept>
#include <string>

#include <pybind11/pybind11.h>

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

} // namespace pyarb
