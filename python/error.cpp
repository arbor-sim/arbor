#include <pybind11/pybind11.h>

#include "error.hpp"

namespace pyarb {

std::exception_ptr py_exception;
std::mutex py_callback_mutex;

void py_reset_and_throw() {
    if (py_exception) {
        std::exception_ptr copy = py_exception;
        py_exception = nullptr;
        std::rethrow_exception(copy);
    }
}

} // namespace pyarb
