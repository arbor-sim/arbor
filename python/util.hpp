#pragma once

#include <pybind11/pybind11.h>

#include "strprintf.hpp"

namespace pyarb {
namespace util {

namespace py = pybind11;

inline
std::string to_path(py::object fn) {
    if (py::isinstance<py::str>(fn)) {
        return std::string{py::str(fn)};
    }
    else if (py::isinstance(fn,
                            py::module_::import("pathlib").attr("Path"))) {
        return std::string{py::str(fn)};
    }
    throw std::runtime_error(
        util::strprintf("Cannot convert objects of type '{}' to a path-like.",
                        std::string{py::str(fn.get_type())}));
}

}
}
