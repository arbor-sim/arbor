#pragma once

#include <fstream>

#include <pybind11/pybind11.h>

#include "strprintf.hpp"
#include "error.hpp"

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
        util::pprintf("Cannot convert objects of type {} to a path-like.",
                        std::string{py::str(fn.get_type())}));
}

inline
std::string read_file_or_buffer(py::object fn) {
    if (py::hasattr(fn, "read")) {
        return py::str(fn.attr("read")(-1));
    } else {
        const auto fname = util::to_path(fn);
        std::ifstream fid{fname};
        if (!fid.good()) {
            throw arb::file_not_found_error(fname);
        }
        std::string result;
        fid.seekg(0, fid.end);
        auto sz = fid.tellg();
        fid.seekg(0, fid.beg);
        result.resize(sz);
        fid.read(result.data(), sz);
        return result;
    }
}

}
}
