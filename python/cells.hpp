#pragma once

#include <pybind11/pybind11.h>

#include <arbor/cable_cell_param.hpp>
#include <arbor/mechcat.hpp>
#include <arbor/util/unique_any.hpp>

namespace pyarb {
arb::util::unique_any convert_cell(pybind11::object o);
std::any convert_gprop(pybind11::object o);
}
