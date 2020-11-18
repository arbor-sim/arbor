#pragma once

#include <pybind11/pybind11.h>

#include <arbor/cable_cell_param.hpp>
#include <arbor/mechcat.hpp>
#include <arbor/util/unique_any.hpp>

namespace pyarb {
arb::util::unique_any convert_cell(pybind11::object o);

struct global_props_shim {
    arb::mechanism_catalogue cat;
    arb::cable_cell_global_properties props;
    global_props_shim();
};

}
