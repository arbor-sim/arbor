#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <fstream>
#include <iomanip>

#include <arbor/cable_cell.hpp>
#include <arbor/morph/primitives.hpp>
#include <sup/json_params.hpp>

#include <arborio/jsonio.hpp>

#include "error.hpp"
#include "strprintf.hpp"

namespace pyarb {
    void register_param_loader(pybind11::module& m) {
        m.def("load_default_parameters",
              [](std::string fname) {
                  auto params = arborio::load_cable_cell_parameter_set(fname);
                  arb::cable_cell_global_properties G;
                  G.default_parameters = params;
                  try {
                      arb::check_global_properties(G);
                  }
                  catch (std::exception& e) {
                      throw pyarb_error("Error loading default parameters: default parameter check failed : " + std::string(e.what()));
                  }
                  return params;
              },
              "Load default model parameters from file.");

        m.def("load_decor",
              [](std::string fname) {return arborio::load_decor(fname);},
              "Load decor from file.");

        m.def("store_default_parameters",
              [](const arb::cable_cell_parameter_set& set, std::string fname) {return arborio::store_cable_cell_parameter_set(set, fname);},
              "Write default model parameters to file.");

        m.def("store_decor",
              [](const arb::decor& decor, std::string fname) {return arborio::store_decor(decor, fname);},
              "Write decor to file.");

        // arb::cable_cell_parameter_set
        pybind11::class_<arb::cable_cell_parameter_set> cable_cell_parameter_set(m, "cable_cell_parameter_set");
        cable_cell_parameter_set
                .def("__repr__", [](const arb::cable_cell_parameter_set& s) { return util::pprintf("<arbor.cable_cell_parameter_set>"); })
                .def("__str__", [](const arb::cable_cell_parameter_set& s) { return util::pprintf("(cell_parameter_set)"); });
    }

} //namespace pyarb