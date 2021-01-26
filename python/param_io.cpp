#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <fstream>
#include <iomanip>

#include <arbor/cable_cell.hpp>
#include <arbor/morph/primitives.hpp>
#include "arbor/util/any_visitor.hpp"

#include <arborio/jsonio.hpp>

#include "error.hpp"
#include "strprintf.hpp"

namespace pyarb {

std::variant<arb::decor, arb::cable_cell_parameter_set> load_json(const std::string& fname) {
    std::ifstream fid{fname};
    if (!fid.good()) {
        throw pyarb_error("Can't open file '{}'" + fname);
    }

    std::variant<arb::decor, arb::cable_cell_parameter_set> params;
    try {
        return params = arborio::load_json(fid);
    }
    catch (std::exception& e) {
        throw pyarb_error("Error while trying to load from \"" + fname + "\": " + std::string(e.what()));
    }
};

template <typename T>
void store_json(const T& set, const std::string& fname) {
    std::ofstream fid(fname);
    try {
        return arborio::store_json(set, fid);
    }
    catch (std::exception& e) {
        throw pyarb_error("Error writing \"" + fname + "\": " + std::string(e.what()));
    }
}

void register_param_loader(pybind11::module& m) {
    m.def("load_json",
        &load_json,
        pybind11::arg_v("filename", "the name of the file."),
        "Load decor or cable_parameter_set from file.");

    m.def("store_json",
        [](const arb::cable_cell_parameter_set& set, std::string fname) {
            return store_json<arb::cable_cell_parameter_set>(set, fname);
        },
        pybind11::arg_v("object", "the cable_parameter_set or decor object."),
        pybind11::arg_v("filename", "the name of the file."),
        "Write decor or cable_parameter_set to file.");

    m.def("store_json",
          [](const arb::decor& decor, std::string fname) {
            return store_json<arb::decor>(decor, fname);
          },
          pybind11::arg_v("object", "the cable_parameter_set or decor object."),
          pybind11::arg_v("filename", "the name of the file."),
          "Write decor or cable_parameter_set to file.");

    // arb::cable_cell_ion_data
    pybind11::class_<arb::cable_cell_ion_data> cable_ion_data(m, "cable_ion_data");
    cable_ion_data
        .def_readwrite("int_con", &arb::cable_cell_ion_data::init_int_concentration, "initial internal concentration [mM].")
        .def_readwrite("ext_con", &arb::cable_cell_ion_data::init_ext_concentration, "initial external concentration [mM].")
        .def_readwrite("rev_pot", &arb::cable_cell_ion_data::init_reversal_potential, "initial reversal potential [mV].")
        .def("__repr__", [](const arb::cable_cell_ion_data& set) {
          return util::pprintf("(cable_ion_data (int_con {}, ext_con {}, rev_pot {}) )",
                             set.init_int_concentration, set.init_ext_concentration, set.init_reversal_potential);})
        .def("__str__", [](const arb::cable_cell_ion_data& set) {
          return util::pprintf("(cable_ion_data (int_con {}, ext_con {}, rev_pot {}) )",
                               set.init_int_concentration, set.init_ext_concentration, set.init_reversal_potential);});

    // arb::cable_cell_parameter_set
    pybind11::class_<arb::cable_cell_parameter_set> cable_parameter_set(m, "cable_parameter_set");
    cable_parameter_set
        .def(pybind11::init<const arb::cable_cell_parameter_set&>())
        .def_readwrite("Vm", &arb::cable_cell_parameter_set::init_membrane_potential, "initial membrane voltage [mV].")
        .def_readwrite("cm", &arb::cable_cell_parameter_set::membrane_capacitance, "membrane capacitance [F/m²].")
        .def_readwrite("rL", &arb::cable_cell_parameter_set::axial_resistivity, "axial resistivity [Ω·cm].")
        .def_readwrite("tempK", &arb::cable_cell_parameter_set::temperature_K, "temperature [Kelvin].")
        .def_readwrite("ion_data", &arb::cable_cell_parameter_set::ion_data, "dictionary of ion string to initial internal "
                                                                             "and external concentrations [mM] and reversal potential [mV]")
        .def_readwrite("method", &arb::cable_cell_parameter_set::reversal_potential_method, "dictionary of ion string to reversal potential method")
        .def_readwrite("discretization", &arb::cable_cell_parameter_set::discretization, "the cv policy")
        .def("__repr__", [](const arb::cable_cell_parameter_set& s) { return util::pprintf("<arbor.cable_parameter_set>"); })
        .def("__str__", [](const arb::cable_cell_parameter_set& s) { return util::pprintf("(cable_parameter_set)"); });
}
} //namespace pyarb