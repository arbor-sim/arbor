#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <fstream>
#include <iomanip>

#include <arbor/cable_cell.hpp>

#include <arborio/cableio.hpp>

#include "error.hpp"
#include "strprintf.hpp"

namespace pyarb {

arborio::cable_cell_component load_component(const std::string& fname) {
    std::ifstream fid{fname};
    if (!fid.good()) {
        throw pyarb_error("Can't open file '{}'" + fname);
    }
    auto component = arborio::parse_component(fid);
    if (!component) {
        throw pyarb_error("Error while trying to load component from \"" + fname + "\": " + component.error().what());
    }
    return component.value();
};

template<typename T>
void write_component(const T& component, const std::string& fname) {
    std::ofstream fid(fname);
    arborio::write_component(fid, component, arborio::meta_data{});
}

void write_component(const arborio::cable_cell_component& component, const std::string& fname) {
    std::ofstream fid(fname);
    arborio::write_component(fid, component);
}

void register_cable_loader(pybind11::module& m) {
    m.def("load_component",
          &load_component,
          pybind11::arg_v("filename", "the name of the file."),
          "Load arbor-component (decor, morphology, label_dict, cable_cell) from file.");

    m.def("write_component",
          [](const arborio::cable_cell_component& d, const std::string& fname) {
            return write_component(d, fname);
          },
          pybind11::arg_v("object", "the cable_component object."),
          pybind11::arg_v("filename", "the name of the file."),
          "Write cable_component to file.");

    m.def("write_component",
          [](const arb::decor& d, const std::string& fname) {
            return write_component<arb::decor>(d, fname);
          },
          pybind11::arg_v("object", "the decor object."),
          pybind11::arg_v("filename", "the name of the file."),
          "Write decor to file.");

    m.def("write_component",
          [](const arb::label_dict& d, const std::string& fname) {
            return write_component<arb::label_dict>(d, fname);
          },
          pybind11::arg_v("object", "the label_dict object."),
          pybind11::arg_v("filename", "the name of the file."),
          "Write label_dict to file.");

    m.def("write_component",
          [](const arb::morphology& d, const std::string& fname) {
            return write_component<arb::morphology>(d, fname);
          },
          pybind11::arg_v("object", "the morphology object."),
          pybind11::arg_v("filename", "the name of the file."),
          "Write morphology to file.");

    m.def("write_component",
          [](const arb::cable_cell& d, const std::string& fname) {
            return write_component<arb::cable_cell>(d, fname);
          },
          pybind11::arg_v("object", "the cable_cell object."),
          pybind11::arg_v("filename", "the name of the file."),
          "Write cable_cell to file.");

    // arborio::meta_data
    pybind11::class_<arborio::meta_data> component_meta_data(m, "component_meta_data");
    component_meta_data
        .def_readwrite("version", &arborio::meta_data::version, "cable-cell component version.");

    // arborio::cable_cell_component
    pybind11::class_<arborio::cable_cell_component> cable_component(m, "cable_component");
    cable_component
        .def_readwrite("meta_data", &arborio::cable_cell_component::meta, "cable-cell component meta-data.")
        .def_readwrite("component", &arborio::cable_cell_component::component, "cable-cell component.")
        .def("__repr__", [](const arborio::cable_cell_component& comp) {
            std::stringstream stream;
            arborio::write_component(stream, comp);
            return "<arbor.cable_component>\n"+stream.str();
        })
        .def("__str__", [](const arborio::cable_cell_component& comp) {
          std::stringstream stream;
          arborio::write_component(stream, comp);
          return "<arbor.cable_component>\n"+stream.str();
        });
}
}
