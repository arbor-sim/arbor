#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <fstream>
#include <iomanip>

#include <arbor/cable_cell.hpp>
#include <arbor/util/any_visitor.hpp>

#include <arborio/cableio.hpp>

#include "error.hpp"
#include "util.hpp"
#include "strprintf.hpp"
#include "proxy.hpp"

namespace pyarb {

namespace py = pybind11;

arborio::cable_cell_component load_component(py::object fn) {
    const auto fname = util::to_path(fn);
    std::ifstream fid{fname};
    if (!fid.good()) {
        throw arb::file_not_found_error(fname);
    }
    auto component = arborio::parse_component(fid);
    if (!component) {
        throw pyarb_error("Error while trying to load component from \"" + fname + "\": " + component.error().what());
    }
    return component.value();
};

template<typename T>
void write_component(const T& component, py::object fn) {
    std::ofstream fid(util::to_path(fn));
    arborio::write_component(fid, component, arborio::meta_data{});
}

void write_component(const arborio::cable_cell_component& component, py::object fn) {
    std::ofstream fid(util::to_path(fn));
    arborio::write_component(fid, component);
}

void register_cable_loader(pybind11::module& m) {
    m.def("load_component",
          &load_component,
          pybind11::arg_v("filename", "the name of the file."),
          "Load arbor-component (decor, morphology, label_dict, cable_cell) from file.");

    m.def("write_component",
          [](const arborio::cable_cell_component& d, py::object fn) {
            return write_component(d, fn);
          },
          pybind11::arg_v("object", "the cable_component object."),
          pybind11::arg_v("filename", "the path of the file."),
          "Write cable_component to file.");

    m.def("write_component",
          [](const arb::decor& d, py::object fn) {
            return write_component<arb::decor>(d, fn);
          },
          pybind11::arg_v("object", "the decor object."),
          pybind11::arg_v("filename", "the name of the file."),
          "Write decor to file.");

    m.def("write_component",
          [](const arb::label_dict& d, py::object fn) {
            return write_component<arb::label_dict>(d, fn);
          },
          pybind11::arg_v("object", "the label_dict object."),
          pybind11::arg_v("filename", "the name of the file."),
          "Write label_dict to file.");

    m.def("write_component",
          [](const arb::morphology& d, py::object fn) {
            return write_component<arb::morphology>(d, fn);
          },
          pybind11::arg_v("object", "the morphology object."),
          pybind11::arg_v("filename", "the name of the file."),
          "Write morphology to file.");

    m.def("write_component",
          [](const arb::cable_cell& d, py::object fn) {
            return write_component<arb::cable_cell>(d, fn);
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
        .def_property_readonly(
            "component",
            [](const arborio::cable_cell_component& c) {
                using py_cable_cell_variant = std::variant<arb::morphology, pyarb::label_dict_proxy, arb::decor, arb::cable_cell>;
                auto cable_cell_variant_visitor = arb::util::overload(
                    [&](const arb::morphology& p) { return py_cable_cell_variant(p);},
                    [&](const arb::label_dict& p) { return py_cable_cell_variant(label_dict_proxy(p));},
                    [&](const arb::decor& p)      { return py_cable_cell_variant(p);},
                    [&](const arb::cable_cell& p) { return py_cable_cell_variant(p);});
                return std::visit(cable_cell_variant_visitor, c.component);
            },
            "cable-cell component.")
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
