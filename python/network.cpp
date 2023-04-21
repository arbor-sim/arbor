#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include <arbor/network.hpp>
#include <arbor/util/any_visitor.hpp>
#include <arborio/label_parse.hpp>
#include <arborio/networkio.hpp>

#include <unordered_map>
#include <string>
#include <variant>
#include <functional>

#include "error.hpp"
#include "util.hpp"
#include "strprintf.hpp"

namespace py = pybind11;

namespace pyarb {

void register_network(py::module& m) {
    using namespace py::literals;

    py::class_<arb::network_site_info> network_site_info(
        m, "network_site_info", "Identifies a network site to connect to / from");
    network_site_info.def_readwrite("gid", &arb::network_site_info::gid)
        .def_readwrite("lid", &arb::network_site_info::lid)
        .def_readwrite("kind", &arb::network_site_info::kind)
        .def_readwrite("label", &arb::network_site_info::label)
        .def_readwrite("location", &arb::network_site_info::location)
        .def_readwrite("global_location", &arb::network_site_info::global_location)
        .def("__repr__", [](const arb::network_site_info& s) {
            return util::pprintf("<arbor.network_site_info: lid {}, kind {}, label \"{}\", "
                                 "location {}, global_location {}>",
                s.lid,
                s.kind,
                s.label,
                s.location,
                s.global_location);
        });

    py::class_<arb::network_selection> network_selection(
        m, "network_selection", "Network selection.");
    network_selection
        .def_static("custom",
            [](arb::network_selection::custom_func_type func) {
                return arb::network_selection::custom(
                    [=](const arb::network_site_info& src, const arb::network_site_info& dest) {
                        return try_catch_pyexception(
                            [&]() {
                                pybind11::gil_scoped_acquire guard;
                                return func(src, dest);
                            },
                            "Python error already thrown");
                    });
            })
        .def("__str__",
            [](const arb::network_selection& s) {
                return util::pprintf("<arbor.network_selection: {}>", s);
            })
        .def("__repr__", [](const arb::network_selection& s) { return util::pprintf("{}", s); });

    py::class_<arb::network_value> network_value(m, "network_value", "Network value.");
    network_value
        .def_static("custom",
            [](arb::network_value::custom_func_type func) {
                return arb::network_value::custom(
                    [=](const arb::network_site_info& src, const arb::network_site_info& dest) {
                        return try_catch_pyexception(
                            [&]() {
                                pybind11::gil_scoped_acquire guard;
                                return func(src, dest);
                            },
                            "Python error already thrown");
                    });
            })
        .def("__str__",
            [](const arb::network_value& v) {
                return util::pprintf("<arbor.network_value: {}>", v);
            })
        .def("__repr__", [](const arb::network_value& v) { return util::pprintf("{}", v); });

    py::class_<arb::network_description> network_description(
        m, "network_description", "Network description.");
    network_description.def(
        py::init(
            [](std::string selection,
                std::string weight,
                std::string delay,
                std::unordered_map<std::string,
                    std::variant<std::string, arb::network_selection, arb::network_value>> map) {
                arb::network_label_dict dict;
                for (const auto& [label, v]: map) {
                    const auto& dict_label = label;
                    std::visit(
                        arb::util::overload(
                            [&](const std::string& s) {
                                auto sel = arborio::parse_network_selection_expression(s);
                                if (sel) {
                                    dict.set(dict_label, *sel);
                                    return;
                                }

                                auto val = arborio::parse_network_value_expression(s);
                                if (val) {
                                    dict.set(dict_label, *val);
                                    return;
                                }

                                throw pyarb_error(
                                    std::string("Failed to parse \"") + dict_label +
                                    "\" label in dict of network description. \nSelection "
                                    "label parse error:\n" +
                                    sel.error().what() + "\nValue label parse error:\n" +
                                    val.error().what());
                            },
                            [&](const arb::network_selection& sel) { dict.set(dict_label, sel); },
                            [&](const arb::network_value& val) { dict.set(dict_label, val); }),
                        v);
                }
                auto desc =  arb::network_description{
                    arborio::parse_network_selection_expression(selection).unwrap(),
                    arborio::parse_network_value_expression(weight).unwrap(),
                    arborio::parse_network_value_expression(delay).unwrap(),
                    dict};
                return desc;
            }),
        "selection"_a,
        "weight"_a,
        "delay"_a,
        "dict"_a,
        "Construct network description.");
}

}  // namespace pyarb
