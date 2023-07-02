#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include <arbor/load_balance.hpp>
#include <arbor/network.hpp>
#include <arbor/network_generation.hpp>
#include <arbor/util/any_visitor.hpp>
#include <arborio/label_parse.hpp>
#include <arborio/networkio.hpp>

#include <unordered_map>
#include <string>
#include <variant>
#include <functional>

#include "context.hpp"
#include "error.hpp"
#include "recipe.hpp"
#include "strprintf.hpp"
#include "util.hpp"

namespace py = pybind11;

namespace pyarb {

void register_network(py::module& m) {
    using namespace py::literals;

    py::class_<arb::network_site_info> network_site_info(
        m, "network_site_info", "Identifies a network site to connect to / from");
    network_site_info.def_readwrite("gid", &arb::network_site_info::gid)
        .def_readwrite("kind", &arb::network_site_info::kind)
        .def_readwrite("label", &arb::network_site_info::label)
        .def_readwrite("location", &arb::network_site_info::location)
        .def_readwrite("global_location", &arb::network_site_info::global_location)
        .def("__repr__", [](const arb::network_site_info& s) { return util::pprintf("{}", s); })
        .def("__str__", [](const arb::network_site_info& s) { return util::pprintf("{}", s); });

    py::class_<arb::network_connection_info> network_connection_info(
        m, "network_connection_info", "Identifies a network connection");
    network_connection_info.def_readwrite("src", &arb::network_connection_info::src)
        .def_readwrite("dest", &arb::network_connection_info::dest)
        .def("__repr__", [](const arb::network_connection_info& c) { return util::pprintf("{}", c); })
        .def("__str__", [](const arb::network_connection_info& c) { return util::pprintf("{}", c); });

    py::class_<arb::network_selection> network_selection(
        m, "network_selection", "Network selection.");

    network_selection
        .def_static("custom",
            [](arb::network_selection::custom_func_type func) {
                return arb::network_selection::custom(
                    [=](const arb::network_connection_info& c) {
                        return try_catch_pyexception(
                            [&]() {
                                pybind11::gil_scoped_acquire guard;
                                return func(c);
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
                    [=](const arb::network_connection_info& c) {
                        return try_catch_pyexception(
                            [&]() {
                                pybind11::gil_scoped_acquire guard;
                                return func(c);
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

    m.def(
        "generate_network_connections",
        [](const std::shared_ptr<py_recipe>& rec,
            std::shared_ptr<context_shim> ctx,
            std::optional<arb::domain_decomposition> decomp) {
            py_recipe_shim rec_shim(rec);

            if (!ctx) ctx = std::make_shared<context_shim>(arb::make_context());
            if (!decomp) decomp = arb::partition_load_balance(rec_shim, ctx->context);

            return generate_network_connections(rec_shim, ctx->context, decomp.value());
        },
        "recipe"_a,
        pybind11::arg_v("context", pybind11::none(), "Execution context"),
        pybind11::arg_v("decomp", pybind11::none(), "Domain decomposition"),
        "Generate network connections from the network description in the recipe. Will only "
        "generate connections with local gids in the domain composition as destination.");
}

}  // namespace pyarb
