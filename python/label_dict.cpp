#include "label_dict.hpp"

#include <pybind11/pybind11.h>

namespace pyarb {

namespace py = pybind11;

void register_label_dict(py::module& m) {
    py::class_<label_dict_proxy> label_dict(m, "label_dict",
        "A dictionary of labelled region and locset definitions, with a\n"
        "unique label assigned to each definition.");
    label_dict
        .def(py::init<>(),
             "Create an empty label dictionary.")
        .def(py::init<const std::unordered_map<std::string, std::string>&>(),
            "Initialize a label dictionary from a dictionary with string labels as keys,"
            " and corresponding definitions as strings.")
        .def(py::init<const label_dict_proxy&>(),
            "Initialize a label dictionary from another one")
        .def(py::init([](py::iterator& it) {
                label_dict_proxy ld;
                for (; it != py::iterator::sentinel(); ++it) {
                    const auto tuple = it->cast<py::sequence>();
                    const auto key   = tuple[0].cast<std::string>();
                    const auto value = tuple[1].cast<std::string>();
                    ld.set(key, value);
                }
                return ld;
            }),
            "Initialize a label dictionary from an iterable of key, definition pairs")
        .def("add_swc_tags",
             [](label_dict_proxy& l) { return l.add_swc_tags(); },
             "Add standard SWC tagged regions.\n"
             " - soma: (tag 1)\n"
             " - axon: (tag 2)\n"
             " - dend: (tag 3)\n"
             " - apic: (tag 4)")
        .def("__setitem__",
            [](label_dict_proxy& l, const char* name, const char* desc) {
                l.set(name, desc);})
        .def("__getitem__",
            [](label_dict_proxy& l, const char* name) {
                if (auto v = l.getitem(name)) return v.value();
                throw py::key_error(name);
            })
        .def("__len__", &label_dict_proxy::size)
        .def("__iter__",
            [](const label_dict_proxy &ld) {
                return py::make_key_iterator(ld.cache.begin(), ld.cache.end());},
            py::keep_alive<0, 1>())
        .def("__contains__",
             [](const label_dict_proxy &ld, const char* name) {
                 return ld.contains(name);})
        .def("keys",
            [](const label_dict_proxy &ld) {
                return py::make_key_iterator(ld.cache.begin(), ld.cache.end());},
            py::keep_alive<0, 1>())
        .def("items",
             [](const label_dict_proxy &ld) {
                 return py::make_iterator(ld.cache.begin(), ld.cache.end());},
             py::keep_alive<0, 1>())
        .def("values",
             [](const label_dict_proxy &ld) {
                 return py::make_value_iterator(ld.cache.begin(), ld.cache.end());
             },
             py::keep_alive<0, 1>())
        .def("append", [](label_dict_proxy& l, const label_dict_proxy& other, const char* prefix) {
                l.import(other, prefix);
            },
            "other"_a, "The label_dict to be imported"
            "prefix"_a="", "optional prefix appended to the region and locset labels",
            "Import the entries of a another label dictionary with an optional prefix.")
        .def("update", [](label_dict_proxy& l, const label_dict_proxy& other) {
                l.import(other);
            },
            "other"_a, "The label_dict to be imported"
            "Import the entries of a another label dictionary.")
        .def_readonly("regions", &label_dict_proxy::regions,
             "The region definitions.")
        .def_readonly("locsets", &label_dict_proxy::locsets,
             "The locset definitions.")
        .def("__repr__", [](const label_dict_proxy& d){return d.to_string();})
        .def("__str__",  [](const label_dict_proxy& d){return d.to_string();});
}
}
