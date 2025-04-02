#include "label_dict.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace pyarb {

namespace py = pybind11;
using namespace py::literals;
void register_label_dict(py::module& m) {

    struct label_dict_iter_state {
        label_dict_iter_state(const ::pyarb::label_dict& ld_) {
            items_.insert(items_.end(), ld_.regions.begin(), ld_.regions.end());
            items_.insert(items_.end(), ld_.locsets.begin(), ld_.locsets.end());
            items_.insert(items_.end(), ld_.iexpressions.begin(), ld_.iexpressions.end());
            std::sort(items_.begin(), items_.end());
        }
        std::vector<std::pair<std::string, std::string>> items_;
        size_t idx = 0;
        std::pair<std::string, std::string> next() {
            if (idx == items_.size()) throw py::stop_iteration();
            return items_[idx++];
        }
    };

    struct py_label_dict_key_iterator {
        py_label_dict_key_iterator(const ::pyarb::label_dict& ld): state{ld} { }
        label_dict_iter_state state;
        std::string next() { return state.next().first; }
    };
    struct py_label_dict_item_iterator {
        py_label_dict_item_iterator(const ::pyarb::label_dict& ld): state{ld} { }
        label_dict_iter_state state;
        std::tuple<std::string, std::string> next() { return state.next(); }
    };
    struct py_label_dict_value_iterator {
        py_label_dict_value_iterator(const ::pyarb::label_dict& ld): state{ld} { }
        label_dict_iter_state state;
        std::string next() { return state.next().second; }
    };

    py::class_<py_label_dict_key_iterator>(m, "LabelDictKeyIterator")
        .def("__iter__", [](py_label_dict_key_iterator &it) -> py_label_dict_key_iterator& { return it; })
        .def("__next__", &py_label_dict_key_iterator::next);

    py::class_<py_label_dict_value_iterator>(m, "LabelDictValueIterator")
        .def("__iter__", [](py_label_dict_value_iterator &it) -> py_label_dict_value_iterator& { return it; })
        .def("__next__", &py_label_dict_value_iterator::next);

    py::class_<py_label_dict_item_iterator>(m, "LabelDictItemIterator")
        .def("__iter__", [](py_label_dict_item_iterator &it) -> py_label_dict_item_iterator& { return it; })
        .def("__next__", &py_label_dict_item_iterator::next);


    py::class_<label_dict> pyld(m, "label_dict",
        "A dictionary of labelled region and locset definitions, with a\n"
        "unique label assigned to each definition.");
    pyld
        .def(py::init<>(),
             "Create an empty label dictionary.")
        .def(py::init<const std::unordered_map<std::string, std::string>&>(),
            "Initialize a label dictionary from a dictionary with string labels as keys,"
            " and corresponding definitions as strings.")
        .def(py::init<const label_dict&>(),
            "Initialize a label dictionary from another one")
        .def(py::init([](py::iterator& it) {
                label_dict ld;
                for (; it != py::iterator::sentinel(); ++it) {
                    const auto tuple = it->cast<py::sequence>();
                    const auto key   = tuple[0].cast<std::string>();
                    const auto value = tuple[1].cast<std::string>();
                    ld.setitem(key, value);
                }
                return ld;
            }),
            "Initialize a label dictionary from an iterable of key, definition pairs")
        .def("add_swc_tags", &label_dict::add_swc_tags,
             "Add standard SWC tagged regions.\n"
             " - soma: (tag 1)\n"
             " - axon: (tag 2)\n"
             " - dend: (tag 3)\n"
             " - apic: (tag 4)")
        .def("__setitem__", &label_dict::setitem)
        .def("__getitem__", &label_dict::getitem)
        .def("__len__", &label_dict::size)
        .def("__contains__", &label_dict::contains)
        .def("__iter__",
            [](const label_dict& ld) { return py_label_dict_item_iterator(ld); },
            py::keep_alive<0, 1>())
        .def("keys",
            [](const label_dict &ld) { return py_label_dict_key_iterator(ld); },
            py::keep_alive<0, 1>())
        .def("items",
             [](const label_dict &ld) { return py_label_dict_item_iterator(ld); },
             py::keep_alive<0, 1>())
        .def("values",
             [](const label_dict &ld) { return py_label_dict_value_iterator(ld); },
             py::keep_alive<0, 1>())
        .def("append",
             [](label_dict& l, const label_dict& other, const std::string& prefix) { return l.extend(other, prefix); },
            "other"_a, "The label_dict to be imported", "prefix"_a="", "optional prefix appended to the region and locset labels",
            "Import the entries of a another label dictionary with an optional prefix.")
        .def("update",
             [](label_dict& l, const label_dict& other) { return l.extend(other); },
            "other"_a, "The label_dict to be imported"
            "Import the entries of a another label dictionary.")
        .def_readonly("regions", &label_dict::regions, "The region definitions.")
        .def_readonly("locsets", &label_dict::locsets, "The locset definitions.")
        .def_readonly("iexprs", &label_dict::iexpressions, "The iexpr definitions.")
        .def("__repr__", &label_dict::to_string)
        .def("__str__", &label_dict::to_string);
}
}
