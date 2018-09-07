#include <string>

#include <arbor/common_types.hpp>

#include "strings.hpp"

#include <pybind11/pybind11.h>

namespace arb {
namespace py {

void register_identifiers(pybind11::module& m) {
    using namespace pybind11::literals;

    pybind11::class_<cell_member_type> cell_member(m, "cell_member",
        "For global identification of a cell-local item.\n\n"
        "Items of cell_member must:\n"
        "(1) be associated with a unique cell, identified by the member gid;\n"
        "(2) identify an item within a cell-local collection by the member index.\n");

    cell_member
        .def(pybind11::init<>())
        .def(pybind11::init(
            [](arb::cell_gid_type gid, arb::cell_lid_type idx) {
                arb::cell_member_type m;
                m.gid = gid;
                m.index = idx;
                return m;
            }),
            "gid"_a,
            "index"_a,
            "Construct with gid and index.")
        .def_readwrite("gid",   &cell_member_type::gid,
            "The global identifier of the cell.")
        .def_readwrite("index", &cell_member_type::index,
            "Cell-local index of the item.")
        .def("__str__",  &cell_member_string)
        .def("__repr__", &cell_member_string);
}

} // namespace py
} // namespace arb
