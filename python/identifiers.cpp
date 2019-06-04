#include <string>

#include <arbor/common_types.hpp>

#include <pybind11/pybind11.h>

#include "strings.hpp"

namespace pyarb {

void register_identifiers(pybind11::module& m) {
    using namespace pybind11::literals;

    pybind11::class_<arb::cell_member_type> cell_member(m, "cell_member",
        "For global identification of a cell-local item.\n\n"
        "Items of cell_member must:\n"
        "(1) be associated with a unique cell, identified by the member gid;\n"
        "(2) identify an item within a cell-local collection by the member index.\n");

    cell_member
        .def(pybind11::init<>(),
            "Construct a cell member with default values gid = 0 and index = 0.")
        .def(pybind11::init(
            [](arb::cell_gid_type gid, arb::cell_lid_type idx) {
                arb::cell_member_type m;
                m.gid = gid;
                m.index = idx;
                return m;
            }),
            "gid"_a,
            "index"_a,
            "Construct a cell member with arguments:/n"
               "  gid:     The global identifier of the cell.\n"
               "  index:   The cell-local index of the item.\n")
        .def_readwrite("gid",   &arb::cell_member_type::gid,
            "The global identifier of the cell.")
        .def_readwrite("index", &arb::cell_member_type::index,
            "Cell-local index of the item.")
        .def("__str__",  &cell_member_string)
        .def("__repr__", &cell_member_string);

    pybind11::enum_<arb::cell_kind>(m, "cell_kind",
        "Enumeration used to indentify the cell kind, used by the model to group equal kinds in the same cell group.")
        .value("benchmark", arb::cell_kind::benchmark,
            "Proxy cell used for benchmarking.")
        .value("cable", arb::cell_kind::cable,
            "A cell with morphology described by branching 1D cable segments.")
        .value("lif", arb::cell_kind::lif,
            "Leaky-integrate and fire neuron.")
        .value("spike_source", arb::cell_kind::spike_source,
            "Proxy cell that generates spikes from a spike sequence provided by the user.");
}

} // namespace pyarb
