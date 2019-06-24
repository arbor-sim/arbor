#include <ostream>
#include <string>

#include <pybind11/pybind11.h>

#include <arbor/common_types.hpp>

#include "strprintf.hpp"

namespace pyarb {

using util::pprintf;

void register_identifiers(pybind11::module& m) {
    using namespace pybind11::literals;

    pybind11::class_<arb::cell_member_type> cell_member(m, "cell_member",
        "For global identification of a cell-local item.\n\n"
        "Items of cell_member must:\n"
        "  (1) be associated with a unique cell, identified by the member gid;\n"
        "  (2) identify an item within a cell-local collection by the member index.\n");

    cell_member
        .def(pybind11::init(
            [](arb::cell_gid_type gid, arb::cell_lid_type idx) {
                return arb::cell_member_type{gid, idx};
            }),
            "gid"_a, "index"_a,
            "Construct a cell member with arguments:\n"
            "  gid:     The global identifier of the cell.\n"
            "  index:   The cell-local index of the item.\n")
        .def_readwrite("gid",   &arb::cell_member_type::gid,
            "The global identifier of the cell.")
        .def_readwrite("index", &arb::cell_member_type::index,
            "Cell-local index of the item.")
        .def("__str__", [](arb::cell_member_type m) {return pprintf("<arbor.cell_member: gid {}, index {}>", m.gid, m.index);})
        .def("__repr__",[](arb::cell_member_type m) {return pprintf("<arbor.cell_member: gid {}, index {}>", m.gid, m.index);});

    pybind11::enum_<arb::cell_kind>(m, "cell_kind",
        "Enumeration used to identify the cell kind, used by the model to group equal kinds in the same cell group.")
        .value("benchmark", arb::cell_kind::benchmark,
            "Proxy cell used for benchmarking.")
        .value("cable", arb::cell_kind::cable,
            "A cell with morphology described by branching 1D cable segments.")
        .value("lif", arb::cell_kind::lif,
            "Leaky-integrate and fire neuron.")
        .value("spike_source", arb::cell_kind::spike_source,
            "Proxy cell that generates spikes from a spike sequence provided by the user.");

    pybind11::enum_<arb::backend_kind>(m, "backend",
        "Enumeration used to indicate which hardware backend to execute a cell group on.")
        .value("gpu", arb::backend_kind::gpu,
            "Use GPU backend.")
        .value("multicore", arb::backend_kind::multicore,
            "Use multicore backend.");

    pybind11::enum_<arb::binning_kind>(m, "binning",
        "Enumeration for event time binning policy.")
        .value("none", arb::binning_kind::none,
            "No binning policy.")
        .value("regular", arb::binning_kind::regular,
            "Round time down to multiple of binning interval.")
        .value("following", arb::binning_kind::following,
            "Round times down to previous event if within binning interval.");
}

} // namespace pyarb
