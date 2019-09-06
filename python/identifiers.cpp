#include <ostream>
#include <string>

#include <pybind11/pybind11.h>

#include <arbor/common_types.hpp>

#include "strprintf.hpp"
#include "recipe.hpp"

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

//    pybind11::class_<pyarb::probe_info_shim> probe_info(m, "probe_info",
    pybind11::class_<arb::probe_info> probe_info(m, "probe_info",
        "Probes are specified in the recipe objects that are used to initialize a model;\n"
        "the specification of the item or value that is subjected to a probe\n"
        "will be specific to a particular cell type.");
    probe_info
        .def(pybind11::init(
            [](arb::cell_member_type id, arb::probe_tag tag, pybind11::object address) {
//                return pyarb::probe_info_shim{id, tag, address};
                return arb::probe_info{id, tag, address};
            }),
            "id"_a, "tag"_a, "address"_a,
            "Construct a probe_info with arguments:\n"
            "  id:      The cell gid, index of the probe.\n"
            "  tag:     An opaque key.\n"
            "  address: The cell-type specific location info.\n")
//        .def_readwrite("id",      &pyarb::probe_info_shim::id,      "Cell gid, index of probe.")
//        .def_readwrite("tag",     &pyarb::probe_info_shim::tag,     "Opaque key, returned in sample record.")
//        .def_readwrite("address", &pyarb::probe_info_shim::address, "Cell-type specific location info, specific to cell kind of id.gid")
//        .def("__str__", [](pyarb::probe_info_shim i) {return pprintf("<arbor.probe_info: id {}, tag {}, address {}>", i.id, i.tag, i.address);})
//        .def("__repr__",[](pyarb::probe_info_shim i) {return pprintf("<arbor.probe_info: id {}, tag {}, address {}>", i.id, i.tag, i.address);});
        .def_readwrite("id",      &arb::probe_info::id,      "Cell gid, index of probe.")
        .def_readwrite("tag",     &arb::probe_info::tag,     "Opaque key, returned in sample record.")
        .def_readwrite("address", &arb::probe_info::address, "Cell-type specific location info, specific to cell kind of id.gid")
        .def("__str__", [](arb::probe_info i) {return pprintf("<arbor.probe_info: id {}, tag {}>", i.id, i.tag);})
        .def("__repr__",[](arb::probe_info i) {return pprintf("<arbor.probe_info: id {}, tag {}>", i.id, i.tag);});
}

} // namespace pyarb
