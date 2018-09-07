#include <arbor/benchmark_cell.hpp>
#include <arbor/mc_cell.hpp>
#include <arbor/mc_segment.hpp>

#include "strings.hpp"

#include <pybind11/pybind11.h>

namespace pyarb {

/*
 * Create cell with just a soma:
 *
 * Soma:
 *    diameter: 18.8 µm
 *    mechanisms: HH (default params)
 *    bulk resistivitiy: 100 Ω·cm [default]
 *    capacitance: 0.01 F/m² [default]
 */

auto make_cell_soma_only() {
    arb::mc_cell c;

    auto soma = c.add_soma(18.8/2.0);
    soma->add_mechanism("hh");

    return c;
}

template <typename Sched>
arb::benchmark_cell py_make_benchmark_cell(const Sched& sched) {
    return arb::benchmark_cell(sched.schedule(), 1.0);
}

void register_cells(pybind11::module& m) {
    using namespace pybind11::literals;

    // Wrap cell description type.
    pybind11::class_<arb::mc_cell> mccell(m, "mccell");

    pybind11::class_<arb::segment_location> segment_location(m, "segment_location");
    segment_location
        .def(pybind11::init<arb::cell_lid_type, double>())
        .def_readwrite("segment", &arb::segment_location::segment)
        .def_readwrite("position", &arb::segment_location::position)
        .def("__str__",  &segment_location_string)
        .def("__repr__", &segment_location_string);

    mccell.def("add_synapse",
        [](arb::mc_cell& c, arb::segment_location l) {
            c.add_synapse(l, "expsyn");},
        "location"_a)
    .def("add_stimulus",
        [](arb::mc_cell& c, arb::segment_location loc, double t0, double duration, double weight) {
            c.add_stimulus(loc, {t0, duration, weight});},
        "Add stimulus to the cell",
        "location"_a, "t0 (ms)"_a, "duration (ms)"_a, "weight (nA)"_a)
    .def("add_detector",  &arb::mc_cell::add_detector,
        "location"_a, "threashold(mV)"_a)
    .def("__str__",  &cell_string)
    .def("__repr__", &cell_string);

    m.def("make_soma_cell", &make_cell_soma_only,
        "Make a single compartment cell with properties:"
        "\n    diameter 18.8 µm;"
        "\n    mechanisms HH;"
        "\n    bulk resistivitiy 100 Ω·cm;"
        "\n    capacitance 0.01 F⋅m⁻²." );

    // Cell kinds.

    pybind11::class_<arb::benchmark_cell> benchmark_cell(m, "benchmark_cell");

    benchmark_cell
        .def(pybind11::init<>())
        .def_readwrite("realtime_ratio", &arb::benchmark_cell::realtime_ratio,
            "Time taken in ms to advance the cell one ms of simulation time. \n"
            "If equal to 1, then a single cell can be advanced in realtime.");
}

} // namespace pyarb
