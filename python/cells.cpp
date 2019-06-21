#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <arbor/benchmark_cell.hpp>
#include <arbor/cable_cell.hpp>
#include <arbor/lif_cell.hpp>
#include <arbor/schedule.hpp>
#include <arbor/spike_source_cell.hpp>
#include <arbor/util/unique_any.hpp>

#include "error.hpp"
#include "schedule.hpp"
#include "strprintf.hpp"

namespace pyarb {

// Convert a cell description inside a Python object to a cell
// description in a unique_any, as required by the recipe interface.
//
// Warning: requires that the GIL has been acquired before calling,
// if there is a segmentation fault in the cast or isinstance calls,
// check that the caller has the GIL.
arb::util::unique_any convert_cell(pybind11::object o) {
    using pybind11::isinstance;
    using pybind11::cast;

    pybind11::gil_scoped_acquire guard;
    if (isinstance<arb::spike_source_cell>(o)) {
        return arb::util::unique_any(cast<arb::spike_source_cell>(o));
    }
    if (isinstance<arb::benchmark_cell>(o)) {
        return arb::util::unique_any(cast<arb::benchmark_cell>(o));
    }
    if (isinstance<arb::lif_cell>(o)) {
        return arb::util::unique_any(cast<arb::lif_cell>(o));
    }
    if (isinstance<arb::cable_cell>(o)) {
        return arb::util::unique_any(cast<arb::cable_cell>(o));
    }

    throw pyarb_error("recipe.cell_description returned \""
                      + std::string(pybind11::str(o))
                      + "\" which does not describe a known Arbor cell type");
}

//
// Somewhat hacky bit of code for generating cells with random morphologies.
//

// Parameters used to generate the random cell morphologies.
struct cell_parameters {
    cell_parameters() = default;

    //  Maximum number of levels in the cell (not including the soma)
    unsigned max_depth = 5;

    // The following parameters are described as ranges.
    // The first value is at the soma, and the last value is used on the last level.
    // Values at levels in between are found by linear interpolation.
    std::array<double,2> branch_probs = {1.0, 0.5}; //  Probability of a branch occuring.
    std::array<unsigned,2> compartments = {20, 2};  //  Compartment count on a branch.
    std::array<double,2> lengths = {200, 20};       //  Length of branch in μm.

    // The number of synapses per cell.
    unsigned synapses = 1;

    friend std::ostream& operator<<(std::ostream& o, const cell_parameters& p) {
        return
            o << "<cell_parameters: depth " << p.max_depth
              << ", synapses " << p.synapses
              << ", branch_probs [" << p.branch_probs[0] << ":" << p.branch_probs[1] << "]"
              << ", compartments [" << p.compartments[0] << ":" << p.compartments[1] << "]"
              << ", lengths ["      << p.lengths[0]      << ":" << p.lengths[1] << "]>";
    }
};

// Helper used to interpolate in branch_cell.
template <typename T>
double interp(const std::array<T,2>& r, unsigned i, unsigned n) {
    double p = i * 1./(n-1);
    double r0 = r[0];
    double r1 = r[1];
    return r[0] + p*(r1-r0);
}

arb::cable_cell branch_cell(arb::cell_gid_type gid, const cell_parameters& params) {
    arb::cable_cell cell;

    // Add soma.
    auto soma = cell.add_soma(12.6157/2.0); // For area of 500 μm².
    soma->rL = 100;
    soma->add_mechanism("hh");

    std::vector<std::vector<unsigned>> levels;
    levels.push_back({0});

    // Standard mersenne_twister_engine seeded with gid.
    std::mt19937 gen(gid);
    std::uniform_real_distribution<double> dis(0, 1);

    double dend_radius = 0.5; // Diameter of 1 μm for each cable.

    unsigned nsec = 1;
    for (unsigned i=0; i<params.max_depth; ++i) {
        // Branch prob at this level.
        double bp = interp(params.branch_probs, i, params.max_depth);
        // Length at this level.
        double l = interp(params.lengths, i, params.max_depth);
        // Number of compartments at this level.
        unsigned nc = std::round(interp(params.compartments, i, params.max_depth));

        std::vector<unsigned> sec_ids;
        for (unsigned sec: levels[i]) {
            for (unsigned j=0; j<2; ++j) {
                if (dis(gen)<bp) {
                    sec_ids.push_back(nsec++);
                    auto dend = cell.add_cable(sec, arb::section_kind::dendrite, dend_radius, dend_radius, l);
                    dend->set_compartments(nc);
                    dend->add_mechanism("pas");
                    dend->rL = 100;
                }
            }
        }
        if (sec_ids.empty()) {
            break;
        }
        levels.push_back(sec_ids);
    }

    // Add spike threshold detector at the soma.
    cell.add_detector({0,0}, 10);

    // Add a synapse to the mid point of the first dendrite.
    cell.add_synapse({1, 0.5}, "expsyn");

    // Add additional synapses.
    for (unsigned i=1u; i<params.synapses; ++i) {
        cell.add_synapse({1, 0.5}, "expsyn");
    }

    return cell;
}

//
// string printers
//

std::string lif_str(const arb::lif_cell& c){
    return util::pprintf(
                "<arbor.lif_cell: tau_m {}, V_th {}, C_m {}, E_L {}, V_m {}, t_ref {}, V_reset {}>",
                c.tau_m, c.V_th, c.C_m, c.E_L, c.V_m, c.t_ref, c.V_reset);
}

void register_cells(pybind11::module& m) {
    using namespace pybind11::literals;

    pybind11::class_<arb::spike_source_cell> spike_source_cell(m, "spike_source_cell",
        "A spike source cell, that generates a user-defined sequence of spikes that act as inputs for other cells in the network.");

    spike_source_cell
        .def(pybind11::init<>(
            [](const regular_schedule_shim& sched){
                return arb::spike_source_cell{sched.schedule()};}),
            "schedule"_a, "Construct a spike source cell that generates spikes at regular intervals.")
        .def(pybind11::init<>(
            [](const explicit_schedule_shim& sched){
                return arb::spike_source_cell{sched.schedule()};}),
            "schedule"_a, "Construct a spike source cell that generates spikes at a sequence of user-defined times.")
        .def(pybind11::init<>(
            [](const poisson_schedule_shim& sched){
                return arb::spike_source_cell{sched.schedule()};}),
            "schedule"_a, "Construct a spike source cell that generates spikes at times defined by a Poisson sequence.")
        .def("__repr__", [](const arb::spike_source_cell&){return "<arbor.spike_source_cell>";})
        .def("__str__",  [](const arb::spike_source_cell&){return "<arbor.spike_source_cell>";});

    pybind11::class_<arb::benchmark_cell> benchmark_cell(m, "benchmark_cell",
        "A benchmarking cell, used by Arbor developers to test communication performance.\n"
        "A benchmark cell generates spikes at a user-defined sequence of time points, and\n"
        "the time taken to integrate a cell can be tuned by setting the real_time ratio,\n"
        "for example if realtime_ratio=2, a cell will take 2 seconds of CPU time to\n"
        "simulate 1 second.\n");

    benchmark_cell
        .def(pybind11::init<>(
            [](const regular_schedule_shim& sched, double ratio){
                return arb::benchmark_cell{sched.schedule(), ratio};}),
            "schedule"_a, "realtime_ratio"_a=1.0,
            "Construct a benchmark cell that generates spikes at regular intervals.")
        .def(pybind11::init<>(
            [](const explicit_schedule_shim& sched, double ratio){
                return arb::benchmark_cell{sched.schedule(), ratio};}),
            "schedule"_a, "realtime_ratio"_a=1.0,
            "Construct a benchmark cell that generates spikes at a sequence of user-defined times.")
        .def(pybind11::init<>(
            [](const poisson_schedule_shim& sched, double ratio){
                return arb::benchmark_cell{sched.schedule(), ratio};}),
            "schedule"_a, "realtime_ratio"_a=1.0,
            "Construct a benchmark cell that generates spikes at times defined by a Poisson sequence.")
        .def("__repr__", [](const arb::benchmark_cell&){return "<arbor.benchmark_cell>";})
        .def("__str__",  [](const arb::benchmark_cell&){return "<arbor.benchmark_cell>";});

    pybind11::class_<arb::lif_cell> lif_cell(m, "lif_cell",
        "A benchmarking cell, used by Arbor developers to test communication performance.");

    lif_cell
        .def(pybind11::init<>())
        .def_readwrite("tau_m", &arb::lif_cell::tau_m,  "Membrane potential decaying constant [ms].")
        .def_readwrite("V_th",  &arb::lif_cell::V_th,   "Firing threshold [mV].")
        .def_readwrite("C_m",   &arb::lif_cell::C_m,    "Membrane capacitance [pF].")
        .def_readwrite("E_L",   &arb::lif_cell::E_L,    "Resting potential [mV].")
        .def_readwrite("V_m",   &arb::lif_cell::V_m,    "Initial value of the Membrane potential [mV].")
        .def_readwrite("t_ref", &arb::lif_cell::t_ref,  "Refractory period [ms].")
        .def_readwrite("V_reset", &arb::lif_cell::V_reset, "Reset potential [mV].")
        .def("__repr__", &lif_str)
        .def("__str__",  &lif_str);

    pybind11::class_<cell_parameters> cell_params(m, "cell_parameters", "Parameters used to generate the random cell morphologies.");
    cell_params
        .def(pybind11::init<>())
        .def_readwrite("depth", &cell_parameters::max_depth,"The maximum depth of the branch structure.")
        .def_readwrite("lengths",   &cell_parameters::lengths,  "Length of branch in μm [range].")
        .def_readwrite("synapses",  &cell_parameters::synapses, "The number of randomly generated synapses on the cell.")
        .def_readwrite("branch_probs", &cell_parameters::branch_probs, "Probability of a branch occuring [range].")
        .def_readwrite("compartments", &cell_parameters::compartments, "Compartment count on a branch [range].")
        .def("__repr__", util::to_string<cell_parameters>)
        .def("__str__",  util::to_string<cell_parameters>);

    // Wrap cable cell description type.
    // Cable cells are going to be replaced with a saner API, so we don't go
    // adding much in the way of interface. Instead we just provide a helper
    // that will generate random cell morphologies for benchmarking.
    pybind11::class_<arb::cable_cell> cable_cell(m, "cable_cell");

    cable_cell
        .def("__repr__", [](const arb::cable_cell&){return "<arbor.cable_cell>";})
        .def("__str__",  [](const arb::cable_cell&){return "<arbor.cable_cell>";});

    m.def("branch_cell", &branch_cell,
        "Construct a branching cell with a random morphology and synapse end points locations described by params.\n"
        "seed is an integral value used to seed the random number generator, for which the gid of the cell is a good default.",
        "seed"_a,
        "params"_a=cell_parameters());
}

}
