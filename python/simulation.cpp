#include <pybind11/pybind11.h>

#include <arbor/common_types.hpp>
#include <arbor/sampling.hpp>
#include <arbor/simple_sampler.hpp>
#include <arbor/simulation.hpp>

#include "context.hpp"
#include "recipe.hpp"

namespace pyarb {


// A functor that models arb::simple_sampler.
// Holds a shared pointer to the trace_data used to store the samples, so that if
// the trace_data in sample_recorder is garbage collected in Python, stores will
// not seg fault.
template <typename V>
struct sample_callback {
    std::shared_ptr<arb::trace_data<V>> sample_store;

    sample_callback(const std::shared_ptr<arb::trace_data<V>>& state):
        sample_store(state)
    {}

    void operator() (arb::cell_member_type probe_id, arb::probe_tag tag, std::size_t n, const arb::sample_record* samples) {
        for (std::size_t i = 0; i<n; ++i) {
            if (auto p = arb::util::any_cast<const V*>(samples[i].data)) {
                sample_store->insert(sample_store->end(), {samples[i].time, *p});
            }
            else {
                throw std::runtime_error("unexpected sample type");
            }
        }
    };
};

// Helper type for recording samples from a simulation.
// This type is wrapped in Python, to expose sample_recorder::sample_store.
template <typename V>
struct sample_recorder {
    std::shared_ptr<arb::trace_data<V>> sample_store;

    sample_callback<V> callback() {
        // initialize the sample_store
        sample_store = std::make_shared<arb::trace_data<V>>();

        // The callback holds a copy of sample_store, i.e. the shared
        // pointer is held by both the sample_recorder and the callback, so if
        // the sample_recorder is destructed in the calling Python code, attempts
        // to write to sample_store inside the callback will not seg fault.
        return sample_callback<V>(sample_store);
    }

    const arb::trace_data<V> samples() const {
        return *sample_store;
    }
};

template <typename V>
std::shared_ptr<sample_recorder<V>> add_samplers(arb::simulation& sim, arb::time_type interval) {
    auto r = std::make_shared<sample_recorder<V>>();
    sim.add_sampler(arb::all_probes, arb::regular_schedule(interval), r->callback());
    return r;
}

template <typename V>
std::string sample_str(const arb::trace_entry<V>& t) {
    return util::pprintf("<arbor.sample: time {} ms, value {}>", t.t, t.v);
}

void register_simulation(pybind11::module& m) {
    using namespace pybind11::literals;

    // Simulation
    pybind11::class_<arb::simulation> simulation(m, "simulation",
        "The executable form of a model.\n"
        "A simulation is constructed from a recipe, and then used to update and monitor model state.");
    simulation
        // A custom constructor that wraps a python recipe with arb::py_recipe_shim
        // before forwarding it to the arb::recipe constructor.
        .def(pybind11::init(
            [](std::shared_ptr<py_recipe>& rec, const arb::domain_decomposition& decomp, const context_shim& ctx) {
                return new arb::simulation(py_recipe_shim(rec), decomp, ctx.context);
            }),
            // Release the python gil, so that callbacks into the python recipe don't deadlock.
            pybind11::call_guard<pybind11::gil_scoped_release>(),
            "Initialize the model described by a recipe, with cells and network distributed\n"
            "according to the domain decomposition and computational resources described by a context.",
            "recipe"_a, "domain_decomposition"_a, "context"_a)
        .def("reset", &arb::simulation::reset,
            pybind11::call_guard<pybind11::gil_scoped_release>(),
            "Reset the state of the simulation to its initial state.")
        .def("run", &arb::simulation::run,
            pybind11::call_guard<pybind11::gil_scoped_release>(),
            "Run the simulation from current simulation time to tfinal [ms], with maximum time step size dt [ms].",
            "tfinal"_a, "dt"_a=0.025)
        .def("set_binning_policy", &arb::simulation::set_binning_policy,
            "Set the binning policy for event delivery, and the binning time interval if applicable [ms].",
            "policy"_a, "bin_interval"_a)
        .def("remove_samplers", &arb::simulation::remove_all_samplers, "Remove all samplers from probes.")
        .def("__str__",  [](const arb::simulation&){ return "<arbor.simulation>"; })
        .def("__repr__", [](const arb::simulation&){ return "<arbor.simulation>"; });

    // Trace
    pybind11::class_<arb::trace_entry<double>> trace(m, "trace");
    trace
        .def(pybind11::init<>())
        .def_readwrite("time", &arb::trace_entry<double>::t, "The sample time [ms] at a specific probe.")
        .def_readwrite("value", &arb::trace_entry<double>::v, "The sample value at a specific probe.")
        .def("__str__",  &sample_str<double>)
        .def("__repr__", &sample_str<double>);

    // Samples
    pybind11::class_<sample_recorder<double>, std::shared_ptr<sample_recorder<double>>> sarec(m, "sample_recorder");
    sarec
        .def(pybind11::init<>())
        .def_property_readonly("samples", &sample_recorder<double>::samples, "A list of the recorded samples.");

    m.def("add_samplers", &add_samplers<double>,
        "Attach a sample recorder to an arbor simulation.\n"
        "The recorder will record all samples from a sampling interval [ms] matching all probe ids.",
        "simulation"_a, "interval"_a);
}

} // namespace pyarb
