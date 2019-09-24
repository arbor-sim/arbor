#include <pybind11/pybind11.h>

#include <arbor/common_types.hpp>
#include <arbor/sampling.hpp>
#include <arbor/simulation.hpp>

#include "context.hpp"
#include "error.hpp"
#include "recipe.hpp"

namespace pyarb {

//// ToDo: move to probes.cpp
//// ToDo: trace entry of different types/container (e.g. vector of doubles to get all samples of a cell)
//
//struct trace_entry {
//    arb::time_type t;
//    double v;
//};
//
//// A helper struct (state) ensuring that only one thread can write to the buffer holding the trace entries (mapped by probe id)
//struct sampler_state {
//    std::mutex mutex;
//    std::unordered_map<arb::cell_member_type, std::vector<trace_entry>> buffer;
//
//    std::vector<trace_entry>& locked_sampler_vec(arb::cell_member_type pid) {
//        // lock the mutex, s.t. other threads cannot write
//        std::lock_guard<std::mutex> lock(mutex);
//        // return or create entry
//        return buffer[pid];
//    }
//
//    // helper function to search probe id in buffer
//    bool has_pid(arb::cell_member_type pid) {
//        std::unordered_map<arb::cell_member_type, std::vector<trace_entry>>::iterator it = buffer.find(pid);
//        return (it != buffer.end());
//    }
//
//    // helper function to push back to locked vector
//    void push_back(arb::cell_member_type pid, trace_entry value) {
//        auto& v = locked_sampler_vec(pid);
//        v.push_back(std::move(value));
//    }
//
//    // helper function to return whole buffer
//    const std::unordered_map<arb::cell_member_type, std::vector<trace_entry>>& samples() const {
//        return buffer;
//    }
//
//    // helper function to return trace entry of probe id
//    const std::vector<trace_entry>& samples_of(arb::cell_member_type pid) {
//        return buffer[pid];
//    }
//};
//
//// A functor that models arb::sampler_function.
//// Holds a shared pointer to the trace_entry used to store the samples, so that if
//// the trace_entry in sample_recorder is garbage collected in Python, stores will
//// not seg fault.
//
//struct sample_callback {
//    std::shared_ptr<sampler_state> sample_store;
//
//    sample_callback(const std::shared_ptr<sampler_state>& state):
//        sample_store(state)
//    {}
//
//    void operator() (arb::cell_member_type probe_id, arb::probe_tag tag, std::size_t n, const arb::sample_record* recs) {
//        // lock before write
//        auto& v = sample_store->locked_sampler_vec(probe_id);
//        for (std::size_t i = 0; i<n; ++i) {
//            if (auto p = arb::util::any_cast<const double*>(recs[i].data)) {
//                v.push_back({recs[i].time, *p});
//            }
//            else {
//                throw std::runtime_error("unexpected sample type");
//            }
//        }
//    };
//};
//
//// Helper type for recording samples from a simulation.
//// This type is wrapped in Python, to expose sample_recorder::sample_store.
//struct sample_recorder {
//    std::shared_ptr<sampler_state> sample_store;
//
//    sample_callback callback() {
//        // initialize the sample_store
//        sample_store = std::make_shared<sampler_state>();
//
//        // The callback holds a copy of sample_store, i.e. the shared
//        // pointer is held by both the sample_recorder and the callback, so if
//        // the sample_recorder is destructed in the calling Python code, attempts
//        // to write to sample_store inside the callback will not seg fault.
//        return sample_callback(sample_store);
//    }
//
//    const std::vector<trace_entry>& samples(arb::cell_member_type pid) const {
//        if (sample_store->has_pid(pid)) {
//            return sample_store->samples_of(pid);
//        }
//        throw std::runtime_error(util::pprintf("probe id {} does not exist", pid));
//    }
//};
//
//// Adds sampler to one probe with pid
//std::shared_ptr<sample_recorder> attach_sample_recorder_on_probe(arb::simulation& sim, arb::time_type interval, arb::cell_member_type pid) {
//    auto r = std::make_shared<sample_recorder>();
//    sim.add_sampler(arb::one_probe(pid), arb::regular_schedule(interval), r->callback());
//    return r;
//}
//
//// Adds sampler to all probes
//std::shared_ptr<sample_recorder> attach_sample_recorder(arb::simulation& sim, arb::time_type interval) {
//    auto r = std::make_shared<sample_recorder>();
//    sim.add_sampler(arb::all_probes, arb::regular_schedule(interval), r->callback());
//    return r;
//}
//
//std::string sample_str(const trace_entry& s) {
//        return util::pprintf("<arbor.sample: time {} ms, value {}>", s.t, s.v);
//}

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
        .def("__str__",  [](const arb::simulation&){ return "<arbor.simulation>"; })
        .def("__repr__", [](const arb::simulation&){ return "<arbor.simulation>"; });

//    // Samples
//    pybind11::class_<trace_entry> sample(m, "sample");
//    sample
//        .def(pybind11::init<>())
//        .def_readwrite("time", &trace_entry::t, "The sample time [ms] at a specific probe.")
//        .def_readwrite("value", &trace_entry::v, "The sample value at a specific probe.")
//        .def("__str__",  &sample_str)
//        .def("__repr__", &sample_str);
//
//    // Sample recorder
//    pybind11::class_<sample_recorder, std::shared_ptr<sample_recorder>> samplerec(m, "sample_recorder");
//    samplerec
//        .def(pybind11::init<>())
//        .def("samples", &sample_recorder::samples,
//            "A list of the recorded samples of a probe with probe id.",
//            "pid"_a);
//
//    m.def("attach_sample_recorder", &attach_sample_recorder,
//        "Attach a sample recorder to an arbor simulation.\n"
//        "The recorder will record all samples from a sampling interval [ms] matching all probe ids.",
//        "simulation"_a, "interval"_a);
//
//    m.def("attach_sample_recorder_on_probe", &attach_sample_recorder_on_probe,
//        "Attach a sample recorder to an arbor simulation.\n"
//        "The recorder will record all samples from a sampling interval [ms] matching one probe id.",
//        "simulation"_a, "interval"_a, "pid"_a);
}

} // namespace pyarb
