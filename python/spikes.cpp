#include <memory>
#include <vector>

#include <arbor/spike.hpp>
#include <arbor/simulation.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "strings.hpp"

namespace arb {
namespace py {

// A functor that models arb::spike_export_function.
// Holds a shared pointer to the spike_vec used to store the spikes, so that if
// the spike_vec in spike_recorder is garbage collected in Python, stores will
// not seg fault.
struct spike_callback {
    using spike_vec = std::vector<arb::spike>;

    std::shared_ptr<spike_vec> spike_store;

    spike_callback(const std::shared_ptr<spike_vec>& state):
        spike_store(state)
    {}

    void operator() (const spike_vec& spikes) {
        spike_store->insert(spike_store->end(), spikes.begin(), spikes.end());
    };
};

// Helper type for recording spikes from a simulation.
// This type is wrapped in Python, to expose spike_recorder::spike_store.
struct spike_recorder {
    using spike_vec = std::vector<arb::spike>;
    std::shared_ptr<spike_vec> spike_store;

    spike_callback callback() {
        // initialize the spike_store
        spike_store = std::make_shared<spike_vec>();

        // The callback holds a copy of spike_store, i.e. the shared
        // pointer is held by both the spike_recorder and the callback, so if
        // the spike_recorder is destructed in the calling Python code, attempts
        // to write to spike_store inside the callback will not seg fault.
        return spike_callback(spike_store);
    }
};

std::shared_ptr<spike_recorder> make_spike_recorder(arb::simulation& sim) {
    auto r = std::make_shared<spike_recorder>();
    sim.set_global_spike_callback(r->callback());
    return r;
}

void register_spike_handling(pybind11::module& m) {
    pybind11::class_<arb::spike> spike(m, "spike");
    spike
        .def(pybind11::init<>())
        .def_readwrite("source", &arb::spike::source)
        .def_readwrite("time", &arb::spike::time)
        .def("__str__",  &spike_string)
        .def("__repr__", &spike_string);

    // Use shared_ptr for spike_recorder, so that all copies of a recorder will
    // see the spikes from the simulation with which the recorder's callback has been
    // registered.
    pybind11::class_<spike_recorder, std::shared_ptr<spike_recorder>> sprec(m, "spike_recorder");
    sprec
        .def(pybind11::init<>())
        .def_property_readonly("spikes", [](const spike_recorder& s) {return *(s.spike_store.get());} );

    m.def("make_spike_recorder", &make_spike_recorder);
}

} // namespace py
} // namespace arb
