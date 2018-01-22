#pragma once

#include <model.hpp>

// A functor that models arb::model::spike_export_function.
// Holds a shared pointer to the spike_vec used to store the spikes, so that if
// the spike_vec in spike_recorder is garbage collected in Python, stores will
// not seg fault.
struct spike_callback {
    using spike_vec = std::vector<arb::spike>;

    std::shared_ptr<spike_vec> spike_store;

    spike_callback(const std::shared_ptr<spike_vec>& state):
        spike_store(state) {}

    void operator() (const spike_vec& spikes) {
        spike_store->insert(spike_store->end(), spikes.begin(), spikes.end());
    };
};

// Helper type for recording spikes from a model.
// This type is wrapped in Python, to expose spike_recorder::spike_store.
struct spike_recorder {
    using export_func = arb::model::spike_export_function;
    using spike_vec = std::vector<arb::spike>;
    std::shared_ptr<spike_vec> spike_store;

    spike_callback callback() {
        // initialize the spike_store
        spike_store = std::make_shared<spike_vec>();

        // The callback makes a copy spike_store, so that the shared
        // pointer is held by both the spike_recorder instance, and by the
        // callback, so if the spike_recorder is destructed in the calling
        // Python code, attempts to write to spike_store inside the callback
        // will not seg fault.
        return spike_callback(spike_store);
    }
};

// Returns a spike_recorder that has been registered with the model m.
std::shared_ptr<spike_recorder> make_spike_recorder(arb::model& m);
