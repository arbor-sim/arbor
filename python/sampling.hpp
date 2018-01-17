#pragma once

#include <model.hpp>

// Helper type for recording spikes from a model.
// This type is wrapped in Python, to expose spike_recorder::spike_store.
struct spike_recorder {
    using export_func = arb::model::spike_export_function;
    using spike_vec = std::vector<arb::spike>;
    std::shared_ptr<spike_vec> spike_store;

    export_func callback() {
        // The callback returned should capture by value, so that the shared
        // pointer is held by both the spike_recorder instance, and by the
        // callback, so if the spike_recorder is destructed in the calling
        // Python code, callbacks to this lambda will not seg fault.
        return [=](const spike_vec& spikes) {
            spike_store->insert(spike_store->end(), spikes.begin(), spikes.end());
        };
    }
};

// Returns a spike_recorder that has been registered with the model m.
spike_recorder make_spike_recorder(arb::model& m);
