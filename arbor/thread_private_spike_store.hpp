#pragma once

#include <memory>
#include <vector>

#include <arbor/common_types.hpp>
#include <arbor/spike.hpp>

#include "threading/threading.hpp"

namespace arb {

struct local_spike_store_type;

/// Handles the complexity of managing thread private buffers of spikes.
/// Internally stores one thread private buffer of spikes for each hardware thread.
/// This can be accessed directly using the get() method, which returns a reference to
/// The thread private buffer of the calling thread.
/// The insert() and gather() methods add a vector of spikes to the buffer,
/// and collate all of the buffers into a single vector respectively.
class thread_private_spike_store {
public :
    thread_private_spike_store();
    ~thread_private_spike_store();

    thread_private_spike_store(thread_private_spike_store&& t);
    thread_private_spike_store(const task_system_handle& ts);

    /// Collate all of the individual buffers into a single vector of spikes.
    /// Does not modify the buffer contents.
    std::vector<spike> gather() const;

    /// Return a reference to the thread private buffer of the calling thread
    std::vector<spike>& get();

    /// Clear all of the thread private buffers
    void clear();

    /// Append the passed spikes to the end of the thread private buffer of the
    /// calling thread
    void insert(const std::vector<spike>& spikes) {
        auto& buff = get();
        buff.insert(buff.end(), spikes.begin(), spikes.end());
    }

private :
    /// thread private storage for accumulating spikes
    std::unique_ptr<local_spike_store_type> impl_;
};

} // namespace arb
