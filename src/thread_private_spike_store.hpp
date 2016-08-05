#pragma once

#include <vector>

#include <common_types.hpp>
#include <spike.hpp>
#include <threading/threading.hpp>

namespace nest {
namespace mc {

/// Handles the complexity of managing thread private buffers of spikes.
/// Internally stores one thread private buffer of spikes for each hardware thread.
/// This can be accessed directly using the get() method, which returns a reference to
/// The thread private buffer of the calling thread.
/// The insert() and gather() methods add a vector of spikes to the buffer,
/// and collate all of the buffers into a single vector respectively.
template <typename Time>
class thread_private_spike_store {
public :
    using id_type = cell_gid_type;
    using time_type = Time;
    using spike_type = spike<cell_member_type, time_type>;

    /// Collate all of the individual buffers into a single vector of spikes.
    /// Does not modify the buffer contents.
    std::vector<spike_type> gather() const {
        std::vector<spike_type> spikes;
        unsigned num_spikes = 0u;
        for (auto& b : buffers_) {
            num_spikes += b.size();
        }
        spikes.reserve(num_spikes);

        for (auto& b : buffers_) {
            spikes.insert(spikes.begin(), b.begin(), b.end());
        }

        return spikes;
    }

    /// Return a reference to the thread private buffer of the calling thread
    std::vector<spike_type>& get() {
        return buffers_.local();
    }

    /// Return a reference to the thread private buffer of the calling thread
    const std::vector<spike_type>& get() const {
        return buffers_.local();
    }

    /// Clear all of the thread private buffers
    void clear() {
        for (auto& b : buffers_) {
            b.clear();
        }
    }

    /// Append the passed spikes to the end of the thread private buffer of the
    /// calling thread
    void insert(const std::vector<spike_type>& spikes) {
        auto& buff = get();
        buff.insert(buff.end(), spikes.begin(), spikes.end());
    }

private :
    /// thread private storage for accumulating spikes
    using local_spike_store_type =
        threading::enumerable_thread_specific<std::vector<spike_type>>;

    local_spike_store_type buffers_;
};

} // namespace mc
} // namespace nest
