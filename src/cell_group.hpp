#pragma once

#include <memory>
#include <vector>

#include <cell.hpp>
#include <common_types.hpp>
#include <event_binner.hpp>
#include <event_queue.hpp>
#include <sampling.hpp>
#include <schedule.hpp>
#include <spike.hpp>

namespace arb {

class cell_group {
public:
    virtual ~cell_group() = default;

    virtual cell_kind get_cell_kind() const = 0;

    virtual void reset() = 0;
    virtual void set_binning_policy(binning_kind policy, time_type bin_interval) = 0;
    virtual void advance(time_type tfinal, time_type dt) = 0;
    virtual void enqueue_events(const std::vector<postsynaptic_spike_event>& events) = 0;
    virtual const std::vector<spike>& spikes() const = 0;
    virtual void clear_spikes() = 0;

    // Sampler association methods below should be thread-safe, as they might be invoked
    // from a sampler call back called from a different cell group running on a different thread.

    virtual void add_sampler(sampler_association_handle, cell_member_predicate, schedule, sampler_function, sampling_policy) = 0;
    virtual void remove_sampler(sampler_association_handle) = 0;
    virtual void remove_all_samplers() = 0;
};

using cell_group_ptr = std::unique_ptr<cell_group>;

template <typename T, typename... Args>
cell_group_ptr make_cell_group(Args&&... args) {
    return cell_group_ptr(new T(std::forward<Args>(args)...));
}

} // namespace arb
