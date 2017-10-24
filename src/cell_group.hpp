#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include <cell.hpp>
#include <common_types.hpp>
#include <event_binner.hpp>
#include <event_queue.hpp>
#include <sampling.hpp>
#include <schedule.hpp>
#include <spike.hpp>
#include <util/rangeutil.hpp>

namespace arb {

class cell_group {
public:
    virtual ~cell_group() = default;

    virtual cell_kind get_cell_kind() const = 0;

    virtual void reset() = 0;
    virtual void set_binning_policy(binning_kind policy, time_type bin_interval) = 0;
    virtual void advance(time_type tfinal, time_type dt, std::size_t epoch) = 0;

    // Pass events to be delivered to targets in the cell group in a future epoch.
    // events:
    //    An unsorted vector of post-synaptic events is maintained for each gid
    //    on the local domain. These event lists are stored in a vector, with one
    //    entry for each gid. Event lists for a cell group are contiguous in the
    //    vector, in same order that input gid were provided to the cell_group
    //    constructor.
    // tfinal:
    //    The final time for the current integration epoch. This may be used
    //    by the cell_group implementation to omptimise event queue wrangling.
    // epoch:
    //    The current integration epoch. Events in events are due for delivery
    //    in epoch+1 and later.
    virtual void enqueue_events(
            util::subrange_view_type<std::vector<std::vector<postsynaptic_spike_event>>> events,
            time_type tfinal,
            std::size_t epoch) = 0;
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
