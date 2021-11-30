#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include <arbor/common_types.hpp>
#include <arbor/sampling.hpp>
#include <arbor/schedule.hpp>
#include <arbor/spike.hpp>
#include <arbor/spike_event.hpp>

#include "epoch.hpp"
#include "event_binner.hpp"
#include "util/rangeutil.hpp"

// The specialized cell_group constructors are expected to accept at least:
// - The gid vector of the cells belonging to the cell_group.
// - The recipe.
// - 2 cell_label_range objects, one for the targets and one for the sources,
//   that are to be filled during the construction of the cell group. These
//   ranges are needed to map (gid, label) pairs to their corresponding lid sets.
namespace arb {

using event_lane_subrange = util::subrange_view_type<std::vector<pse_vector>>;

class cell_group {
public:
    virtual ~cell_group() = default;

    virtual cell_kind get_cell_kind() const = 0;

    virtual void reset() = 0;
    virtual void set_binning_policy(binning_kind policy, time_type bin_interval) = 0;
    virtual void advance(epoch epoch, time_type dt, const event_lane_subrange& events) = 0;

    virtual const std::vector<spike>& spikes() const = 0;
    virtual void clear_spikes() = 0;

    // Sampler association methods below should be thread-safe, as they might be invoked
    // from a sampler call back called from a different cell group running on a different thread.

    virtual void add_sampler(sampler_association_handle, cell_member_predicate, schedule, sampler_function, sampling_policy) = 0;
    virtual void remove_sampler(sampler_association_handle) = 0;
    virtual void remove_all_samplers() = 0;

    // Probe metadata queries might also be called while a simulation is running, and so should
    // also be thread-safe.

    virtual std::vector<probe_metadata> get_probe_metadata(cell_member_type) const {
        return {};
    }
};

using cell_group_ptr = std::unique_ptr<cell_group>;

} // namespace arb
