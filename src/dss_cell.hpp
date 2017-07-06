#pragma once
#pragma once

#include <vector>

#include <common_types.hpp>
#include <util/debug.hpp>

namespace nest {
namespace mc {

/// data spike source: A cell that generates spikes provided as a vector of
/// floating point valued spiketimes at the start of a run

class dss_cell {
public:
    struct dss_cell_description {
        std::vector<time_type> spike_times;

        /// The description needs a vector of doubles for the description
        dss_cell_description(std::vector<time_type> &  spike_times):
            spike_times(spike_times)
        {}
    };

    dss_cell(dss_cell_description descr){
        spike_times_.reserve(descr.spike_times.size());
        std::copy(descr.spike_times.begin(), descr.spike_times.end(), back_inserter(spike_times_));

        // Just be save and sort the spike times, it assures that what we
        // are doing is correct and if this sort is a bottle neck we might have
        // other problems.
        std::sort(spike_times_.begin(), spike_times_.end());


        not_emit_it = spike_times_.begin();
    }

    /// Return the kind of cell, used for grouping into cell_groups
    cell_kind  get_cell_kind() const  {
        return cell_kind::data_spike_source;
    }

    /// Get  all spikes until tfinal.
    // updates the internal time state to tfinal as a side effect
    std::vector<time_type> spikes_until(time_type tfinal) {
        /*
        std::vector<time_type> spikes_this_period;

        // Collect spikes in this period
        unsigned spike_idx = not_emit_idx;
        // for loop will exit immediately at end of list
        for (; spike_idx < spike_times_.size(); ++spike_idx) {
            // Exit if we are past tfinal
            if (spike_times_[spike_idx] >= tfinal) {
                break;
            }

            spikes_this_period.push_back(spike_times_[spike_idx]);
        }
        // Save spike_idx for next function call
        not_emit_idx = spike_idx;

        return spikes_this_period;
        */
        auto first = not_emit_it;
        not_emit_it = std::find_if(
            first, spike_times_.end(),
            [tfinal](time_type t) {return t >= tfinal; }
        );

        return { first, not_emit_it };
    }

    /// reset internal point to non emitted spike to the first in the storage
    void reset() {
        not_emit_it = spike_times_.begin();
    }

private:
    // (sorted) Vector of spikes that to be emitted by this spike source
    std::vector<time_type> spike_times_;

    // Index to the first spike not yet emitted
    std::vector<time_type>::iterator not_emit_it;
};

} // namespace mc
} // namespace nest
