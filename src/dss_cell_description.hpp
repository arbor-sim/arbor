#pragma once
#pragma once

#include <vector>

#include <common_types.hpp>
#include <util/debug.hpp>

namespace nest {
namespace mc {

/// Description for a data spike source: A cell that generates spikes provided as a vector of
/// floating point valued spiketimes at the start of a run

struct dss_cell_description {
    std::vector<time_type> spike_times;

    /// The description needs a vector of doubles for the description
    dss_cell_description(std::vector<time_type> &  spike_times) :
        spike_times(spike_times)
    {}
};


} // namespace mc
} // namespace nest
