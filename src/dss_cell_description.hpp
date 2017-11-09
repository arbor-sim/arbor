#pragma once

#include <vector>

#include <common_types.hpp>

namespace arb {

/// Description for a data spike source: a cell that generates spikes provided as a vector of
/// spike times at the start of a run.

struct dss_cell_description {
    std::vector<time_type> spike_times;

    /// The description needs a vector of doubles for the description
    dss_cell_description(std::vector<time_type> spike_times):
        spike_times(std::move(spike_times))
    {}
};

} // namespace arb
