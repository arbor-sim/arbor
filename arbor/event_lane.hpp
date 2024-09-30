#pragma once

#include <arbor/spike_event.hpp>

#include "util/rangeutil.hpp"

namespace arb {

using event_lane_subrange = util::subrange_view_type<std::vector<pse_vector>>;

} // namespace arb
