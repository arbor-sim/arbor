#pragma once

#pragma once
#pragma once

#include <common_types.hpp>
#include <util/debug.hpp>
#include <vector>

namespace nest {
namespace mc {

class cell_interface {
public:
    virtual ~cell_interface() = default;

    /// Return the kind of cell, used for grouping into cell_groups
    virtual cell_kind const get_cell_kind() const = 0;

    /// Collect all spikes until tfinal.
    // updates the internal time state to tfinal as a side effect
    virtual std::vector<time_type> spikes_until(time_type tfinal) = 0;

    /// reset internal state;
    virtual void reset() = 0;

};
} // namespace mc
} // namespace nest
