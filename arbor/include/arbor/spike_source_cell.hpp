#pragma once

#include <arbor/common_types.hpp>
#include <arbor/schedule.hpp>

namespace arb {

// Cell description returned by recipe::cell_description(gid) for cells with
// recipe::cell_kind(gid) returning cell_kind::spike_source

struct spike_source_cell {
    schedule seq;
    cell_tag_type source; // Label of source.
};

} // namespace arb
