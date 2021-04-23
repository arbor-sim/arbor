#pragma once

#include <arbor/common_types.hpp>
#include <arbor/schedule.hpp>

namespace arb {

// Cell description returned by recipe::cell_description(gid) for cells with
// recipe::cell_kind(gid) returning cell_kind::spike_source

struct spike_source_cell {
    cell_tag_type source; // Label of source.
    schedule seq;

    spike_source_cell() = delete;
    spike_source_cell(cell_tag_type source, schedule seq): source(std::move(source)), seq(std::move(seq)) {};
};

} // namespace arb
