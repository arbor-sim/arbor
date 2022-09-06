#pragma once

#include <arbor/common_types.hpp>
#include <arbor/export.hpp>
#include <arbor/schedule.hpp>

namespace arb {

// Cell description returned by recipe::cell_description(gid) for cells with
// recipe::cell_kind(gid) returning cell_kind::spike_source

struct ARB_SYMBOL_VISIBLE spike_source_cell {
    cell_tag_type source; // Label of source.
    std::vector<schedule> seqs;

    spike_source_cell() = delete;
    template<typename... Seqs>
    spike_source_cell(cell_tag_type source, Seqs&&... seqs): source(std::move(source)), seqs{std::forward<Seqs>(seqs)...} {}
    spike_source_cell(cell_tag_type source, std::vector<schedule> seqs): source(std::move(source)), seqs(std::move(seqs)) {}
};

} // namespace arb
