#pragma once

/*
 * Common definitions for index types etc. across prototype simulator
 * library. (Expect analogues in future versions to be template parameters?)
 */

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <iosfwd>
#include <type_traits>
#include <iostream>

#include <arbor/util/lexcmp_def.hpp>
#include <arbor/util/hash_def.hpp>
#include <utility>

namespace arb {

// For identifying cells globally.

using cell_gid_type = std::uint32_t;

// For sizes of collections of cells.

using cell_size_type = std::make_unsigned_t<cell_gid_type>;

// For indexes into cell-local data.
//
// Local indices for items within a particular cell-local collection should be
// zero-based and numbered contiguously.

using cell_lid_type = std::uint32_t;

// Local labels for items within a particular cell-local collection
using cell_tag_type = std::string;

// For counts of cell-local data.

using cell_local_size_type = std::make_unsigned_t<cell_lid_type>;

// For global identification of an item of cell local data.
//
// Items of cell_member_type must:
//
//  * be associated with a unique cell, identified by the member `gid`
//    (see: cell_gid_type);
//
//  * identify an item within a cell-local collection by the member `index`
//    (see: cell_lid_type).

struct cell_member_type {
    cell_gid_type gid;
    cell_lid_type index;
};

// Items of cell_label_type must:
//
//  * be associated with a unique cell, identified by the member `gid`
//    (see: cell_gid_type);
//
//  * identify a labeled item within a cell-local collection by the label `label`
//    (see: cell_tag_type).

struct cell_label_type {
    cell_gid_type gid;
    cell_tag_type label;
};

// Pair of indexes that describe range of local indices.
// Returned by cable_cell::place() calls, so that the caller can
// refer to targets, detectors, etc on the cell.
struct lid_range {
    cell_lid_type begin;
    cell_lid_type end;
    lid_range() {};
    lid_range(cell_lid_type b, cell_lid_type e):
        begin(b), end(e) {}
};

enum class cell_lid_policy {
    round_robin,
    assert_univalent
};

ARB_DEFINE_LEXICOGRAPHIC_ORDERING(cell_member_type,(a.gid,a.index),(b.gid,b.index))
ARB_DEFINE_LEXICOGRAPHIC_ORDERING(lid_range,(a.begin, a.end),(b.begin,b.end))

struct cell_labeled_range {
    std::vector<cell_gid_type> gids;
    std::vector<cell_tag_type> labels;
    std::vector<lid_range> ranges;

    mutable std::vector<cell_lid_type> indices;

    cell_labeled_range() = delete;
    cell_labeled_range(std::vector<cell_gid_type> gids, std::vector<cell_tag_type> lbls, std::vector<lid_range> rngs):
        gids(std::move(gids)), labels(std::move(lbls)), ranges(std::move(rngs)), indices(ranges.size(), 0) {};

    std::optional<cell_lid_type> get_lid(const cell_label_type& elem, cell_lid_policy policy=cell_lid_policy::round_robin) const {
        auto it = std::lower_bound(gids.begin(), gids.end(), elem.gid);
        if (*it != elem.gid) return std::nullopt;

        auto first = it - gids.begin();
        auto last  = std::upper_bound(gids.begin(), gids.end(), elem.gid) - gids.begin();

        auto lit = std::lower_bound(labels.begin()+first, labels.begin()+last, elem.label);
        if (*lit != elem.label) return std::nullopt;

        auto label_idx = lit - labels.begin();

        auto range = ranges[label_idx];
        auto size = range.end - range.begin;

        switch (policy) {
            case cell_lid_policy::round_robin:
            {
                auto idx = indices[label_idx];
                indices[label_idx] = (idx+1)%size;
                return idx + range.begin;
            }
            case cell_lid_policy::assert_univalent:
            {
                if (size != 1) {
                    return std::nullopt;
                }
                return range.begin;
            }
        }
        return std::nullopt;
    }
};

// For storing time values [ms]

using time_type = double;
constexpr time_type terminal_time = std::numeric_limits<time_type>::max();

// Extra contextual information associated with a probe.

using probe_tag = int;

// For holding counts and indexes into generated sample data.

using sample_size_type = std::int32_t;

// Enumeration for execution back-end targets, as specified in domain decompositions.

enum class backend_kind {
    multicore,   //  Use multicore back-end for all computation.
    gpu          //  Use gpu back-end when supported by cell_group implementation.
};

// Enumeration used to indentify the cell type/kind, used by the model to
// group equal kinds in the same cell group.

enum class cell_kind {
    cable,   // Our own special mc neuron.
    lif,       // Leaky-integrate and fire neuron.
    spike_source,     // Cell that generates spikes at a user-supplied sequence of time points.
    benchmark,        // Proxy cell used for benchmarking.
};

// Enumeration for event time binning policy.

enum class binning_kind {
    none,
    regular,   // => round time down to multiple of binning interval.
    following, // => round times down to previous event if within binning interval.
};

std::ostream& operator<<(std::ostream& o, cell_member_type m);
std::ostream& operator<<(std::ostream& o, cell_kind k);
std::ostream& operator<<(std::ostream& o, backend_kind k);

} // namespace arb

ARB_DEFINE_HASH(arb::cell_member_type, a.gid, a.index)
