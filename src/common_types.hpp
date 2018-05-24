#pragma once

/*
 * Common definitions for index types etc. across prototype simulator
 * library. (Expect analogues in future versions to be template parameters?)
 */

#include <cstddef>
#include <functional>
#include <limits>
#include <iosfwd>
#include <type_traits>

#include <util/lexcmp_def.hpp>

namespace arb {

// For identifying cells globally.

using cell_gid_type = std::uint32_t;

// For sizes of collections of cells.

using cell_size_type = typename std::make_unsigned<cell_gid_type>::type;

// For indexes into cell-local data.
//
// Local indices for items within a particular cell-local collection should be
// zero-based and numbered contiguously.

using cell_lid_type = std::uint32_t;

// For counts of cell-local data.

using cell_local_size_type = typename std::make_unsigned<cell_lid_type>::type;

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

// Constraints on possible index conflicts can be used to select a more
// efficient indexed update, gather or scatter.

enum class index_constraint {
    none = 0,
    // For indices k[0], k[1],...:
    independent, // k[i]==k[j] => i=j.
    contiguous,  // k[i]==k[0]+i
    constant     // k[i]==k[j] ∀ i, j
};

DEFINE_LEXICOGRAPHIC_ORDERING(cell_member_type,(a.gid,a.index),(b.gid,b.index))

// For storing time values [ms]

using time_type = float;
constexpr time_type max_time = std::numeric_limits<time_type>::max();

// Extra contextual information associated with a probe.

using probe_tag = int;

// For holding counts and indexes into generated sample data.

using sample_size_type = std::int32_t;

// Enumeration used to indentify the cell type/kind, used by the model to
// group equal kinds in the same cell group.

enum class cell_kind {
    cable1d_neuron,           // Our own special mc neuron
    lif_neuron,               // Leaky-integrate and fire neuron
    regular_spike_source,     // Regular spiking source
    data_spike_source,        // Spike source from values inserted via description
};

} // namespace arb

std::ostream& operator<<(std::ostream& O, arb::cell_member_type m);
std::ostream& operator<<(std::ostream& O, arb::cell_kind k);

namespace std {
    template <> struct hash<arb::cell_member_type> {
        std::size_t operator()(const arb::cell_member_type& m) const {
            using namespace arb;
            static_assert(sizeof(std::size_t)>sizeof(cell_gid_type), "invalid size assumptions for hash of cell_member_type");

            std::size_t k = ((std::size_t)m.gid << (8*sizeof(cell_gid_type))) + m.index;
            return std::hash<std::size_t>{}(k);
        }
    };
}

