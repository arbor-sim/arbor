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

#include <arbor/util/lexcmp_def.hpp>

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

ARB_DEFINE_LEXICOGRAPHIC_ORDERING(cell_member_type,(a.gid,a.index),(b.gid,b.index))

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

namespace std {
    template <> struct hash<arb::cell_member_type> {
        std::size_t operator()(const arb::cell_member_type& m) const {
            using namespace arb;
            if (sizeof(std::size_t)>sizeof(cell_gid_type)) {
                constexpr unsigned shift = 8*sizeof(cell_gid_type);

                std::size_t k = m.gid;
                k <<= (shift/2); // dodge gcc shift warning when other branch taken
                k <<= (shift/2);
                k += m.index;
                return std::hash<std::size_t>{}(k);
            }
            else {
                constexpr std::size_t prime1 = 93481;
                constexpr std::size_t prime2 = 54517;

                std::size_t k = prime1;
                k = k*prime2 + m.gid;
                k = k*prime2 + m.index;
                return k;
            }
        }
    };
}

