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
#include <string>
#include <type_traits>

#include <arbor/util/hash_def.hpp>
#include <arbor/export.hpp>

namespace arb {

// Internal hashes use this 64bit id

using hash_type = std::size_t;

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
    auto operator<=>(const cell_member_type&) const = default;
};

// Pair of indexes that describe range of local indices.

struct lid_range {
    cell_lid_type begin = 0;
    cell_lid_type end = 0;
    lid_range() = default;
    lid_range(cell_lid_type b, cell_lid_type e): begin(b), end(e) {}
    auto operator<=>(const lid_range&) const = default;
};

// Global range of indices with given step size.

struct gid_range {
    cell_gid_type begin = 0;
    cell_gid_type end = 0;
    cell_gid_type step = 1;
    gid_range() = default;
    gid_range(cell_gid_type b, cell_gid_type e): begin(b), end(e), step(1) {}
    gid_range(cell_gid_type b, cell_gid_type e, cell_gid_type s): begin(b), end(e), step(s) {}
};

// Policy for selecting a cell_lid_type from a range of possible values.

enum class lid_selection_policy {
    round_robin,
	round_robin_halt,
    assert_univalent // throw if the range of possible lids is wider than 1
};

// For referring to a labeled placement on an unspecified cell.
// The placement may be associated with multiple locations, the policy
// is used to select a specific location.

struct cell_local_label_type {
    cell_tag_type tag;
    lid_selection_policy policy;

    cell_local_label_type(cell_tag_type tag, lid_selection_policy policy=lid_selection_policy::assert_univalent):
        tag(std::move(tag)), policy(policy) {}
};

// For referring to a labeled placement on a cell identified by gid.

struct cell_global_label_type {
    cell_gid_type gid;
    cell_local_label_type label;

    cell_global_label_type(cell_gid_type gid, cell_local_label_type label): gid(gid), label(std::move(label)) {}
    cell_global_label_type(cell_gid_type gid, cell_tag_type tag): gid(gid), label(std::move(tag)) {}
    cell_global_label_type(cell_gid_type gid, cell_tag_type tag, lid_selection_policy policy): gid(gid), label(std::move(tag), policy) {}
};

// User facing handle for referring to probes and similar

struct cell_address_type {
    cell_gid_type gid;
    cell_tag_type tag;

    // NOTE: We _need_ this explicitly to avoid `cell_address_type{42, 0}` from triggering
    cell_address_type(cell_gid_type, std::nullptr_t) = delete;
    cell_address_type(cell_gid_type gid_, cell_tag_type tag_): gid(gid_), tag(std::move(tag_)) {}

    // NOTE: We _really_ do not want this to occur, *EVER*.
    cell_address_type() = delete;

    cell_address_type(const cell_address_type&) = default;
    cell_address_type(cell_address_type&&) = default;

    cell_address_type& operator=(const cell_address_type&) = default;
    cell_address_type& operator=(cell_address_type&&) = default;

    auto operator<=>(const cell_address_type&) const = default;
};

struct cell_remote_label_type {
    cell_gid_type rid;     // remote id
    cell_lid_type index = 0; // index on remote id

    auto operator<=>(const cell_remote_label_type&) const = default;
};

// For storing time values [ms]

using time_type = double;
constexpr time_type terminal_time = std::numeric_limits<time_type>::max();

// For holding counts and indexes into generated sample data.
using sample_index_type = std::int32_t;
using sample_size_type  = std::uint32_t;

// Enumeration for execution back-end targets, as specified in domain decompositions.
// NOTE(important): Given in order of priority, ie we will attempt schedule gpu before
//                  MC groups, for reasons of effiency. Ugly, but as we do not have more
//                  backends, this is OK for now.
enum class backend_kind {
    gpu,         //  Use gpu back-end when supported by cell_group implementation.
    multicore,   //  Use multicore back-end for all computation.
};

// Enumeration used to indentify the cell type/kind, used by the model to
// group equal kinds in the same cell group.

enum class ARB_SYMBOL_VISIBLE cell_kind {
    cable,        // Our own special mc neuron.
    lif,          // Leaky-integrate and fire neuron.
    spike_source, // Cell that generates spikes at a user-supplied sequence of time points.
    benchmark,    // Proxy cell used for benchmarking.
};

ARB_ARBOR_API std::ostream& operator<<(std::ostream& o, lid_selection_policy m);
ARB_ARBOR_API std::ostream& operator<<(std::ostream& o, cell_member_type m);
ARB_ARBOR_API std::ostream& operator<<(std::ostream& o, cell_kind k);
ARB_ARBOR_API std::ostream& operator<<(std::ostream& o, backend_kind k);

} // namespace arb

ARB_DEFINE_HASH(arb::cell_address_type, a.gid, a.tag)
ARB_DEFINE_HASH(arb::cell_member_type, a.gid, a.index)
