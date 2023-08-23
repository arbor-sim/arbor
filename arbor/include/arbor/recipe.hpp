#pragma once

#include <any>
#include <utility>
#include <vector>

#include <arbor/export.hpp>
#include <arbor/common_types.hpp>
#include <arbor/event_generator.hpp>
#include <arbor/util/unique_any.hpp>

namespace arb {

struct probe_info {
    cell_tag_type tag;

    // Address type will be specific to cell kind of cell `id.gid`.
    std::any address;

    probe_info(probe_info&) = default;
    probe_info(const probe_info&) = default;
    probe_info(probe_info&&) = default;

    // Implicit ctor uses tag of zero.
    template <typename X>
    probe_info(X&& x, const cell_tag_type& tag):
        tag(tag), address(std::forward<X>(x)) {}
};

/* Recipe descriptions are cell-oriented: in order that the building
 * phase can be distributed, and in order that the recipe description
 * can be built indepedently of any runtime execution environment.
 */

// Note: `cell_connection` and `connection` have essentially the same data
// and represent the same thing conceptually. `cell_connection` objects
// are notionally described in terms of external cell identifiers instead
// of internal gids, but we are not making the distinction between the
// two in the current code. These two types could well be merged.

template<typename L>
struct cell_connection_base {
    // Connection end-points are represented by pairs
    // (cell index, source/target index on cell).
    L source;
    cell_local_label_type target;

    float weight;
    float delay;

    cell_connection_base(L src, cell_local_label_type dst, float w, float d):
        source(std::move(src)), target(std::move(dst)), weight(w), delay(d) {}
};

using cell_connection     = cell_connection_base<cell_global_label_type>;
using ext_cell_connection = cell_connection_base<cell_remote_label_type>;

struct gap_junction_connection {
    cell_global_label_type peer;
    cell_local_label_type local;
    double weight; //unit-less

    gap_junction_connection(cell_global_label_type peer, cell_local_label_type local, double g):
        peer(std::move(peer)), local(std::move(local)), weight(g) {}
};

struct ARB_ARBOR_API has_gap_junctions {
    virtual std::vector<gap_junction_connection> gap_junctions_on(cell_gid_type) const {
        return {};
    }
    virtual ~has_gap_junctions() {}
};

struct ARB_ARBOR_API has_synapses {
    virtual std::vector<cell_connection> connections_on(cell_gid_type) const {
        return {};
    }
    virtual ~has_synapses() {}
};

struct ARB_ARBOR_API has_external_synapses {
    virtual std::vector<ext_cell_connection> external_connections_on(cell_gid_type) const {
        return {};
    }
};

struct ARB_ARBOR_API has_probes {
    virtual std::vector<probe_info> get_probes(cell_gid_type gid) const {
        return {};
    }
    virtual ~has_probes() {}
};

struct ARB_ARBOR_API has_generators {
    virtual std::vector<event_generator> event_generators(cell_gid_type) const {
        return {};
    }
    virtual ~has_generators() {}
};

// Toppings allow updating a simulation
struct ARB_ARBOR_API connectivity:
        public has_synapses,
               has_external_synapses,
               has_generators {
    virtual ~connectivity() {}
};

// Recipes allow building a simulation by lazy queries
struct ARB_ARBOR_API recipe: public has_gap_junctions, has_probes, connectivity {
    // number of cells to build
    virtual cell_size_type num_cells() const = 0;
    // Cell description type will be specific to cell kind of cell with given gid.
    virtual util::unique_any get_cell_description(cell_gid_type gid) const = 0;
    // Query cell kind per gid
    virtual cell_kind get_cell_kind(cell_gid_type) const = 0;
    // Global property type will be specific to given cell kind.
    virtual std::any get_global_properties(cell_kind) const { return std::any{}; };

    virtual ~recipe() {}
};

} // namespace arb
