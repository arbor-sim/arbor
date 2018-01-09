#pragma once

#include <cstddef>
#include <memory>
#include <unordered_map>
#include <stdexcept>

#include <cell.hpp>
#include <common_types.hpp>
#include <event_generator.hpp>
#include <util/unique_any.hpp>

namespace arb {

struct probe_info {
    cell_member_type id;
    probe_tag tag;

    // Address type will be specific to cell kind of cell `id.gid`.
    util::any address;
};

class invalid_recipe_error: public std::runtime_error {
public:
    invalid_recipe_error(std::string whatstr): std::runtime_error(std::move(whatstr)) {}
};

/* Recipe descriptions are cell-oriented: in order that the building
 * phase can be done distributedly and in order that the recipe
 * description can be built indepdently of any runtime execution
 * environment, connection end-points are represented by pairs
 * (cell index, source/target index on cell).
 */

using cell_connection_endpoint = cell_member_type;

// Note: `cell_connection` and `connection` have essentially the same data
// and represent the same thing conceptually. `cell_connection` objects
// are notionally described in terms of external cell identifiers instead
// of internal gids, but we are not making the distinction between the
// two in the current code. These two types could well be merged.

struct cell_connection {
    cell_connection_endpoint source;
    cell_connection_endpoint dest;

    float weight;
    float delay;

    cell_connection(cell_connection_endpoint src, cell_connection_endpoint dst, float w, float d):
        source(src), dest(dst), weight(w), delay(d)
    {}
};

class recipe {
public:
    virtual cell_size_type num_cells() const = 0;

    // Cell description type will be specific to cell kind of cell with given gid.
    virtual util::unique_any get_cell_description(cell_gid_type gid) const = 0;
    virtual cell_kind get_cell_kind(cell_gid_type) const = 0;

    virtual cell_size_type num_sources(cell_gid_type) const { return 0; }
    virtual cell_size_type num_targets(cell_gid_type) const { return 0; }
    virtual cell_size_type num_probes(cell_gid_type)  const { return 0; }

    virtual std::vector<event_generator_ptr> event_generators(cell_gid_type) const {
        return {};
    }
    virtual std::vector<cell_connection> connections_on(cell_gid_type) const {
        return {};
    }
    virtual probe_info get_probe(cell_member_type) const {
        throw std::logic_error("no probes");
    }


    // Global property type will be specific to given cell kind.
    virtual util::any get_global_properties(cell_kind) const { return util::any{}; };
};

} // namespace arb
