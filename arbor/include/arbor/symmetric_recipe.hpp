#pragma once

#include <cstddef>
#include <memory>
#include <unordered_map>
#include <stdexcept>

#include <arbor/recipe.hpp>

namespace arb {

// tile inherits from recipe but is not a regular recipe.
// It is used to describe a recipe on a single rank
// that will be translated to all other ranks using symmetric_recipe.
// This means it can allow connections from gid >= ncells
// which should not be allowed for other recipes.
class tile: public recipe {
public:
    virtual cell_size_type num_tiles() const { return 1; }
};

// symmetric recipe takes a tile and duplicates it across
// as many ranks as tile indicates. Its functions call the
// underlying functions of tile and perform transformations
// on the results when needed.
class symmetric_recipe: public recipe {
public:
    symmetric_recipe(std::unique_ptr<tile> rec): tiled_recipe_(std::move(rec)) {}

    cell_size_type num_cells() const override {
        return tiled_recipe_->num_cells() * tiled_recipe_->num_tiles();
    }

    util::unique_any get_cell_description(cell_gid_type i) const override {
        return tiled_recipe_->get_cell_description(i % tiled_recipe_->num_cells());
    }

    cell_kind get_cell_kind(cell_gid_type i) const override {
        return tiled_recipe_->get_cell_kind(i % tiled_recipe_->num_cells());
    }

    cell_size_type num_sources(cell_gid_type i) const override {
        return tiled_recipe_->num_sources(i % tiled_recipe_->num_cells());
    }

    cell_size_type num_targets(cell_gid_type i) const override {
        return tiled_recipe_->num_targets(i % tiled_recipe_->num_cells());
    }

    cell_size_type num_probes(cell_gid_type i) const override {
        return tiled_recipe_->num_probes(i % tiled_recipe_->num_cells());
    }

    // Only function that calls the underlying tile's function on the same gid.
    // This is because applying transformations to event generators is not straightforward.
    std::vector<event_generator> event_generators(cell_gid_type i) const override {
        return tiled_recipe_->event_generators(i);
    }

    // Take connections_on from the original tile recipe for the cell we are duplicating.
    // Transate the source and destination gids
    std::vector<cell_connection> connections_on(cell_gid_type i) const override {
        int n_local = tiled_recipe_->num_cells();
        int n_global = num_cells();
        int offset = (i / n_local) * n_local;

        std::vector<cell_connection> conns = tiled_recipe_->connections_on(i % n_local);

        for (unsigned j = 0; j < conns.size(); j++) {
            conns[j].source.gid = (conns[j].source.gid + offset) % n_global;
            conns[j].dest.gid = (conns[j].dest.gid + offset) % n_global;
        }
        return conns;
    }

    probe_info get_probe(cell_member_type probe_id) const override {
        probe_id.gid %= tiled_recipe_->num_cells();
        return tiled_recipe_->get_probe(probe_id);
    }

    util::any get_global_properties(cell_kind ck) const override {
        return tiled_recipe_->get_global_properties(ck);
    };

    std::unique_ptr<tile> tiled_recipe_;
};
} // namespace arb
