#pragma once

#include <any>

#include <arbor/export.hpp>
#include <arbor/recipe.hpp>
#include <arbor/util/unique_any.hpp>

namespace arb {

// `tile` inherits from recipe but is not a regular recipe.
// It is used to describe a recipe on a single rank
// that will be translated to all other ranks using symmetric_recipe.
// This means it can allow connections from gid >= ncells
// which should not be allowed for other recipes.
class tile: public recipe {
public:
    virtual cell_size_type num_tiles() const { return 1; }
};

// `symmetric_recipe` takes a tile and duplicates it across
// as many ranks as tile indicates. Its functions call the
// underlying functions of tile and perform transformations
// on the results when needed.
class ARB_ARBOR_API symmetric_recipe: public recipe {
public:
    symmetric_recipe(std::unique_ptr<tile> rec): tiled_recipe_(std::move(rec)) {}

    cell_size_type num_cells() const override;

    util::unique_any get_cell_description(cell_gid_type i) const override;

    cell_kind get_cell_kind(cell_gid_type i) const override;

    std::vector<event_generator> event_generators(cell_gid_type i) const override;

    std::vector<cell_connection> connections_on(cell_gid_type i) const override;

    std::vector<probe_info> get_probes(cell_gid_type i) const override;

    std::any get_global_properties(cell_kind ck) const override;

    std::unique_ptr<tile> tiled_recipe_;
};
} // namespace arb
