#pragma once

#include <cstddef>
#include <memory>
#include <stdexcept>

namespace nest {
namespace mc {

struct cell_count_info {
    cell_size_type num_sources;
    cell_size_type num_targets;
    cell_size_type num_probes;
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
};

class recipe {
public:
    virtual cell_size_type num_cells() const =0;

    virtual cell get_cell(cell_gid_type) const =0;
    virtual cell_kind get_cell_kind(cell_gid_type) const = 0;

    virtual cell_count_info get_cell_count_info(cell_gid_type) const =0;
    virtual std::vector<cell_connection> connections_on(cell_gid_type) const =0;
};


/*
 * Recipe consisting of a single, unconnected cell
 * is particularly simple. Note keeps a reference to
 * the provided cell, so be aware of life time issues.
 */

class singleton_recipe: public recipe {
public:
    singleton_recipe(const cell& the_cell): cell_(the_cell) {}

    cell_size_type num_cells() const override {
        return 1;
    }

    cell get_cell(cell_gid_type) const override {
        return cell(clone_cell, cell_);
    }

    cell_kind get_cell_kind(cell_gid_type) const override {
        return cell_.get_cell_kind();
    }

    cell_count_info get_cell_count_info(cell_gid_type) const override {
        cell_count_info k;
        k.num_sources = cell_.detectors().size();
        k.num_targets = cell_.synapses().size();
        k.num_probes = cell_.probes().size();

        return k;
    }

    std::vector<cell_connection> connections_on(cell_gid_type) const override {
        return std::vector<cell_connection>{};
    }

private:
    const cell& cell_;
};

} // namespace mc
} // namespace nest
