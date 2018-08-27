#pragma once

#include <cstddef>
#include <memory>
#include <unordered_map>
#include <stdexcept>

#include <arbor/recipe.hpp>

namespace arb {

class tiled_recipe: public recipe {
public:
    virtual cell_size_type num_tiles() const { return 1; }
};

template<class overriden_tiled_recipe>
class symmetric_recipe: public recipe {
public:
    symmetric_recipe(overriden_tiled_recipe rec): tiled_recipe_(rec) {}

    cell_size_type num_cells() const override {
        return tiled_recipe_.num_cells() * tiled_recipe_.num_tiles();
    }

    util::unique_any get_cell_description(cell_gid_type i) const override {
        return tiled_recipe_.get_cell_description(i % tiled_recipe_.num_cells());
    }

    cell_kind get_cell_kind(cell_gid_type i) const override {
        return tiled_recipe_.get_cell_kind(i % tiled_recipe_.num_cells());
    }

    cell_size_type num_sources(cell_gid_type i) const override {
        return tiled_recipe_.num_sources(i % tiled_recipe_.num_cells());
    }

    cell_size_type num_targets(cell_gid_type i) const override {
        return tiled_recipe_.num_targets(i % tiled_recipe_.num_cells());
    }

    cell_size_type num_probes(cell_gid_type i) const override {
        return tiled_recipe_.num_probes(i % tiled_recipe_.num_cells());
    }

    std::vector<event_generator> event_generators(cell_gid_type i) const override {
        return tiled_recipe_.event_generators(i % tiled_recipe_.num_cells());
    }

    std::vector<cell_connection> connections_on(cell_gid_type i) const override {
        int n_local = tiled_recipe_.num_cells();
        int offset = (i / n_local) * n_local;

        int n_global = num_cells();

        std::vector<cell_connection> conns = tiled_recipe_.connections_on(i % n_local);

        for (int j = 0; j < conns.size(); j++) {
            conns[j].source = (conns[j].source + offset) % n_global;
            conns[j].dest = (conns[j].dest + offset) % n_global;
        }
        return conns;
    }


    probe_info get_probe(cell_member_type probe_id) const override {
        probe_id.gid %= tiled_recipe_.num_cells();
        return tiled_recipe_.get_probe(probe_id);
    }

    util::any get_global_properties(cell_kind ck) const override {
        return tiled_recipe_.get_global_properties(ck);
    };

    overriden_tiled_recipe tiled_recipe_;
};
} // namespace arb
