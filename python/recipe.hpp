#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <recipe.hpp>

namespace arb {
class py_recipe: public arb::recipe {
public:
    using recipe::recipe;

    cell_size_type num_cells() const override {
        PYBIND11_OVERLOAD_PURE(cell_size_type, recipe, num_cells);
    }

    // Cell description type will be specific to cell kind of cell with given gid.
    util::any get_cell_description(cell_gid_type gid) const override {
        PYBIND11_OVERLOAD_PURE(util::any, recipe, get_cell_description, gid);
    }

    cell_kind get_cell_kind(cell_gid_type gid) const override {
        PYBIND11_OVERLOAD_PURE(cell_kind, recipe, get_cell_kind, gid);
    }

    // TODO: the following need to be implemented

    //cell_size_type num_sources(cell_gid_type gid) const {return 0;}
    //cell_size_type num_targets(cell_gid_type gid) const {return 0;}
    //cell_size_type num_probes(cell_gid_type gid) const {return 0;}

    //std::vector<event_generator_ptr> event_generators(cell_gid_type) const {return {};};

    //std::vector<cell_connection> connections_on(cell_gid_type) const {return {};};
    //probe_info get_probe(cell_member_type probe_id) const {return {};};

    // Global property type will be specific to given cell kind.
    //util::any get_global_properties(cell_kind) const { return util::any{}; };
};

} // namespace arb

