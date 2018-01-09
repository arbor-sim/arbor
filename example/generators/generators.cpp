/*
 * A miniapp that demonstrates how to use event generators.
 *
 * The miniapp builds a simple model
 */

#include <common_types.hpp>
#include <recipe.hpp>

using arb::cell_gid_type;
using arb::cell_lid_type;
using arb::cell_size_type;
using arb::cell_kind;

class recipe: public arb::recipe {
    cell_size_type num_cells() const override {
        return 1;
    }

    arb::util::unique_any get_cell_description(cell_gid_type gid) const override {
    }
    cell_kind get_cell_kind(cell_gid_type) const override {
    }

    cell_size_type num_sources(cell_gid_type) const override {
    }
    cell_size_type num_targets(cell_gid_type) const override {
    }

    std::vector<arb::event_generator_ptr> event_generators(cell_gid_type) const override {
    }

    std::vector<arb::cell_connection> connections_on(cell_gid_type) const override {
    }

    // Global property type will be specific to given cell kind.
    //util::any get_global_properties(cell_kind) const { return util::any{}; };
};

int main() {
}
