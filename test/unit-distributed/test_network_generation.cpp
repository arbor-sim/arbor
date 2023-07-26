#include <gtest/gtest.h>
#include <unordered_map>
#include <vector>

#include <arbor/cable_cell.hpp>
#include <arbor/lif_cell.hpp>
#include <arbor/spike_source_cell.hpp>
#include <arbor/cable_cell_param.hpp>
#include <arbor/common_types.hpp>
#include <arbor/load_balance.hpp>
#include <arbor/morph/segment_tree.hpp>
#include <arbor/network_generation.hpp>
#include <arbor/recipe.hpp>
#include <arborio/label_parse.hpp>

#include "test.hpp"
#include "execution_context.hpp"

using namespace arb;
using namespace arborio::literals;


namespace {
// Create alternatingly a cable, lif and spike source cell with at most one source or destination
class network_test_recipe: public arb::recipe {
public:
    network_test_recipe(unsigned num_cells,
        network_selection selection,
        network_value weight,
        network_value delay):
        num_cells_(num_cells),
        selection_(selection),
        weight_(weight),
        delay_(delay) {
        gprop_.default_parameters = arb::neuron_parameter_defaults;
    }

    cell_size_type num_cells() const override {
        return num_cells_;
    }

    arb::util::unique_any get_cell_description(cell_gid_type gid) const override {
	if(gid % 3 == 1) {
	    return lif_cell("source", "target");
	}
	if(gid % 3 == 2) {
	    return spike_source_cell("spike_source");
	}

	// cable cell
        int stag = 1;                 // soma tag
        int dtag = 3;                 // Dendrite tag.
        double srad = 12.6157 / 2.0;  // soma radius
        double drad = 0.5;            // Diameter of 1 μm for each dendrite cable.
        arb::segment_tree tree;
        tree.append(
            arb::mnpos, {0, 0, -srad, srad}, {0, 0, srad, srad}, stag);  // For area of 500 μm².
        tree.append(0, {0, 0, 2 * srad, drad}, dtag);

        arb::label_dict labels;
        labels.set("soma", reg::tagged(stag));
        labels.set("dend", reg::tagged(dtag));

        auto decor = arb::decor{}
                         .paint("soma"_lab, arb::density("hh"))
                         .paint("dend"_lab, arb::density("pas"))
                         .set_default(arb::axial_resistivity{100})  // [Ω·cm]
                         .place(arb::mlocation{0, 0}, arb::threshold_detector{10}, "detector")
                         .place(arb::mlocation{0, 0.5}, arb::synapse("expsyn"), "primary_syn");

        return arb::cable_cell(arb::morphology(tree), decor, labels);
    }

    cell_kind get_cell_kind(cell_gid_type gid) const override {
	if(gid % 3 == 1) {
	    return cell_kind::lif;
	}
	if(gid % 3 == 2) {
	    return cell_kind::spike_source;
	}

        return cell_kind::cable;
    }

    arb::isometry get_cell_isometry(cell_gid_type gid) const override {
        // place cells with equal distance on a circle
        const double angle = 2 * 3.1415926535897932 * gid / num_cells_;
        const double radius = 500.0;
        return arb::isometry::translate(radius * std::cos(angle), radius * std::sin(angle), 0.0);
    };

    std::optional<arb::network_description> network_description() const override {
        return arb::network_description{selection_, weight_, delay_, {}};
    };

    std::vector<arb::event_generator> event_generators(cell_gid_type gid) const override {
        return {};
    }

    std::vector<arb::probe_info> get_probes(cell_gid_type gid) const override {
	return {};
    }

    std::any get_global_properties(arb::cell_kind) const override { return gprop_; }

private:
    cell_size_type num_cells_;
    arb::cable_cell_global_properties gprop_;
    network_selection selection_;
    network_value weight_, delay_;
};

}  // namespace

TEST(network_generation, all) {
    const auto& ctx = g_context;
    const int num_ranks = ctx->distributed->size();

    const auto selection = network_selection::all();
    const auto weight = 2.0;
    const auto delay = 3.0;

    const auto num_cells = 3 * num_ranks;

    auto rec = network_test_recipe(num_cells, selection, weight, delay);

    const auto decomp = partition_load_balance(rec, ctx);

    const auto connections = generate_network_connections(rec, ctx, decomp);

    std::unordered_map<cell_gid_type, std::vector<network_connection_info>> connections_by_dest;

    for(const auto& c : connections) {
        EXPECT_EQ(c.weight, weight);
        EXPECT_EQ(c.delay, delay);
        connections_by_dest[c.dest.gid].emplace_back(c);
    }

    for (const auto& group: decomp.groups()) {
        const auto num_dest = group.kind == cell_kind::spike_source ? 0 : 1;
        for(const auto gid : group.gids) {
	    EXPECT_EQ(connections_by_dest[gid].size(), num_cells * num_dest);
	}
    }
}


TEST(network_generation, cable_only) {
    const auto& ctx = g_context;
    const int num_ranks = ctx->distributed->size();

    const auto selection = intersect(network_selection::source_cell_kind(cell_kind::cable),
        network_selection::destination_cell_kind(cell_kind::cable));
    const auto weight = 2.0;
    const auto delay = 3.0;

    const auto num_cells = 3 * num_ranks;

    auto rec = network_test_recipe(num_cells, selection, weight, delay);

    const auto decomp = partition_load_balance(rec, ctx);

    const auto connections = generate_network_connections(rec, ctx, decomp);

    std::unordered_map<cell_gid_type, std::vector<network_connection_info>> connections_by_dest;

    for(const auto& c : connections) {
        EXPECT_EQ(c.weight, weight);
        EXPECT_EQ(c.delay, delay);
        connections_by_dest[c.dest.gid].emplace_back(c);
    }

    for (const auto& group: decomp.groups()) {
        for(const auto gid : group.gids) {
	    // Only one third is a cable cell
            EXPECT_EQ(connections_by_dest[gid].size(),
                group.kind == cell_kind::cable ? num_cells / 3 : 0);
        }
    }
}
