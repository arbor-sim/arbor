#include <gtest/gtest.h>

#include <any>
#include <vector>

#include <arbor/common_types.hpp>
#include <arbor/domain_decomposition.hpp>
#include <arbor/load_balance.hpp>
#include <arbor/cable_cell.hpp>
#include <arbor/recipe.hpp>
#include <arbor/simulation.hpp>

#include <arborenv/default_env.hpp>

#include "util/span.hpp"

#include "../common_cells.hpp"

using namespace arb;
using arb::util::make_span;

namespace {
    class custom_recipe: public recipe {
    public:
        custom_recipe(std::vector<cable_cell> cells,
                      std::vector<std::vector<cell_connection>> conns,
                      std::vector<std::vector<gap_junction_connection>> gjs,
                      std::vector<std::vector<arb::event_generator>> gens):
            num_cells_(cells.size()),
            connections_(conns),
            gap_junctions_(gjs),
            event_generators_(gens),
            cells_(cells) {}

        cell_size_type num_cells() const override {
            return num_cells_;
        }
        arb::util::unique_any get_cell_description(cell_gid_type gid) const override {
            return cells_.at(gid);
        }
        cell_kind get_cell_kind(cell_gid_type gid) const override {
            return cell_kind::cable;
        }
        std::vector<gap_junction_connection> gap_junctions_on(cell_gid_type gid) const override {
            return gap_junctions_.at(gid);
        }
        std::vector<cell_connection> connections_on(cell_gid_type gid) const override {
            return connections_.at(gid);
        }
        std::vector<arb::event_generator> event_generators(cell_gid_type gid) const override {
            return event_generators_.at(gid);
        }
        std::any get_global_properties(cell_kind) const override {
            arb::cable_cell_global_properties a;
            a.default_parameters = arb::neuron_parameter_defaults;
            return a;
        }

    private:
        cell_size_type num_cells_;
        std::vector<std::vector<cell_connection>> connections_;
        std::vector<std::vector<gap_junction_connection>> gap_junctions_;
        std::vector<std::vector<arb::event_generator>> event_generators_;
        std::vector<cable_cell> cells_;
    };

    cable_cell custom_cell(cell_size_type num_detectors, cell_size_type num_synapses, cell_size_type num_gj) {
        arb::segment_tree tree;
        tree.append(arb::mnpos, {0,0,0,10}, {0,0,20,10}, 1); // soma
        tree.append(0, {0,0, 20, 2}, {0,0, 320, 2}, 3);  // dendrite

        arb::cable_cell cell(tree, {});

        arb::decor decorations;

        // Add a num_detectors detectors to the cell.
        for (auto i: util::make_span(num_detectors)) {
            decorations.place(arb::mlocation{0,(double)i/num_detectors}, arb::threshold_detector{10}, "detector"+std::to_string(i));
        }

        // Add a num_synapses synapses to the cell.
        for (auto i: util::make_span(num_synapses)) {
            decorations.place(arb::mlocation{0,(double)i/num_synapses}, arb::synapse("expsyn"), "synapse"+std::to_string(i));
        }

        // Add a num_gj gap_junctions to the cell.
        for (auto i: util::make_span(num_gj)) {
            decorations.place(arb::mlocation{0,(double)i/num_gj}, arb::junction("gj"), "gapjunction"+std::to_string(i));
        }

        return arb::cable_cell(tree, decorations);
    }
}

TEST(recipe, gap_junctions)
{
    auto context = make_context({arbenv::default_concurrency(), -1});

    auto cell_0 = custom_cell(0, 0, 3);
    auto cell_1 = custom_cell(0, 0, 3);

    using policy = lid_selection_policy;
    {
        std::vector<arb::gap_junction_connection> gjs_0 = {{{1, "gapjunction1", policy::assert_univalent}, {"gapjunction0", policy::assert_univalent}, 0.1},
                                                           {{1, "gapjunction2", policy::assert_univalent}, {"gapjunction1", policy::assert_univalent}, 0.1},
                                                           {{1, "gapjunction0", policy::assert_univalent}, {"gapjunction2", policy::assert_univalent}, 0.1}};

        std::vector<arb::gap_junction_connection> gjs_1 = {{{0, "gapjunction0", policy::assert_univalent}, {"gapjunction1", policy::assert_univalent}, 0.1},
                                                           {{0, "gapjunction1", policy::assert_univalent}, {"gapjunction2", policy::assert_univalent}, 0.1},
                                                           {{0, "gapjunction2", policy::assert_univalent}, {"gapjunction0", policy::assert_univalent}, 0.1}};

        auto recipe_0 = custom_recipe({cell_0, cell_1}, {{}, {}}, {gjs_0, gjs_1}, {{}, {}});
        auto decomp_0 = partition_load_balance(recipe_0, context);

        EXPECT_NO_THROW(simulation(recipe_0, context, decomp_0));
    }
    {
        std::vector<arb::gap_junction_connection> gjs_0 = {{{1, "gapjunction1", policy::assert_univalent}, {"gapjunction0", policy::assert_univalent}, 0.1},
                                                           {{1, "gapjunction2", policy::assert_univalent}, {"gapjunction1", policy::assert_univalent}, 0.1},
                                                           {{1, "gapjunction5", policy::assert_univalent}, {"gapjunction2", policy::assert_univalent}, 0.1}};

        std::vector<arb::gap_junction_connection> gjs_1 = {{{0, "gapjunction0", policy::assert_univalent}, {"gapjunction1", policy::assert_univalent}, 0.1},
                                                           {{0, "gapjunction1", policy::assert_univalent}, {"gapjunction2", policy::assert_univalent}, 0.1},
                                                           {{0, "gapjunction2", policy::assert_univalent}, {"gapjunction5", policy::assert_univalent}, 0.1}};

        auto recipe_1 = custom_recipe({cell_0, cell_1}, {{}, {}}, {gjs_0, gjs_1}, {{}, {}});
        auto decomp_1 = partition_load_balance(recipe_1, context);

        EXPECT_THROW(simulation(recipe_1, context, decomp_1), arb::bad_connection_label);

    }
}

TEST(recipe, connections)
{
    auto context = make_context({arbenv::default_concurrency(), -1});

    auto cell_0 = custom_cell(1, 2, 0);
    auto cell_1 = custom_cell(2, 1, 0);
    std::vector<arb::cell_connection> conns_0, conns_1;
    {
        conns_0 = {{{1, "detector0"}, {"synapse0"}, 0.1, 0.1},
                   {{1, "detector1"}, {"synapse0"}, 0.1, 0.1},
                   {{1, "detector0"}, {"synapse1"}, 0.2, 0.4}};

        conns_1 = {{{0, "detector0"}, {"synapse0"}, 0.1, 0.2},
                   {{0, "detector0"}, {"synapse0"}, 0.3, 0.1},
                   {{0, "detector0"}, {"synapse0"}, 0.1, 0.8}};

        auto recipe_0 = custom_recipe({cell_0, cell_1}, {conns_0, conns_1}, {{}, {}},  {{}, {}});
        auto decomp_0 = partition_load_balance(recipe_0, context);

        EXPECT_NO_THROW(simulation(recipe_0, context, decomp_0));
    }
    {
        conns_0 = {{{1, "detector0"}, {"synapse0"}, 0.1, 0.1},
                   {{2, "detector1"}, {"synapse0"}, 0.1, 0.1},
                   {{1, "detector0"}, {"synapse1"}, 0.2, 0.4}};

        conns_1 = {{{0, "detector0"}, {"synapse0"}, 0.1, 0.2},
                   {{0, "detector0"}, {"synapse0"}, 0.3, 0.1},
                   {{0, "detector0"}, {"synapse0"}, 0.1, 0.8}};

        auto recipe_1 = custom_recipe({cell_0, cell_1}, {conns_0, conns_1}, {{}, {}},  {{}, {}});
        auto decomp_1 = partition_load_balance(recipe_1, context);

        EXPECT_THROW(simulation(recipe_1, context, decomp_1), arb::bad_connection_source_gid);
    }
    {
        conns_0 = {{{1, "detector0"}, {"synapse0"}, 0.1, 0.1},
                   {{1, "detector1"}, {"synapse0"}, 0.1, 0.1},
                   {{1, "detector3"}, {"synapse1"}, 0.2, 0.4}};

        conns_1 = {{{0, "detector0"}, {"synapse0"}, 0.1, 0.2},
                   {{0, "detector0"}, {"synapse0"}, 0.3, 0.1},
                   {{0, "detector0"}, {"synapse0"}, 0.1, 0.8}};

        auto recipe_2 = custom_recipe({cell_0, cell_1}, {conns_0, conns_1}, {{}, {}},  {{}, {}});
        auto decomp_2 = partition_load_balance(recipe_2, context);

        EXPECT_THROW(simulation(recipe_2, context, decomp_2), arb::bad_connection_label);
    }
    {
        conns_0 = {{{1, "detector0"}, {"synapse0"}, 0.1, 0.1},
                   {{1, "detector1"}, {"synapse0"}, 0.1, 0.1},
                   {{1, "detector0"}, {"synapse1"}, 0.2, 0.4}};

        conns_1 = {{{0, "detector0"}, {"synapse0"}, 0.1, 0.2},
                   {{0, "detector0"}, {"synapse9"}, 0.3, 0.1},
                   {{0, "detector0"}, {"synapse0"}, 0.1, 0.8}};

        auto recipe_4 = custom_recipe({cell_0, cell_1}, {conns_0, conns_1}, {{}, {}},  {{}, {}});
        auto decomp_4 = partition_load_balance(recipe_4, context);

        EXPECT_THROW(simulation(recipe_4, context, decomp_4), arb::bad_connection_label);
    }
}

TEST(recipe, event_generators) {
    auto context = make_context({arbenv::default_concurrency(), -1});

    auto cell_0 = custom_cell(1, 2, 0);
    auto cell_1 = custom_cell(2, 1, 0);
    {
        std::vector<arb::event_generator>
            gens_0 = {arb::explicit_generator({"synapse0"}, 0.1, std::vector<arb::time_type>{1.0}),
                      arb::explicit_generator({"synapse1"}, 0.1, std::vector<arb::time_type>{2.0})},
            gens_1 = {arb::explicit_generator({"synapse0"}, 0.1, std::vector<arb::time_type>{1.0})};

        auto recipe_0 = custom_recipe({cell_0, cell_1}, {{}, {}}, {{}, {}},  {gens_0, gens_1});
        auto decomp_0 = partition_load_balance(recipe_0, context);

        EXPECT_NO_THROW(simulation(recipe_0, context, decomp_0).run(1, 0.1));
    }
    {
        std::vector<arb::event_generator>
            gens_0 = {arb::regular_generator({"totally-not-a-synapse-42"}, 0.1, 0, 0.001)},
            gens_1 = {};

        auto recipe_0 = custom_recipe({cell_0, cell_1}, {{}, {}}, {{}, {}},  {gens_0, gens_1});
        auto decomp_0 = partition_load_balance(recipe_0, context);

        EXPECT_THROW(simulation(recipe_0, context, decomp_0), arb::bad_connection_label);
    }
}
