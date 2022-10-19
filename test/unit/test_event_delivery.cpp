// Check that events are routed correctly to underlying cell implementations.
//
// Model:
// * N identical cells, one synapse and gap junction site each.
// * Set up dynamics so that one incoming spike generates one outgoing spike.
// * Inject events one per cell in a given order, and confirm generated spikes
//   are in the same order.

#include <arbor/cable_cell.hpp>
#include <arbor/common_types.hpp>
#include <arbor/domain_decomposition.hpp>
#include <arbor/load_balance.hpp>
#include <arbor/morph/segment_tree.hpp>
#include <arbor/simulation.hpp>
#include <arbor/spike.hpp>
#include <arbor/spike_event.hpp>

#include "util/rangeutil.hpp"
#include "util/transform.hpp"

#include "../simple_recipes.hpp"
#include <gtest/gtest.h>

using namespace arb;

using n_cable_cell_recipe = homogeneous_recipe<cell_kind::cable, cable_cell>;

struct test_recipe: public n_cable_cell_recipe {
    explicit test_recipe(int n): n_cable_cell_recipe(n, test_cell()) {}

    static cable_cell test_cell() {
        segment_tree st;
        st.append(mnpos, {0,0, 0,10}, {0,0,20,10}, 1);

        label_dict labels;
        labels.set("soma", arb::reg::tagged(1));

        decor decorations;
        decorations.place(mlocation{0, 0.5}, synapse("expsyn"), "synapse");
        decorations.place(mlocation{0, 0.5}, threshold_detector{-64}, "detector");
        decorations.place(mlocation{0, 0.5}, junction("gj"), "gapjunction");
        cable_cell c(st, decorations, labels);

        return c;
    }
};

using gid_vector = std::vector<cell_gid_type>;
using group_gids_type = std::vector<gid_vector>;

std::vector<cell_gid_type> run_test_sim(const recipe& R, const group_gids_type& group_gids) {

    unsigned n = R.num_cells();
    std::vector<group_description> groups;
    for (const auto& gidvec: group_gids) {
        groups.emplace_back(cell_kind::cable, gidvec, backend_kind::multicore);
    }

    auto C = make_context();
    auto D = domain_decomposition(R, C, groups);
    simulation sim(R, C, D);

    std::vector<spike> spikes;
    sim.set_global_spike_callback(
            [&spikes](const std::vector<spike>& ss) {
                spikes.insert(spikes.end(), ss.begin(), ss.end());
            });

    constexpr time_type ev_delta_t = 0.2;

    cse_vector cell_events;
    for (unsigned i = 0; i<n; ++i) {
        cell_events.push_back({i, {{0u, i*ev_delta_t, 1.f}}});
    }

    sim.inject_events(cell_events);
    sim.run((n+1)*ev_delta_t, 0.01);

    std::vector<cell_gid_type> spike_gids;
    util::sort_by(spikes, [](auto s) { return s.time; });
    util::assign(spike_gids, util::transform_view(spikes, [](auto s) { return s.source.gid; }));

    return spike_gids;
}


TEST(mc_event_delivery, one_cell_per_group) {
    gid_vector spike_gids = run_test_sim(test_recipe(10), {{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}});
    EXPECT_EQ((gid_vector{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}), spike_gids);
}

TEST(mc_event_delivery, two_interleaved_groups) {
    gid_vector spike_gids = run_test_sim(test_recipe(10), {{0, 2, 4, 6, 8}, {1, 3, 5, 7, 9}});
    EXPECT_EQ((gid_vector{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}), spike_gids);
}

typedef std::vector<std::pair<unsigned, unsigned>> cell_gj_pairs;

struct test_recipe_gj: public test_recipe {
    explicit test_recipe_gj(int n, cell_gj_pairs gj_pairs):
        test_recipe(n), gj_pairs_(std::move(gj_pairs)) {}

    std::vector<gap_junction_connection> gap_junctions_on(cell_gid_type i) const override {
        std::vector<gap_junction_connection> gjs;
        for (auto p: gj_pairs_) {
            if (p.first == i) gjs.push_back({{p.second, "gapjunction", lid_selection_policy::assert_univalent},
                                             {"gapjunction", lid_selection_policy::assert_univalent}, 0.});
            if (p.second == i) gjs.push_back({{p.first, "gapjunction", lid_selection_policy::assert_univalent},
                                             {"gapjunction", lid_selection_policy::assert_univalent}, 0.});
        }
        return gjs;
    }

    cell_gj_pairs gj_pairs_;
};

TEST(mc_event_delivery, gj_reordered) {
    test_recipe_gj R(5, {{1u, 3u}, {2u, 4u}});
    gid_vector spike_gids = run_test_sim(R, {{0, 1, 2, 3, 4}});
    EXPECT_EQ((gid_vector{0, 1, 2, 3, 4}), spike_gids);
}
