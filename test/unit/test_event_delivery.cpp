// Check that events are routed correctly to underlying cell implementations.
//
// Model:
// * N identical cells, one synapse and gap junction site each.
// * Set up dynamics so that one incoming spike generates one outgoing spike.
// * Inject events one per cell in a given order, and confirm generated spikes
//   are in the same order.

#include <arbor/common_types.hpp>
#include <arbor/domain_decomposition.hpp>
#include <arbor/simulation.hpp>
#include <arbor/cable_cell.hpp>
#include <arbor/spike.hpp>
#include <arbor/spike_event.hpp>

#include "util/rangeutil.hpp"
#include "util/transform.hpp"

#include "../simple_recipes.hpp"
#include "../gtest.h"

using namespace arb;

using n_cable_cell_recipe = homogeneous_recipe<cell_kind::cable, cable_cell>;

struct test_recipe: public n_cable_cell_recipe {
    explicit test_recipe(int n): n_cable_cell_recipe(n, test_cell()) {}

    static cable_cell test_cell() {
        sample_tree st;
        st.append({0,0,0,10,1});

        label_dict d;
        d.set("soma", arb::reg::tagged(1));

        cable_cell c(st, d);
        c.place(mlocation{0, 0.5}, "expsyn");
        c.place(mlocation{0, 0.5}, threshold_detector{-64});
        c.place(mlocation{0, 0.5}, gap_junction_site{});

        return c;
    }

    cell_size_type num_sources(cell_gid_type) const override { return 1; }
    cell_size_type num_targets(cell_gid_type) const override { return 1; }
};

using gid_vector = std::vector<cell_gid_type>;
using group_gids_type = std::vector<gid_vector>;

std::vector<cell_gid_type> run_test_sim(const recipe& R, const group_gids_type& group_gids) {
    arb::context ctx = make_context(proc_allocation{});
    unsigned n = R.num_cells();

    domain_decomposition D;
    D.gid_domain = [](cell_gid_type) { return 0; };
    D.num_domains = 1;
    D.num_local_cells = n;
    D.num_global_cells = n;

    for (const auto& gidvec: group_gids) {
        group_description group{cell_kind::cable, gidvec, backend_kind::multicore};
        D.groups.push_back(group);
    }

    std::vector<spike> spikes;

    simulation sim(R, D, ctx);
    sim.set_global_spike_callback(
            [&spikes](const std::vector<spike>& ss) {
                spikes.insert(spikes.end(), ss.begin(), ss.end());
            });

    constexpr time_type ev_delta_t = 0.2;

    pse_vector cell_events;
    for (unsigned i = 0; i<n; ++i) {
        cell_events.push_back({{i, 0u}, i*ev_delta_t, 1.f});
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

    cell_size_type num_gap_junction_sites(cell_gid_type) const override { return 1; }

    std::vector<gap_junction_connection> gap_junctions_on(cell_gid_type i) const override {
        std::vector<gap_junction_connection> gjs;
        for (auto p: gj_pairs_) {
            if (p.first == i || p.second == i) gjs.push_back({{p.first, 0u}, {p.second, 0u}, 0.});
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
