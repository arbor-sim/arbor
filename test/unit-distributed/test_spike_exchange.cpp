#include <gtest/gtest.h>
#include "test.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <tuple>
#include <vector>

#include <arbor/event_generator.hpp>
#include <arbor/lif_cell.hpp>
#include <arbor/load_balance.hpp>
#include <arbor/simulation.hpp>
#include <arbor/spike_exchange.hpp>

#include "communication/communicator.hpp"
#include "communication/neighborhood_router.hpp"
#include "lif_cell_group.hpp"
#include "util/span.hpp"

using namespace arb;
namespace U = arb::units;

namespace {
// LIF cells, cells_per_rank on each rank, with connections to random-ish ranks.
// Delays grow with the distance between the ranks of source and target on a
// ring, so that farther levels can be exchanged less often. Weights are below
// threshold, so cells only fire on coincident input, which makes the spike
// trains sensitive to late, missing or duplicated events.
class distance_recipe: public recipe {
public:
    distance_recipe(unsigned num_ranks, unsigned cells_per_rank):
        num_ranks_(num_ranks), cells_per_rank_(cells_per_rank) {}

    cell_size_type num_cells() const override { return num_ranks_*cells_per_rank_; }
    cell_kind get_cell_kind(cell_gid_type) const override { return cell_kind::lif; }
    util::unique_any get_cell_description(cell_gid_type) const override {
        return lif_cell{.source="src", .target="tgt"};
    }

    std::vector<cell_connection> connections_on(cell_gid_type gid) const override {
        auto n = num_cells();
        auto rank = gid/cells_per_rank_;
        std::vector<cell_gid_type> sources = {rank*cells_per_rank_ + (gid + 1)%cells_per_rank_};
        for (unsigned k = 1; k < 4; ++k) sources.push_back((gid*7 + k*(cells_per_rank_ + 3))%n);

        std::vector<cell_connection> conns;
        for (auto src: sources) {
            conns.emplace_back(cell_global_label_type{src, "src"}, cell_local_label_type{"tgt"}, 120.f, delay(src, gid)*U::ms);
        }
        return conns;
    }

    std::vector<event_generator> event_generators(cell_gid_type gid) const override {
        if (gid%cells_per_rank_) return {};
        return {regular_generator({"tgt"}, 250.f, double(gid%3)*U::ms, 7*U::ms)};
    }

    double delay(cell_gid_type src, cell_gid_type tgt) const {
        unsigned a = src/cells_per_rank_, b = tgt/cells_per_rank_;
        auto d = a > b? a - b: b - a;
        // Not a multiple of the epoch length, where rounding slack would hide off-by-one errors.
        return 1.0 + 1.3*std::min(d, num_ranks_ - d);
    }

private:
    unsigned num_ranks_;
    unsigned cells_per_rank_;
};

constexpr unsigned cells_per_rank = 8;

int ring_rank(int rank, int offset, int num_ranks) {
    return ((rank + offset)%num_ranks + num_ranks)%num_ranks;
}

// Neighbourhoods of each rank on a ring: nested, the next {-1, +1}, then the next {-2, ..., +2}.
std::vector<std::vector<int>> nested_rings(int rank, int num_ranks) {
    return {{ring_rank(rank, -1, num_ranks), ring_rank(rank, 1, num_ranks)},
            {ring_rank(rank, -2, num_ranks), ring_rank(rank, -1, num_ranks),
             ring_rank(rank,  1, num_ranks), ring_rank(rank,  2, num_ranks)}};
}

// Asymmetric neighbourhood: we receive from our successor, but it does not receive from us.
std::vector<std::vector<int>> successor(int rank, int num_ranks) {
    return {{ring_rank(rank, 1, num_ranks)}};
}

std::vector<std::vector<int>> everyone(int, int num_ranks) {
    std::vector<int> all(num_ranks);
    std::iota(all.begin(), all.end(), 0);
    return {all};
}

struct run_result {
    std::vector<spike> spikes;
    std::size_t num_spikes;
};

// Simulate until each time in stops in turn, and gather all spikes from all ranks.
run_result run(const std::optional<neighborhood_spike_exchange>& spike_exchange, const std::vector<double>& stops) {
    auto num_ranks = g_context->distributed->size();
    auto rec = distance_recipe(num_ranks, cells_per_rank);
    auto sim = simulation(rec, g_context, partition_load_balance(rec, g_context), 0, spike_exchange);

    std::vector<spike> local;
    sim.set_local_spike_callback([&](const std::vector<spike>& s) { local.insert(local.end(), s.begin(), s.end()); });
    for (auto t: stops) sim.run(t*U::ms, 0.025*U::ms);

    auto spikes = g_context->distributed->gather_spikes(local).values();
    std::sort(spikes.begin(), spikes.end(),
              [](const spike& a, const spike& b) { return std::tie(a.source, a.time) < std::tie(b.source, b.time); });
    return {spikes, sim.num_spikes()};
}

using neighborhood_fn = std::vector<std::vector<int>>(*)(int, int);

std::vector<std::pair<std::string, neighborhood_fn>> neighborhood_shapes() {
    return {{"none",      [](int, int) { return std::vector<std::vector<int>>{}; }},
            {"nested",    nested_rings},
            {"successor", successor},
            {"everyone",  everyone}};
}

std::vector<std::size_t> periods_of(const std::vector<spike_exchange_level>& levels) {
    std::vector<std::size_t> periods;
    for (const auto& l: levels) periods.push_back(l.period);
    return periods;
}

// Check that a rank's levels are neighbourhoods followed by the far level,
// which holds all ranks but ours that are in no neighbourhood.
void expect_valid_levels(const std::vector<spike_exchange_level>& levels, int rank, int num_ranks) {
    ASSERT_FALSE(levels.empty());
    std::vector<int> seen(num_ranks, 0);
    seen[rank] = 1;
    for (std::size_t l = 0; l < levels.size(); ++l) {
        EXPECT_EQ(l + 1 == levels.size(), levels[l].far);
        EXPECT_TRUE(std::is_sorted(levels[l].ranks.begin(), levels[l].ranks.end()));
        for (auto r: levels[l].ranks) ++seen[r];
    }
    EXPECT_EQ(std::vector<int>(num_ranks, 1), seen);
}

// All neighbourhood shapes, and automatic levels.
std::vector<std::pair<std::string, neighborhood_spike_exchange>> exchange_options(int rank, int num_ranks) {
    std::vector<std::pair<std::string, neighborhood_spike_exchange>> options;
    for (const auto& [name, shape]: neighborhood_shapes()) options.push_back({name, {shape(rank, num_ranks)}});
    options.push_back({"automatic", {.automatic = true}});
    return options;
}

} // namespace

TEST(spike_exchange, exchange_period) {
    // Spikes must be exchanged in time for E(k+p+1) to see them: (p+1)*interval <= delay.
    EXPECT_EQ(1u, exchange_period(1.0, 0.5));
    EXPECT_EQ(1u, exchange_period(1.4, 0.5));
    EXPECT_EQ(1u, exchange_period(1.5, 0.5)); // exact multiples lose one epoch to rounding slack
    EXPECT_EQ(2u, exchange_period(1.6, 0.5));
    EXPECT_EQ(4u, exchange_period(2.5001, 0.5));
    EXPECT_EQ(19u, exchange_period(10.01, 0.5));
    for (double delay: {1.0, 1.3, 2.0, 2.5, 3.7, 10.0, 123.4}) {
        for (double interval: {0.025, 0.1, 0.5}) {
            if (delay < 2*interval) continue;
            auto p = exchange_period(delay, interval);
            EXPECT_GE(p, 1u);
            EXPECT_LE((p + 1)*interval, delay);
        }
    }
}

TEST(spike_exchange, same_spikes_as_all_to_all) {
    const int rank = g_context->distributed->id();
    const int num_ranks = g_context->distributed->size();
    const std::vector<double> stops = {50};
    const auto expected = run({}, stops);
    ASSERT_GT(expected.spikes.size(), 0u);
    ASSERT_EQ(expected.spikes.size(), expected.num_spikes);

    for (const auto& [name, options]: exchange_options(rank, num_ranks)) {
        SCOPED_TRACE(name);
        const auto result = run(options, stops);
        EXPECT_EQ(expected.spikes, result.spikes);
        EXPECT_EQ(expected.num_spikes, result.num_spikes);
    }
}

TEST(spike_exchange, same_spikes_over_several_runs) {
    const int rank = g_context->distributed->id();
    const int num_ranks = g_context->distributed->size();
    // The stops are not multiples of the epoch length; the buffered spikes must be flushed at each.
    const std::vector<double> stops = {3.3, 10, 10.1, 31.7, 50};
    const auto expected = run({}, stops);
    ASSERT_GT(expected.spikes.size(), 0u);

    for (const auto& [name, options]: exchange_options(rank, num_ranks)) {
        SCOPED_TRACE(name);
        const auto result = run(options, stops);
        EXPECT_EQ(expected.spikes, result.spikes);
        EXPECT_EQ(expected.num_spikes, result.num_spikes);
    }
}

TEST(spike_exchange, invalid_options) {
    const int num_ranks = g_context->distributed->size();
    auto rec = distance_recipe(num_ranks, cells_per_rank);
    auto decomp = partition_load_balance(rec, g_context);
    EXPECT_THROW(simulation(rec, g_context, decomp, 0, neighborhood_spike_exchange{{{num_ranks}}}), arbor_exception);
    EXPECT_THROW(simulation(rec, g_context, decomp, 0, neighborhood_spike_exchange{{{-1}}}), arbor_exception);
    if (num_ranks > 1) {
        // All ranks must agree on the number of levels.
        auto levels = std::vector<std::vector<int>>(g_context->distributed->id() == 0 ? 1 : 2);
        EXPECT_THROW(simulation(rec, g_context, decomp, 0, neighborhood_spike_exchange{levels}), arbor_exception);
        // All ranks must agree on automatic levels.
        auto automatic = neighborhood_spike_exchange{.automatic = g_context->distributed->id() == 0};
        EXPECT_THROW(simulation(rec, g_context, decomp, 0, automatic), arbor_exception);
    }
    // Automatic levels and explicit neighbourhoods exclude each other.
    EXPECT_THROW(simulation(rec, g_context, decomp, 0, neighborhood_spike_exchange{{{0}}, true}), arbor_exception);
}

// Exchanged in epoch e, spikes become events in epoch e+2. Check that held back
// spikes still arrive in time for all their events, and that each arrives once.
TEST(spike_exchange, communicator_delivers_in_time) {
    const int rank = g_context->distributed->id();
    const int num_ranks = g_context->distributed->size();
    const time_type interval = 0.5;
    const int num_epochs = 40;

    auto rec = distance_recipe(num_ranks, cells_per_rank);
    auto decomp = partition_load_balance(rec, g_context);
    std::vector<cell_gid_type> gids;
    for (const auto& g: decomp->groups()) gids.insert(gids.end(), g.gids.begin(), g.gids.end());
    cell_label_range local_sources, local_targets;
    lif_cell_group group(gids, rec, local_sources, local_targets);
    auto global_sources = g_context->distributed->gather_cell_labels_and_gids({local_sources, gids});

    // Minimum delay of the local connections from each source.
    std::map<cell_gid_type, double> min_delay_from;
    for (auto gid: gids) {
        for (const auto& c: rec.connections_on(gid)) {
            auto [it, _] = min_delay_from.emplace(c.source.gid, std::numeric_limits<double>::infinity());
            it->second = std::min(it->second, double(float(c.delay)));
        }
    }

    for (const auto& [name, options]: exchange_options(rank, num_ranks)) {
        SCOPED_TRACE(name);
        auto comm = communicator(rec, decomp, g_context, options);
        comm.update_connections(rec, decomp, label_resolution_map(global_sources), label_resolution_map({local_targets, gids}));
        comm.schedule_exchange(interval);

        std::map<cell_gid_type, int> num_received;
        for (int e = 0; e < num_epochs; ++e) {
            // Worst case: all cells fire at the start of the epoch.
            std::vector<spike> local_spikes;
            for (auto gid: gids) local_spikes.emplace_back(cell_member_type{gid, 0}, e*interval);
            auto received = e + 1 < num_epochs ? comm.exchange(local_spikes, e) : comm.exchange(local_spikes);
            for (const auto& s: received.from_local.values()) {
                auto it = min_delay_from.find(s.source.gid);
                if (it == min_delay_from.end()) continue;
                ++num_received[s.source.gid];
                EXPECT_GE(s.time + it->second + 1e-9, (e + 2)*interval)
                    << "spike from " << s.source.gid << " at " << s.time << " received in epoch " << e;
            }
        }
        for (const auto& [src, _]: min_delay_from) EXPECT_EQ(num_epochs, num_received[src]) << "source " << src;
    }
}

TEST(spike_exchange, communicator_holds_back_spikes) {
    const int rank = g_context->distributed->id();
    const int num_ranks = g_context->distributed->size();
    const time_type interval = 0.5;

    auto rec = distance_recipe(num_ranks, cells_per_rank);
    auto decomp = partition_load_balance(rec, g_context);
    std::vector<cell_gid_type> gids;
    for (const auto& g: decomp->groups()) gids.insert(gids.end(), g.gids.begin(), g.gids.end());
    cell_label_range local_sources, local_targets;
    lif_cell_group group(gids, rec, local_sources, local_targets);
    auto global_sources = g_context->distributed->gather_cell_labels_and_gids({local_sources, gids});

    // For each source cell, whether it is connected to a cell on this rank.
    std::set<cell_gid_type> connected_sources;
    for (auto gid: gids) {
        for (const auto& c: rec.connections_on(gid)) connected_sources.insert(c.source.gid);
    }

    for (const auto& [name, shape]: neighborhood_shapes()) {
        SCOPED_TRACE(name);
        auto options = neighborhood_spike_exchange{shape(rank, num_ranks)};
        auto comm = communicator(rec, decomp, g_context, options);
        comm.update_connections(rec, decomp, label_resolution_map(global_sources), label_resolution_map({local_targets, gids}));
        comm.schedule_exchange(interval);

        // Expected periods: from the global minimum delay of the connections per level.
        auto num_levels = options.neighborhoods.size() + 1;
        auto level_of = [&](int receiver, int source) {
            auto levels = shape(receiver, num_ranks);
            for (std::size_t l = 0; l < levels.size(); ++l) {
                if (std::find(levels[l].begin(), levels[l].end(), source) != levels[l].end()) return l;
            }
            return levels.size();
        };
        std::vector<double> min_delays(num_levels, std::numeric_limits<double>::infinity());
        for (cell_gid_type tgt = 0; tgt < rec.num_cells(); ++tgt) {
            int tgt_rank = decomp->gid_domain(tgt);
            for (const auto& c: rec.connections_on(tgt)) {
                int src_rank = decomp->gid_domain(c.source.gid);
                if (src_rank == tgt_rank) continue;
                auto& d = min_delays[level_of(tgt_rank, src_rank)];
                d = std::min(d, double(float(c.delay)));
            }
        }
        std::vector<std::size_t> expected_periods;
        for (auto d: min_delays) expected_periods.push_back(std::isfinite(d)? exchange_period(d, interval): 0);
        EXPECT_EQ(expected_periods, periods_of(comm.exchange_levels()));
        expect_valid_levels(comm.exchange_levels(), rank, num_ranks);

        // Every cell fires once. No connection between ranks is short enough to
        // be exchanged every epoch, so in epoch 1 only our own spikes arrive.
        std::vector<spike> local_spikes;
        for (auto gid: gids) local_spikes.emplace_back(cell_member_type{gid, 0}, 0.5);
        auto held = comm.exchange(local_spikes, 1);
        for (auto d: util::make_span(num_ranks)) {
            EXPECT_EQ(d == rank? gids.size(): 0u, held.from_local.count(d));
        }

        // The final exchange delivers all spikes held back, without duplicates.
        std::vector<spike> no_spikes;
        auto flushed = comm.exchange(no_spikes);
        std::set<cell_gid_type> received;
        const auto& part = flushed.from_local.partition();
        for (auto d: util::make_span(num_ranks)) {
            if (d == rank) {
                EXPECT_EQ(0u, flushed.from_local.count(d));
                continue;
            }
            for (auto i: util::make_span(part[d], part[d+1])) {
                auto gid = flushed.from_local.values()[i].source.gid;
                EXPECT_EQ(d, decomp->gid_domain(gid));
                EXPECT_TRUE(received.insert(gid).second) << "duplicate spike from " << gid;
            }
        }
        // The all-gather may also deliver spikes without local targets.
        for (auto gid: connected_sources) {
            if (decomp->gid_domain(gid) == rank) continue;
            EXPECT_TRUE(received.count(gid)) << "missing spike from " << gid;
        }
        EXPECT_EQ(std::size_t(num_ranks*gids.size()), comm.num_spikes());
    }
}

// Automatic levels group source ranks by the class of the exchange period
// their minimum connection delay allows; the slowest class is the far level.
TEST(spike_exchange, automatic_levels) {
    const int rank = g_context->distributed->id();
    const int num_ranks = g_context->distributed->size();
    const time_type interval = 0.5;

    auto rec = distance_recipe(num_ranks, cells_per_rank);
    auto decomp = partition_load_balance(rec, g_context);
    std::vector<cell_gid_type> gids;
    for (const auto& g: decomp->groups()) gids.insert(gids.end(), g.gids.begin(), g.gids.end());
    cell_label_range local_sources, local_targets;
    lif_cell_group group(gids, rec, local_sources, local_targets);
    auto global_sources = g_context->distributed->gather_cell_labels_and_gids({local_sources, gids});

    auto comm = communicator(rec, decomp, g_context, neighborhood_spike_exchange{.automatic = true});
    comm.update_connections(rec, decomp, label_resolution_map(global_sources), label_resolution_map({local_targets, gids}));
    comm.schedule_exchange(interval);

    // From the recipe: minimum delay between each pair of ranks, and its period class.
    const auto inf = std::numeric_limits<double>::infinity();
    std::vector<std::vector<double>> min_delay(num_ranks, std::vector<double>(num_ranks, inf));
    for (cell_gid_type tgt = 0; tgt < rec.num_cells(); ++tgt) {
        for (const auto& c: rec.connections_on(tgt)) {
            auto& d = min_delay[decomp->gid_domain(tgt)][decomp->gid_domain(c.source.gid)];
            d = std::min(d, double(float(c.delay)));
        }
    }
    auto class_of = [&](int receiver, int source) {
        auto d = min_delay[receiver][source];
        return receiver == source || !std::isfinite(d)? -1: int(period_class(exchange_period(d, interval)));
    };
    int far = -1;
    for (int r = 0; r < num_ranks; ++r) {
        for (int s = 0; s < num_ranks; ++s) far = std::max(far, class_of(r, s));
    }
    far = std::max(far, 0);

    // Our neighbourhoods, in any order.
    std::vector<std::vector<int>> expected(far);
    for (int s = 0; s < num_ranks; ++s) {
        auto c = class_of(rank, s);
        if (c >= 0 && c < far) expected[c].push_back(s);
    }
    auto levels = comm.exchange_levels();
    expect_valid_levels(levels, rank, num_ranks);
    std::vector<std::vector<int>> neighborhoods;
    for (const auto& l: levels) {
        if (!l.far) neighborhoods.push_back(l.ranks);
    }
    EXPECT_EQ(expected, neighborhoods);

    // Periods: from the global minimum delay per class.
    std::vector<double> class_delay(far + 1, inf);
    for (int r = 0; r < num_ranks; ++r) {
        for (int s = 0; s < num_ranks; ++s) {
            auto c = class_of(r, s);
            if (c >= 0) class_delay[c] = std::min(class_delay[c], min_delay[r][s]);
        }
    }
    std::vector<std::size_t> expected_periods;
    for (auto d: class_delay) expected_periods.push_back(std::isfinite(d)? exchange_period(d, interval): 0);
    EXPECT_EQ(expected_periods, periods_of(levels));
    // Each level is exchanged at most twice as often as its own delays require.
    for (std::size_t c = 0; c < expected_periods.size(); ++c) {
        if (expected_periods[c]) {
            EXPECT_EQ(c, period_class(expected_periods[c]));
        }
    }
}

TEST(spike_exchange, period_class) {
    EXPECT_EQ(0u, period_class(1));
    EXPECT_EQ(1u, period_class(2));
    EXPECT_EQ(1u, period_class(3));
    EXPECT_EQ(2u, period_class(4));
    EXPECT_EQ(2u, period_class(7));
    EXPECT_EQ(3u, period_class(8));
    EXPECT_EQ(5u, period_class(38));
}

// The simulation reports the levels of its communicator; none for the default exchange.
TEST(spike_exchange, simulation_levels) {
    const int rank = g_context->distributed->id();
    const int num_ranks = g_context->distributed->size();
    auto rec = distance_recipe(num_ranks, cells_per_rank);
    auto decomp = partition_load_balance(rec, g_context);

    EXPECT_TRUE(simulation(rec, g_context, decomp).spike_exchange_levels().empty());

    for (const auto& [name, options]: exchange_options(rank, num_ranks)) {
        SCOPED_TRACE(name);
        auto sim = simulation(rec, g_context, decomp, 0, options);
        auto levels = sim.spike_exchange_levels();
        expect_valid_levels(levels, rank, num_ranks);
        if (!options.automatic) {
            EXPECT_EQ(options.neighborhoods.size() + 1, levels.size());
        }
    }
}
