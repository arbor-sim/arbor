#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include <arbor/context.hpp>
#include <arbor/event_generator.hpp>
#include <arbor/lif_cell.hpp>
#include <arbor/load_balance.hpp>
#include <arbor/simulation.hpp>
#include <arbor/spike_exchange.hpp>
#include <arbor/symmetric_recipe.hpp>

using namespace arb;
namespace U = arb::units;

// A dry run simulates rank 0 of a model in which every rank holds a copy of
// the same tile, shifted by its rank. The spikes of rank 0 must then be exactly
// those of the first tile of the full model simulated on a single rank.

namespace {
constexpr unsigned tiles = 7;
constexpr unsigned cells_per_tile = 8;

// LIF cells with connections from the own tile and from tiles at different
// distances on a ring. Delays grow with that distance and are not multiples
// of the epoch length. Weights are below threshold, so spike trains are
// sensitive to late, missing or duplicated events.
class lif_tile: public tile {
public:
    cell_size_type num_cells() const override { return cells_per_tile; }
    cell_size_type num_tiles() const override { return tiles; }
    cell_kind get_cell_kind(cell_gid_type) const override { return cell_kind::lif; }
    util::unique_any get_cell_description(cell_gid_type) const override {
        return lif_cell{.source="src", .target="tgt"};
    }

    // Sources are relative to tile 0; symmetric_recipe shifts them for the other tiles.
    std::vector<cell_connection> connections_on(cell_gid_type gid) const override {
        std::vector<cell_connection> conns;
        for (unsigned k = 0; k < 4; ++k) {
            unsigned src_tile = (gid*3 + k*5)%tiles;
            if (k == 0) src_tile = 0;
            cell_gid_type src = src_tile*cells_per_tile + (gid + k + 1)%cells_per_tile;
            auto distance = std::min(src_tile, tiles - src_tile);
            conns.emplace_back(cell_global_label_type{src, "src"}, cell_local_label_type{"tgt"},
                               120.f, (1.0 + 1.3*distance)*U::ms);
        }
        return conns;
    }

    // Called with the global gid, unlike all other callbacks.
    std::vector<event_generator> event_generators(cell_gid_type gid) const override {
        auto lid = gid%cells_per_tile;
        if (lid%4) return {};
        return {regular_generator({"tgt"}, 250.f, double(lid%3)*U::ms, 7*U::ms)};
    }
};

int ring(int offset) { return (offset + int(tiles))%tiles; }

std::vector<spike> run(context ctx, const std::optional<neighborhood_spike_exchange>& exchange, std::size_t* num_spikes = nullptr) {
    auto rec = symmetric_recipe(std::make_unique<lif_tile>());
    auto sim = simulation(rec, ctx, partition_load_balance(rec, ctx), 0, exchange);
    std::vector<spike> spikes;
    sim.set_local_spike_callback([&](const std::vector<spike>& s) { spikes.insert(spikes.end(), s.begin(), s.end()); });
    sim.run(30*U::ms, 0.025*U::ms);
    sim.run(50*U::ms, 0.025*U::ms);
    if (num_spikes) *num_spikes = sim.num_spikes();
    std::sort(spikes.begin(), spikes.end(),
              [](const spike& a, const spike& b) { return std::tie(a.source, a.time) < std::tie(b.source, b.time); });
    return spikes;
}

// Spikes of the first tile when simulating all tiles on a single rank.
std::vector<spike> reference_spikes(std::size_t* num_spikes) {
    auto spikes = run(make_context(), {}, num_spikes);
    spikes.erase(std::remove_if(spikes.begin(), spikes.end(),
                                [](const spike& s) { return s.source.gid >= cells_per_tile; }),
                 spikes.end());
    return spikes;
}

context dry_run_context() {
    return make_context(proc_allocation{}, dry_run_info(tiles, cells_per_tile));
}
} // namespace

TEST(dry_run_spike_exchange, all_to_all) {
    std::size_t expected_num_spikes = 0, num_spikes = 0;
    auto expected = reference_spikes(&expected_num_spikes);
    ASSERT_GT(expected.size(), 0u);

    EXPECT_EQ(expected, run(dry_run_context(), {}, &num_spikes));
    EXPECT_EQ(expected_num_spikes, num_spikes);
}

TEST(dry_run_spike_exchange, neighborhoods) {
    std::size_t expected_num_spikes = 0;
    auto expected = reference_spikes(&expected_num_spikes);
    ASSERT_GT(expected.size(), 0u);

    // Levels of rank 0; the other ranks' levels are the same, shifted by their rank.
    const std::vector<std::pair<std::string, std::vector<std::vector<int>>>> shapes = {
        {"none",      {}},
        {"nested",    {{ring(-1), ring(1)}, {ring(-2), ring(-1), ring(1), ring(2)}}},
        {"successor", {{ring(1)}}},
        {"everyone",  {{0, 1, 2, 3, 4, 5, 6}}},
    };
    for (const auto& [name, levels]: shapes) {
        SCOPED_TRACE(name);
        std::size_t num_spikes = 0;
        EXPECT_EQ(expected, run(dry_run_context(), neighborhood_spike_exchange{levels}, &num_spikes));
        EXPECT_EQ(expected_num_spikes, num_spikes);
    }
    {
        SCOPED_TRACE("automatic");
        std::size_t num_spikes = 0;
        EXPECT_EQ(expected, run(dry_run_context(), neighborhood_spike_exchange{.automatic = true}, &num_spikes));
        EXPECT_EQ(expected_num_spikes, num_spikes);
    }
}
