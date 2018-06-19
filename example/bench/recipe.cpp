#include <random>

#include <benchmark_cell.hpp>
#include <common_types.hpp>
#include <time_sequence.hpp>

#include "recipe.hpp"

cell_size_type bench_recipe::num_cells() const {
    return params_.num_cells;
}

arb::util::unique_any bench_recipe::get_cell_description(cell_gid_type gid) const {
    using RNG = std::mt19937_64;
    arb::benchmark_cell cell;
    cell.realtime_ratio = params_.cell.realtime_ratio;
    cell.time_sequence = arb::poisson_time_seq<RNG>(RNG(gid), 0, 1e-3*params_.cell.spike_freq_hz);
    return std::move(cell);
}

arb::cell_kind bench_recipe::get_cell_kind(arb::cell_gid_type gid) const {
    return arb::cell_kind::benchmark;
}

std::vector<arb::cell_connection> bench_recipe::connections_on(cell_gid_type gid) const {
    const auto n = params_.network.fan_in;
    std::vector<arb::cell_connection> cons;
    cons.reserve(n);
    using RNG = std::mt19937_64;
    RNG rng(gid);
    // issue: invalid gid for source in a connection
    std::uniform_int_distribution<cell_gid_type> dist(0, params_.num_cells-2);
    for (unsigned i=0; i<n; ++i) {
        cell_gid_type src = dist(rng);
        if (src>=gid) ++src;
        // note: target is {gid, 0}, i.e. the first (and only) target on the cell
        arb::cell_connection con({src, 0}, {gid, 0}, 1.f, params_.network.min_delay);
        cons.push_back(con);
    }

    return cons;
}

cell_size_type bench_recipe::num_targets(cell_gid_type gid) const {
    // Only one target, to which all incoming connections connect.
    // It might be the case that we can parameterize this, in which case the connections
    // generated in connections_on should end on random cel-local targets.
    return 1;
}

// one spike source per cell
cell_size_type bench_recipe::num_sources(cell_gid_type gid) const {
    return 1;
}
