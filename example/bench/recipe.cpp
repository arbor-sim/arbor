#include <random>

#include <arbor/benchmark_cell.hpp>
#include <arbor/common_types.hpp>
#include <arbor/time_sequence.hpp>

#include "recipe.hpp"

using arb::cell_gid_type;
using arb::cell_size_type;
using arb::cell_kind;

cell_size_type bench_recipe::num_cells() const {
    return params_.num_cells;
}

arb::util::unique_any bench_recipe::get_cell_description(cell_gid_type gid) const {
    using rng_type = std::mt19937_64;
    arb::benchmark_cell cell;
    cell.realtime_ratio = params_.cell.realtime_ratio;

    // The time_sequence of the cell produces the series of time points at
    // which it will spike. We use a poisson_time_seq with a random sequence
    // seeded with the gid. In this way, a cell's random stream depends only
    // on its gid, and will hence give reproducable results when run with
    // different MPI ranks and threads.
    cell.time_sequence =
        arb::poisson_time_seq<rng_type>(
                rng_type(gid), 0, 1e-3*params_.cell.spike_freq_hz);

    return std::move(cell);
}

cell_kind bench_recipe::get_cell_kind(cell_gid_type gid) const {
    return cell_kind::benchmark;
}

std::vector<arb::cell_connection> bench_recipe::connections_on(cell_gid_type gid) const {
    const auto n = params_.network.fan_in;
    std::vector<arb::cell_connection> cons;
    cons.reserve(n);
    using rng_type = std::mt19937_64;
    rng_type rng(gid);

    // Generate n incoming connections on this cell with random sources, where
    // the source can't equal gid (i.e. no self-connections).
    // We want a random distribution that will uniformly draw values from the
    // union of the two ranges: [0, gid-1] AND [gid+1, num_cells-1].
    // To do this, we draw random values in the range [0, num_cells-2], then
    // add 1 to values ≥ gid.

    std::uniform_int_distribution<cell_gid_type> dist(0, params_.num_cells-2);
    for (unsigned i=0; i<n; ++i) {
        // Draw random source and adjust to avoid self-connections if neccesary.
        cell_gid_type src = dist(rng);
        if (src>=gid) ++src;
        // Note: target is {gid, 0}, i.e. the first (and only) target on the cell.
        arb::cell_connection con({src, 0}, {gid, 0}, 1.f, params_.network.min_delay);
        cons.push_back(con);
    }

    return cons;
}

cell_size_type bench_recipe::num_targets(cell_gid_type gid) const {
    // Only one target, to which all incoming connections connect.
    // This could be parameterized, in which case the connections
    // generated in connections_on should end on random cell-local targets.
    return 1;
}

// one spike source per cell
cell_size_type bench_recipe::num_sources(cell_gid_type gid) const {
    return 1;
}
