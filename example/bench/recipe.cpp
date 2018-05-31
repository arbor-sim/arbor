#include <exception>
#include <random>
#include <fstream>

#include <benchmark_cell_group.hpp>
#include <common_types.hpp>
#include <time_sequence.hpp>

#include <json/json.hpp>

#include "recipe.hpp"

cell_size_type bench_recipe::num_cells() const {
    return params_.num_cells;
}

arb::util::unique_any bench_recipe::get_cell_description(cell_gid_type gid) const {
    //std::cout << "generating cell description for: " << gid << "\n";
    using RNG = std::mt19937_64;
    arb::benchmark_cell cell;
    cell.run_time_per_ms = params_.cell.us_per_ms;
    cell.time_sequence = arb::poisson_time_seq<RNG>(RNG(gid), 0, 1e-3*params_.cell.spike_freq_hz);
    return std::move(cell);
}

arb::cell_kind bench_recipe::get_cell_kind(arb::cell_gid_type gid) const {
    return arb::cell_kind::benchmark;
}

std::vector<arb::cell_connection> bench_recipe::connections_on(cell_gid_type gid) const {
    const auto n = params_.network.fan_in;
    //std::cout << "generating " << n << " connections for: " << gid << "\n";
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
        //std::cout << "  " << i << " " << con.source << " -> " << con.dest << "\n";
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

//
// benchmark parameters
//


double bench_params::expected_advance_rate() const {
    return cell.us_per_ms*1e-3;
}
double bench_params::expected_advance_time() const {
    return expected_advance_rate() * tfinal*1e-3 * num_cells;
}
unsigned bench_params::expected_spikes() const {
    return num_cells * tfinal*1e-3 * cell.spike_freq_hz;
}
unsigned bench_params::expected_spikes_per_interval() const {
    return num_cells * network.min_delay*1e-3/2 * cell.spike_freq_hz;
}
unsigned bench_params::expected_events() const {
    return expected_spikes() * network.fan_in;
}
unsigned bench_params::expected_events_per_interval() const {
    return expected_spikes_per_interval() * network.fan_in;
}

std::ostream& operator<<(std::ostream& o, const bench_params& p) {
    o << "benchmark:\n"
      << "  name:          " << p.name << "\n"
      << "  num cells:     " << p.num_cells << "\n"
      << "  duration:      " << p.tfinal << " ms\n"
      << "  fan in:        " << p.network.fan_in << " connections/cell\n"
      << "  min delay:     " << p.network.min_delay << " ms\n"
      << "  spike freq:    " << p.cell.spike_freq_hz << " Hz\n"
      << "  cell overhead: " << p.expected_advance_rate() << " ms to advance 1 ms\n";
    o << "expected:\n"
      << "  cell advance: " << p.expected_advance_time() << " s\n"
      << "  spikes:       " << p.expected_spikes() << "\n"
      << "  events:       " << p.expected_events() << "\n"
      << "  spikes:       " << p.expected_spikes_per_interval() << " per interval\n"
      << "  events:       " << p.expected_events_per_interval()/p.num_cells << " per cell per interval\n";
    return o;
}

bench_params::bench_params(const std::string& fname) {
    std::ifstream f(fname);

    if (!f.good()) {
        throw std::runtime_error("unable to open input parameter file: "+fname);
    }

    nlohmann::json json;
    json << f;

    std::cout << json << "\n";
}
