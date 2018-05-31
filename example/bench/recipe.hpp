#pragma once

#include <ostream>
#include <string>

#include <common_types.hpp>
#include <recipe.hpp>

using arb::cell_kind;
using arb::cell_gid_type;
using arb::cell_size_type;
using arb::time_type;
using arb::cell_member_type;

struct bench_params {
    struct cell_params {
        double spike_freq_hz;   // frequency in hz that cell will generate (poisson) spikes.
        double us_per_ms;       // μs to advance one cell one ms of simlation time
    };
    struct network_params {
        unsigned fan_in;        // number of incoming connections on each cell
        double min_delay;       // used as the delay on all connections
    };
    std::string name;           // name of the model
    unsigned num_cells;         // number of cells in model
    cell_params cell;           // cell parameters for all cells in model
    network_params network;     // description of the network
    time_type tfinal;

    bench_params(const std::string& filename);
    bench_params() = default;

    double expected_advance_rate() const;
    double expected_advance_time() const;
    unsigned expected_spikes() const;
    unsigned expected_spikes_per_interval() const;
    unsigned expected_events() const;
    unsigned expected_events_per_interval() const;
};

class bench_recipe: public arb::recipe {
    bench_params params_;
public:
    bench_recipe(bench_params p): params_(std::move(p)) {}
    cell_size_type num_cells() const override;
    arb::util::unique_any get_cell_description(cell_gid_type gid) const override;
    arb::cell_kind get_cell_kind(arb::cell_gid_type gid) const override;
    cell_size_type num_targets(cell_gid_type gid) const override;
    cell_size_type num_sources(cell_gid_type gid) const override;
    std::vector<arb::cell_connection> connections_on(cell_gid_type) const override;
    //std::vector<arb::event_generator> event_generators(cell_gid_type gid) const override;
};

std::ostream& operator<<(std::ostream& o, const bench_params& p);
