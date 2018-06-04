#pragma once

#include <ostream>
#include <string>

#include <common_types.hpp>

using arb::time_type;

struct bench_params {
    struct cell_params {
        double spike_freq_hz = 20;   // Frequency in hz that cell will generate (poisson) spikes.
        double us_per_ms = 100;      // μs to advance one cell one ms of simlation time.
    };
    struct network_params {
        unsigned fan_in = 5000;      // Number of incoming connections on each cell.
        double min_delay = 10;       // Used as the delay on all connections.
    };
    std::string name = "default";    // Name of the model.
    unsigned num_cells = 1000;       // Number of cells in model.
    time_type duration = 100;          // Simulation duration in ms.

    cell_params cell;                // Cell parameters for all cells in model.
    network_params network;          // Description of the network.

    // Expected simulation performance properties based on model parameters.
    double expected_advance_rate() const;
    double expected_advance_time() const;
    unsigned expected_spikes() const;
    unsigned expected_spikes_per_interval() const;
    unsigned expected_events() const;
    unsigned expected_events_per_interval() const;
};

bench_params read_options(int argc, char** argv);

std::ostream& operator<<(std::ostream& o, const bench_params& p);
