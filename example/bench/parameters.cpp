#include <exception>
#include <fstream>
#include <iostream>
#include <string>

#include <aux/json_params.hpp>

#include "parameters.hpp"

double bench_params::expected_advance_time() const {
    return cell.realtime_ratio * duration*1e-3 * num_cells;
}
unsigned bench_params::expected_spikes() const {
    return num_cells * duration*1e-3 * cell.spike_freq_hz;
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
    o << "benchmark parameters:\n"
      << "  name:          " << p.name << "\n"
      << "  num cells:     " << p.num_cells << "\n"
      << "  duration:      " << p.duration << " ms\n"
      << "  fan in:        " << p.network.fan_in << " connections/cell\n"
      << "  min delay:     " << p.network.min_delay << " ms\n"
      << "  spike freq:    " << p.cell.spike_freq_hz << " Hz\n"
      << "  cell overhead: " << p.cell.realtime_ratio << " ms to advance 1 ms\n";
    o << "expected:\n"
      << "  cell advance: " << p.expected_advance_time() << " s\n"
      << "  spikes:       " << p.expected_spikes() << "\n"
      << "  events:       " << p.expected_events() << "\n"
      << "  spikes:       " << p.expected_spikes_per_interval() << " per interval\n"
      << "  events:       " << p.expected_events_per_interval()/p.num_cells << " per cell per interval";
    return o;
}

bench_params read_options(int argc, char** argv) {
    using aux::param_from_json;

    bench_params params;
    if (argc<2) {
        std::cout << "Using default parameters.\n";
        return params;
    }
    if (argc>2) {
        throw std::runtime_error("More than command line one option not permitted.");
    }

    std::string fname = argv[1];
    std::cout << "Loading parameters from file: " << fname << "\n";
    std::ifstream f(fname);

    if (!f.good()) {
        throw std::runtime_error("Unable to open input parameter file: "+fname);
    }

    nlohmann::json json;
    json << f;

    param_from_json(params.name, "name", json);
    param_from_json(params.num_cells, "num-cells", json);
    param_from_json(params.duration, "duration", json);
    param_from_json(params.network.min_delay, "min-delay", json);
    param_from_json(params.network.fan_in, "fan-in", json);
    param_from_json(params.cell.realtime_ratio, "realtime-ratio", json);
    param_from_json(params.cell.spike_freq_hz, "spike-frequency", json);

    for (auto it=json.begin(); it!=json.end(); ++it) {
        std::cout << "  Warning: unused input parameter: \"" << it.key() << "\"\n";
    }
    std::cout << "\n";

    return params;
}
