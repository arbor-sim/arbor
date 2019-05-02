#include <iostream>

#include <array>
#include <cmath>
#include <fstream>
#include <random>

#include <arbor/cable_cell.hpp>

#include <sup/json_params.hpp>

struct gap_params {
    gap_params() = default;

    std::string name = "default";
    unsigned n_cables = 3;
    unsigned n_cells_per_cable = 5;
    double stim_duration = 30;
    double event_min_delay = 10;
    double event_weight = 0.05;
    double sim_duration = 100;
    bool print_all = true;
};

gap_params read_options(int argc, char** argv) {
    using sup::param_from_json;

    gap_params params;
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
    param_from_json(params.n_cables, "n-cables", json);
    param_from_json(params.n_cells_per_cable, "n-cells-per-cable", json);
    param_from_json(params.stim_duration, "stim-duration", json);
    param_from_json(params.event_min_delay, "event-min-delay", json);
    param_from_json(params.event_weight, "event-weight", json);
    param_from_json(params.sim_duration, "sim-duration", json);
    param_from_json(params.print_all, "print-all", json);

    if (!json.empty()) {
        for (auto it=json.begin(); it!=json.end(); ++it) {
            std::cout << "  Warning: unused input parameter: \"" << it.key() << "\"\n";
        }
        std::cout << "\n";
    }

    return params;
}
