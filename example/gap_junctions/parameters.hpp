#include <iostream>

#include <array>
#include <cmath>
#include <fstream>
#include <random>

#include <arbor/mc_cell.hpp>

#include <sup/json_params.hpp>

struct gap_params {
    gap_params() = default;

    std::string name = "default";
    unsigned cells_per_ring = 5;
    unsigned num_rings = 2;
    double duration = 100;
    double delay = 5;
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
    param_from_json(params.cells_per_ring, "num-cells", json);
    param_from_json(params.num_rings, "num-rings", json);
    param_from_json(params.duration, "duration", json);
    param_from_json(params.delay, "delay", json);

    if (!json.empty()) {
        for (auto it=json.begin(); it!=json.end(); ++it) {
            std::cout << "  Warning: unused input parameter: \"" << it.key() << "\"\n";
        }
        std::cout << "\n";
    }

    return params;
}
