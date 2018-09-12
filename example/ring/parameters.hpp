#include <iostream>

#include <array>
#include <cmath>
#include <fstream>
#include <random>

#include <arbor/mc_cell.hpp>

#include <ancillary/json_params.hpp>

// Parameters used to generate the random cell morphologies.
struct cell_parameters {
    cell_parameters() = default;

    //  Maximum number of levels in the cell (not including the soma)
    unsigned max_depth = 5;

    // The following parameters are described as ranges.
    // The first value is at the soma, and the last value is used on the last level.
    // Values at levels in between are found by linear interpolation.
    std::array<double,2> branch_probs = {1.0, 0.5}; //  Probability of a branch occuring.
    std::array<unsigned,2> compartments = {20, 2};  //  Compartment count on a branch.
    std::array<double,2> lengths = {200, 20};       //  Length of branch in μm.
};

struct ring_params {
    ring_params() = default;

    std::string name = "default";
    unsigned num_cells = 10;
    double min_delay = 10;
    double duration = 100;
    cell_parameters cell;
};

ring_params read_options(int argc, char** argv) {
    using anc::param_from_json;

    ring_params params;
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
    param_from_json(params.min_delay, "min-delay", json);
    param_from_json(params.cell.max_depth, "depth", json);
    param_from_json(params.cell.branch_probs, "branch-probs", json);
    param_from_json(params.cell.compartments, "compartments", json);
    param_from_json(params.cell.lengths, "lengths", json);

    if (!json.empty()) {
        for (auto it=json.begin(); it!=json.end(); ++it) {
            std::cout << "  Warning: unused input parameter: \"" << it.key() << "\"\n";
        }
        std::cout << "\n";
    }

    return params;
}
