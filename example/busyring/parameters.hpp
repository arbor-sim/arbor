#include <iostream>

#include <array>
#include <cmath>
#include <fstream>
#include <random>

#include <sup/json_params.hpp>

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

    // The number of synapses per cell.
    unsigned synapses = 1;
};

struct ring_params {
    ring_params() = default;

    std::string name = "default";
    unsigned num_cells = 1000;
    unsigned ring_size = 10;
    double min_delay = 10;
    double duration = 100;
    double dt = 0.025;
    bool record_voltage = false;
    std::string odir = ".";
    cell_parameters cell;
};

static
ring_params read_options(int argc, char** argv) {
    const char* usage = "Usage:  arbor-busyring [params [opath]]\n\n"
                        "Driver for the Arbor busyring benchmark\n\n"
                        "Options:\n"
                        "   params: JSON file with model parameters.\n"
                        "   opath: output path.\n";
    using sup::param_from_json;

    ring_params params;
    if (argc<2) {
        return params;
    }
    if (argc>3) {
        std::cout << usage << std::endl;
        throw std::runtime_error("More than two command line options is not permitted.");
    }

    // Assume that the first argument is a json parameter file
    std::string fname = argv[1];
    std::ifstream f(fname);

    if (!f.good()) {
        throw std::runtime_error("Unable to open input parameter file: "+fname);
    }

    nlohmann::json json;
    f >> json;

    param_from_json(params.name, "name", json);
    param_from_json(params.num_cells, "num-cells", json);
    param_from_json(params.ring_size, "ring-size", json);
    param_from_json(params.duration, "duration", json);
    param_from_json(params.dt, "dt", json);
    param_from_json(params.min_delay, "min-delay", json);
    param_from_json(params.record_voltage, "record", json);
    param_from_json(params.cell.max_depth, "depth", json);
    param_from_json(params.cell.branch_probs, "branch-probs", json);
    param_from_json(params.cell.compartments, "compartments", json);
    param_from_json(params.cell.lengths, "lengths", json);
    param_from_json(params.cell.synapses, "synapses", json);

    if (!json.empty()) {
        for (auto it=json.begin(); it!=json.end(); ++it) {
            std::cout << "  Warning: unused input parameter: \"" << it.key() << "\"\n";
        }
        std::cout << "\n";
    }

    // Set optional output path if a second argument was passed
    if (argc==3) {
        params.odir = argv[2];
    }

    return params;
}
