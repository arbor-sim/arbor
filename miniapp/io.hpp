#pragma once

#include <json/src/json.hpp>

namespace nest {
namespace mc {
namespace io {

// holds the options for a simulation run
struct cl_options {
    std::string ifname;
    uint32_t cells;
    uint32_t synapses_per_cell;
    uint32_t compartments_per_segment;
    double tfinal;
    double dt;
    bool all_to_all;

    // TODO the normalize bit should be moved to the model_parameters when
    // we start having more models
    void check_and_normalize() {
        if(all_to_all) {
            synapses_per_cell = cells - 1;
        }
    }
};

std::ostream& operator<<(std::ostream& o, const cl_options& opt);

cl_options read_options(int argc, char** argv);

} // namespace io
} // namespace mc
} // namespace nest
