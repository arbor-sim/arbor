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
};

std::ostream& operator<<(std::ostream& o, const cl_options& opt);

cl_options read_options(int argc, char** argv);

} // namespace io
} // namespace mc
} // namespace nest
