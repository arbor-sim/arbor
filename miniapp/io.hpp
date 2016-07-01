#pragma once

#include <json/src/json.hpp>

namespace nest {
namespace mc {
namespace io {

// holds the options for a simulation run
struct options {
    int cells;
    int synapses_per_cell;
    int compartments_per_segment;
};

std::ostream& operator<<(std::ostream& o, const options& opt);

options read_options(std::string fname);

} // namespace io
} // namespace mc
} // namespace nest
