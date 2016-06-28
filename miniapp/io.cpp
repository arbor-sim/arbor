#include "io.hpp"

namespace nest {
namespace mc {
namespace io {

// read simulation options from json file with name fname
// for now this is just a placeholder
options read_options(std::string fname) {
    // 10 cells, 1 synapses per cell, 10 compartments per segment
    return {200, 1, 100};
}

std::ostream& operator<<(std::ostream& o, const options& opt) {
    o << "simultion options:\n";
    o << "  cells                : " << opt.cells << "\n";
    o << "  compartments/segment : " << opt.compartments_per_segment << "\n";
    o << "  synapses/cell        : " << opt.synapses_per_cell << "\n";

    return o;
}

} // namespace io
} // namespace mc
} // namespace nest
