#include <algorithm>

#include "affinity.hpp"
#include "gpu.hpp"
#include "node_info.hpp"

namespace arb {
namespace hw {

// Return a node_info that describes the hardware resources available on this node.
// If unable to determine the number of available cores, assumes that there is one
// core available.
node_info get_node_info() {
    auto res = num_cores();
    unsigned ncpu = res? *res: 1u;
    return {ncpu, num_gpus()};
}

} // namespace util
} // namespace arb
