#pragma once

namespace nest {
namespace mc {
namespace hw {

// Information about the computational resources available on a compute node.
// Currently a simple enumeration of the number of cpu cores and gpus, which
// will become richer.
struct node_info {
    node_info() = default;
    node_info(unsigned c, unsigned g):
        num_cpu_cores(c), num_gpus(g)
    {}

    unsigned num_cpu_cores = 1;
    unsigned num_gpus = 0;
};

node_info get_node_info();

} // namespace util
} // namespace mc
} // namespace nest
