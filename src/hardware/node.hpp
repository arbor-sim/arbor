#pragma once

namespace nest {
namespace mc {
namespace hw {

// Information about the computational resources available on a compute node.
// Currently a simple enumeration of the number of cpu cores and gpus, which
// will become richer.
struct node {
    node();
    node(int c, int g);

    int num_cpu_cores;
    int num_gpus;
};

} // namespace util
} // namespace mc
} // namespace nest
