#include "affinity.hpp"
#include "gpu.hpp"
#include "node.hpp"

namespace nest {
namespace mc {
namespace hw {

node::node():
    num_gpus(num_available_gpus())
{
    // If unable to determine the number of cores, use 1 core by default
    auto avail = num_available_cores();
    if (!avail || *avail==0u) {
        num_cpu_cores = 1;
    }
    num_cpu_cores = *avail;
}

node::node(int c, int g):
    num_cpu_cores(c), num_gpus(g)
{}

} // namespace util
} // namespace mc
} // namespace nest
