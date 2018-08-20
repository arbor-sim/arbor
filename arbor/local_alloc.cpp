#include <arbor/domain_decomposition.hpp>
#include <arbor/execution_context.hpp>

#include "hardware/node_info.hpp"
#include "threading/threading.hpp"

namespace arb {

proc_allocation local_allocation(const execution_context& ctx) {
    proc_allocation info;
    info.num_threads = ctx.thread_pool->get_num_threads();
    info.num_gpus = arb::hw::node_gpus();

    return info;
}

} // namespace arb
