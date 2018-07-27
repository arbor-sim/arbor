#include <arbor/domain_decomposition.hpp>
#include <arbor/threadinfo.hpp>
#include <arbor/execution_context.hpp>
#include <threading/cthread.hpp>

#include "hardware/node_info.hpp"

namespace arb {

proc_allocation local_allocation(execution_context* ctx) {
    proc_allocation info;
    info.num_threads = ctx->thread_pool->get_num_threads();
    info.num_gpus = arb::hw::node_gpus();

    return info;
}

} // namespace arb
