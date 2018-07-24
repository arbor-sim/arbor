#pragma once

#include <memory>
#include <string>

#include <arbor/domain_decomposition.hpp>
#include <arbor/distributed_context.hpp>
#include <arbor/util/pp_util.hpp>
#include <arbor/threadinfo.hpp>


namespace arb {
namespace threading {
    class task_system;
}
using task_system_handle = std::shared_ptr<threading::task_system>;

task_system_handle make_thread_pool (int nthreads);

struct execution_context {
    // TODO: use a shared_ptr for distributed_context
    distributed_context distributed;
    task_system_handle thread_pool;

    execution_context(): thread_pool(arb::make_thread_pool(arb::num_threads())) {};
    execution_context(proc_allocation nd): thread_pool(arb::make_thread_pool(nd.num_threads)) {};
};

}
