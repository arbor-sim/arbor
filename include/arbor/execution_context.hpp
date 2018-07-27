#pragma once

#include <memory>
#include <string>

#include <arbor/distributed_context.hpp>
#include <arbor/util/pp_util.hpp>
#include <arbor/threadinfo.hpp>


namespace arb {
namespace threading {
    class task_system;
}
using task_system_handle = std::shared_ptr<threading::task_system>;
using distributed_context_handle = std::shared_ptr<distributed_context>;

task_system_handle make_thread_pool();

struct execution_context {
    distributed_context_handle distributed;
    task_system_handle thread_pool;

    execution_context(): distributed(std::make_shared<distributed_context>()),
                         thread_pool(arb::make_thread_pool())  {};

};

}
