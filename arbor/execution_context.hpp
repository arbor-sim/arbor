#pragma once

#include <memory>

#include <arbor/context.hpp>

#include "distributed_context.hpp"
#include "threading/threading.hpp"
#include "gpu_context.hpp"

namespace arb {

struct execution_context {
    std::shared_ptr<distributed_context> distributed;
    std::shared_ptr<threading::task_system> thread_pool;
    std::shared_ptr<gpu_context> gpu;

    execution_context();
    execution_context(const proc_allocation& resources);
};

} // namespace arb
