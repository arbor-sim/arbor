#pragma once

#include <memory>

#include <arbor/export.hpp>
#include <arbor/context.hpp>

#include "distributed_context.hpp"
#include "threading/threading.hpp"
#include "gpu_context.hpp"

namespace arb {

// execution_context is a simple container for the state relating to
// execution resources.
// Specifically, it has handles for the distributed context, gpu
// context and thread pool.
//
// Note: the public API uses an opaque handle arb::context for
// execution_context, to hide implementation details of the
// container and its constituent contexts from the public API.

struct ARB_ARBOR_API execution_context {
    distributed_context_handle distributed;
    task_system_handle thread_pool;
    gpu_context_handle gpu;

    execution_context(const proc_allocation& resources = proc_allocation{});

    // Use a template for constructing with a specific distributed context.
    // Specialised implementations are implemented in execution_context.cpp.
    template <typename Comm>
    execution_context(const proc_allocation& resources, Comm comm);
};

} // namespace arb
