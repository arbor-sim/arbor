#include <iostream>
#include <memory>

#include <arbor/context.hpp>
#include <arbor/version.hpp>

#include "gpu_context.hpp"
#include "distributed_context.hpp"
#include "execution_context.hpp"
#include "threading/threading.hpp"

#ifdef ARB_MPI_ENABLED
#include <mpi.h>
#endif

namespace arb {

execution_context::execution_context():
    execution_context(proc_allocation())
{}

execution_context::execution_context(const proc_allocation& resources):
    distributed(std::make_shared<distributed_context>()),
    thread_pool(std::make_shared<threading::task_system>(resources.num_threads)),
    gpu(resources.has_gpu()? std::make_shared<gpu_context>(resources.gpu_id)
                           : std::make_shared<gpu_context>())
{}

context make_context() {
    return context(new execution_context(), [](execution_context* p){delete p;});
}

context make_context(const proc_allocation& p) {
    return context(new execution_context(p), [](execution_context* p){delete p;});
}

#ifdef ARB_MPI_ENABLED
template <>
execution_context::execution_context<MPI_Comm>(const proc_allocation& resources, MPI_Comm comm):
    distributed(mpi_context(comm)),
    thread_pool(std::make_shared<threading::task_system>(resources.num_threads)),
    gpu(resources.has_gpu()? std::make_shared<gpu_context>(resources.gpu_id)
                           : std::make_shared<gpu_context>())
{}

template <>
context make_context<MPI_Comm>(const proc_allocation& p, MPI_Comm comm) {
    return context(new execution_context(p, comm), [](execution_context* p){delete p;});
}
#endif

bool has_gpu(const context& ctx) {
    return ctx->gpu->has_gpu();
}

unsigned num_threads(const context& ctx) {
    return ctx->thread_pool->get_num_threads();
}

unsigned num_ranks(const context& ctx) {
    return ctx->distributed->size();
}

unsigned rank(const context& ctx) {
    return ctx->distributed->id();
}

bool has_mpi(const context& ctx) {
    return ctx->distributed->name() == "MPI";
}

} // namespace arb

