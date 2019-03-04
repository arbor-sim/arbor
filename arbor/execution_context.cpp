#include <iostream>
#include <memory>

#include <arbor/context.hpp>

#include "gpu_context.hpp"
#include "distributed_context.hpp"
#include "execution_context.hpp"
#include "threading/threading.hpp"

#ifdef ARB_HAVE_MPI
#include <mpi.h>
#endif

namespace arb {

execution_context::execution_context(const proc_allocation& resources):
    distributed(make_local_context()),
    thread_pool(std::make_shared<threading::task_system>(resources.num_threads)),
    gpu(resources.has_gpu()? std::make_shared<gpu_context>(resources.gpu_id)
                           : std::make_shared<gpu_context>())
{}

context make_context(const proc_allocation& p) {
    return context(new execution_context(p), [](execution_context* p){delete p;});
}

#ifdef ARB_HAVE_MPI
template <>
execution_context::execution_context(const proc_allocation& resources, MPI_Comm comm):
    distributed(make_mpi_context(comm)),
    thread_pool(std::make_shared<threading::task_system>(resources.num_threads)),
    gpu(resources.has_gpu()? std::make_shared<gpu_context>(resources.gpu_id)
                           : std::make_shared<gpu_context>())
{}

template <>
context make_context<MPI_Comm>(const proc_allocation& p, MPI_Comm comm) {
    return context(new execution_context(p, comm), [](execution_context* p){delete p;});
}
#endif
template <>
execution_context::execution_context(
        const proc_allocation& resources,
        dry_run_info d):
        distributed(make_dry_run_context(d.num_ranks, d.num_cells_per_rank)),
        thread_pool(std::make_shared<threading::task_system>(resources.num_threads)),
        gpu(resources.has_gpu()? std::make_shared<gpu_context>(resources.gpu_id)
                               : std::make_shared<gpu_context>())
{}

template <>
context make_context(const proc_allocation& p, dry_run_info d) {
    return context(new execution_context(p, d), [](execution_context* p){delete p;});
}

std::string distribution_type(const context& ctx) {
    return ctx->distributed->name();
}

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

