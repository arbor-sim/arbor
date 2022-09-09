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

ARB_ARBOR_API context make_context(const proc_allocation& p) {
    return std::make_shared<execution_context>(p);
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
ARB_ARBOR_API context make_context<MPI_Comm>(const proc_allocation& p, MPI_Comm comm) {
    return std::make_shared<execution_context>(p, comm);
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
ARB_ARBOR_API context make_context(const proc_allocation& p, dry_run_info d) {
    return std::make_shared<execution_context>(p, d);
}

ARB_ARBOR_API std::string distribution_type(context ctx) {
    return ctx->distributed->name();
}

ARB_ARBOR_API bool has_gpu(context ctx) {
    return ctx->gpu->has_gpu();
}

ARB_ARBOR_API unsigned num_threads(context ctx) {
    return ctx->thread_pool->get_num_threads();
}

ARB_ARBOR_API unsigned num_ranks(context ctx) {
    return ctx->distributed->size();
}

ARB_ARBOR_API unsigned rank(context ctx) {
    return ctx->distributed->id();
}

ARB_ARBOR_API bool has_mpi(context ctx) {
    return ctx->distributed->name() == "MPI";
}

} // namespace arb

