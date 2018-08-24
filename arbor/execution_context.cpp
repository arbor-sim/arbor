#include <iostream>
#include <memory>

#include <arbor/context.hpp>

#include "distributed_context.hpp"
#include "threading/threading.hpp"
#include "gpu_context.hpp"

#include "execution_context.hpp"

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
    std::cout << "initialising with allocation: " << p.num_threads << "t; " << p.gpu_id << "g; " << p.has_gpu() << "?g\n";
    return context(new execution_context(p), [](execution_context* p){delete p;});
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

bool has_mpi(const context& ctx) {
    return ctx->distributed->name() == "mpi";
}

} // namespace arb

