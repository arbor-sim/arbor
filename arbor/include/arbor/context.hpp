#pragma once

#include <memory>

namespace arb {

// Requested dry-run parameters.
struct dry_run_info {
    unsigned num_ranks;
    unsigned num_cells_per_rank;
    dry_run_info(unsigned ranks, unsigned cells_per_rank):
            num_ranks(ranks),
            num_cells_per_rank(cells_per_rank) {}
};

// A description of local computation resources to use in a computation.
// By default, a proc_allocation will comprise one thread and no GPU.

struct proc_allocation {
    unsigned num_threads;

    // The gpu id corresponds to the `int device` parameter used by
    // CUDA API calls to identify gpu devices.
    // A gpud id of -1 indicates no GPU device is to be used.
    // See CUDA documenation for cudaSetDevice and cudaDeviceGetAttribute.
    int gpu_id;

    proc_allocation(): proc_allocation(1, -1) {}

    proc_allocation(unsigned threads, int gpu):
        num_threads(threads),
        gpu_id(gpu)
    {}

    bool has_gpu() const {
        return gpu_id>=0;
    }
};

// arb::execution_context encapsulates the execution resources used in
// a simulation, namely the task system thread pools, GPU handle, and
// MPI communicator if applicable.

// Forward declare execution_context.
struct execution_context;

// arb::context is an opaque handle for the execution context for use
// in the public API, implemented as a unique pointer.
//
// As execution_context is an incomplete type, an explicit deleter must be
// provided.
struct execution_context_deleter {
    void operator()(execution_context*) const;
};
using context = std::unique_ptr<execution_context, execution_context_deleter>;

// Helpers for creating contexts. These are implemented in the back end.

// Non-distributed context using the requested resources.
context make_context(const proc_allocation& resources = proc_allocation{});

// Distributed context that uses MPI communicator comm, and local resources
// described by resources. Or dry run context that uses dry_run_info.
template <typename Comm>
context make_context(const proc_allocation& resources, Comm comm);

// Queries for properties of execution resources in a context.

std::string distribution_type(const context&);
bool has_gpu(const context&);
unsigned num_threads(const context&);
bool has_mpi(const context&);
unsigned num_ranks(const context&);
unsigned rank(const context&);

}
