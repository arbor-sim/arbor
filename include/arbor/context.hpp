#pragma once

#include <memory>

namespace arb {

/// Requested dry-run parameters
struct dry_run_info {
    unsigned num_ranks;
    unsigned num_cells_per_rank;
    dry_run_info(unsigned ranks, unsigned cells_per_rank):
            num_ranks(ranks),
            num_cells_per_rank(cells_per_rank) {}
};

/// A subset of local computation resources to use in a computation.
struct proc_allocation {
    unsigned num_threads;

    // The gpu id corresponds to the `int device` parameter used by CUDA API calls
    // to identify gpu devices.
    // Set to -1 to indicate that no GPU device is to be used.
    // see CUDA documenation for cudaSetDevice and cudaDeviceGetAttribute 
    int gpu_id;

    // By default a proc_allocation uses one thread and no GPU.
    proc_allocation(): proc_allocation(1, -1) {}

    proc_allocation(unsigned threads, int gpu):
        num_threads(threads),
        gpu_id(gpu)
    {}

    bool has_gpu() const {
        return gpu_id>=0;
    }
};

// arb::execution_context is a container defined in the implementation for state
// related to execution resources, specifically thread pools, gpus and MPI
// communicators.

// Forward declare execution_context.
struct execution_context;

// arb::context is an opaque handle for this container presented in the
// public API.
// It doesn't make sense to copy contexts, so we use a std::unique_ptr to
// implement the handle with lifetime management.
//
// Because execution_context is an incomplete type, a destructor prototype must
// be provided.
using context = std::unique_ptr<execution_context, void (*)(execution_context*)>;


// Helpers for creating contexts. These are implemented in the back end.

// Non-distributed context that uses all detected threads and one GPU if available.
context make_context();

// Non-distributed context that uses resources described by resources
context make_context(const proc_allocation& resources);

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
