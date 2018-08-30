#pragma once

#include <memory>

namespace arb {

/// Summary of all available local computation resource.
struct local_resources {
    const unsigned num_threads;
    const unsigned num_gpus;

    local_resources(unsigned threads, unsigned gpus):
        num_threads(threads),
        num_gpus(gpus)
    {}
};

/// Determine available local domain resources.
local_resources get_local_resources();

/// A subset of local computation resources to use in a computation.
struct proc_allocation {
    unsigned num_threads;

    // The gpu id corresponds to the `int device` parameter used by CUDA API calls
    // to identify gpu devices.
    // Set to -1 to indicate that no GPU device is to be used.
    // see CUDA documenation for cudaSetDevice and cudaDeviceGetAttribute 
    int gpu_id;

    // By default a proc_allocation will take all available threads and the
    // GPU with id 0, if available.
    proc_allocation() {
        auto avail = get_local_resources();

        // By default take all available threads.
        num_threads = avail.num_threads;

        // Take the first GPU, if available.
        gpu_id = avail.num_gpus>0? 0: -1;
    }

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
// described by resources.
template <typename Comm>
context make_context(const proc_allocation& resources, Comm comm);

// Queries for properties of execution resources in a context.

bool has_gpu(const context&);
unsigned num_threads(const context&);
bool has_mpi(const context&);
unsigned num_ranks(const context&);
unsigned rank(const context&);

}
