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

// Forward declare execution_context, then use a unique_ptr as a handle.
// Because execution_context is an incomplete type, we have to provide
// a destructor prototype for unique_ptr.
struct execution_context;
using context = std::unique_ptr<execution_context, void (*)(execution_context*)>;

context make_context();
context make_context(const proc_allocation&);
template <typename Comm>
context make_context(const proc_allocation&, Comm);

bool has_gpu(const context&);
unsigned num_threads(const context&);
bool has_mpi(const context&);
unsigned num_ranks(const context&);

}
