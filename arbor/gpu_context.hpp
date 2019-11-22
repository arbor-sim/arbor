#pragma once

#include <cstdlib>
#include <memory>

namespace arb {

class gpu_context {
    int id_ = -1;
    std::size_t attributes_ = 0;

public:
    gpu_context() = default;
    gpu_context(int id);

    bool has_atomic_double() const;
    bool has_gpu() const;
    // Calls cudaSetDevice(id), so that GPU calls from the calling thread will
    // be executed on the GPU.
    void set_gpu() const;
};

using gpu_context_handle = std::shared_ptr<gpu_context>;
gpu_context_handle make_gpu_context(int id);

} // namespace arb
