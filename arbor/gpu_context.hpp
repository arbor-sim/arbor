#pragma once

#include <cstdlib>
#include <memory>

#include <arbor/export.hpp>

namespace arb {

class ARB_ARBOR_API gpu_context {
    int id_ = -1;
    std::size_t attributes_ = 0;

public:
    gpu_context() = default;
    gpu_context(int id);

    bool has_atomic_double() const;
    bool has_gpu() const;
    // Calls set_device(id), so that GPU calls from the calling thread will
    // be executed on the GPU.
    void set_gpu() const;
};

using gpu_context_handle = std::shared_ptr<gpu_context>;
ARB_ARBOR_API gpu_context_handle make_gpu_context(int id);

} // namespace arb
