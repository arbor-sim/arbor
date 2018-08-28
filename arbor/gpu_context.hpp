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

    bool has_concurrent_managed_access() const;
    bool has_atomic_double() const;
    void synchronize_for_managed_access() const;
    bool has_gpu() const;
};

using gpu_context_handle = std::shared_ptr<gpu_context>;
gpu_context_handle make_gpu_context(int id);

} // namespace arb
