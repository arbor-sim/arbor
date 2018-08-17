#pragma once

#include <memory>

#include <arbor/version.hpp>

namespace arb {
struct gpu_context {
    bool has_gpu_;
    size_t attributes_;
    gpu_context();

#ifdef ARB_GPU_ENABLED
    bool has_concurrent_managed_access();
    bool has_atomic_double();
#endif
};
}
