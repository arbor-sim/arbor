#pragma once

namespace arb {

enum gpu_flags {
    has_concurrent_managed_access = 1,
    has_atomic_double = 2
};

struct gpu_context {
    bool has_gpu_;
    size_t attributes_;

    gpu_context();

    bool has_concurrent_managed_access() const;
    bool has_atomic_double() const;
    void synchronize_for_managed_access() const;
};

}
