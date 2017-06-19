#pragma once

#include <backends/multicore/fvm.hpp>

namespace nest {
namespace mc {

// A null back end used as a placeholder for back ends that are not supported
// on the target platform.
struct null_backend: public multicore::backend {
    static bool is_supported() {
        return false;
    }

    static mechanism make_mechanism(
        const std::string&,
        size_type,
        const_iview,
        const_view, const_view, const_view,
        view, view,
        const std::vector<value_type>&,
        const std::vector<size_type>&)
    {
        throw std::runtime_error("attempt to use an unsupported back end");
    }

    static bool has_mechanism(const std::string& name) {
        return false;
    }

    static std::string name() {
        return "null";
    }
};

} // namespace mc
} // namespace nest

// FIXME: This include is where cuda-specific code leaks into the main application.
// e.g.: CUDA kernels, functions marked __host__ __device__, etc.
// Hence why it is guarded with NMC_HAVE_CUDA, and not, NMC_HAVE_GPU, like elsewhere in
// the code. When we implement separate compilation of CUDA, this should be guarded with
// NMC_HAVE_GPU, and the NMC_HAVE_CUDA flag depricated.
#ifdef NMC_HAVE_CUDA
    #include <backends/gpu/fvm.hpp>
#else
namespace nest { namespace mc { namespace gpu {
    using backend = null_backend;
}}} // namespace nest::mc::gpu
#endif
