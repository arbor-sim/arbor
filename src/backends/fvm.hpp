#pragma once

#include <backends/multicore/fvm.hpp>

namespace arb {

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

} // namespace arb

#ifdef ARB_HAVE_GPU
#include <backends/gpu/fvm.hpp>
#else
namespace arb {  namespace gpu {
    using backend = null_backend;
}} // namespace arb::gpu
#endif
