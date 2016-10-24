#pragma once

#include "memory_traits.hpp"

namespace nest {
namespace mc {

/// enumerate different implementation back ends
class enum backend_kind {cpu, gpu};

/// use an impl namespace to define helper metafunctions for picking backend
/// specific policies
/*
namespace impl {
    /// memory_policy helper template
    template <backend_kind>
    struct memory_policy_picker {};

    template <>
    struct memory_policy_picker<backend_kind::cpu> {
        template <typename T, typename I>
        using type = multicore::memory_policy<T, I>;
    };

    template <>
    struct memory_policy_picker<backend_kind::gpu> {
        template <typename T, typename I>
        using type = gpu::memory_policy<T, I>;
    };
}

/// user space helper for picking a backend-specific memory policy
template <typename T, typename I, backend_kind B>
using memory_policy = typename impl::memory_policy_picker<B>::template type<T, I>;
*/

} // namespace mc
} // namespace nest
