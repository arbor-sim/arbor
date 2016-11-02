#pragma once

#include <common_types.hpp>
#include <memory/memory.hpp>

namespace nest {
namespace mc {
namespace gpu {

template <typename T, typename I>
struct memory_traits_generic {
    // define basic types
    using value_type = T;
    using size_type  = I;

    // define storage types
    using array  = memory::device_vector<value_type>;
    using iarray = memory::device_vector<size_type>;

    using view       = typename array::view_type;
    using const_view = typename array::const_view_type;

    using iview       = typename iarray::view_type;
    using const_iview = typename iarray::const_view_type;

    using host_array  = typename memory::host_vector<value_type>;
    using host_iarray = typename memory::host_vector<size_type>;

    using host_view   = typename host_iarray::view_type;
    using host_iview  = typename host_iarray::const_view_type;
};

using memory_traits = memory_traits_generic<double, nest::mc::cell_lid_type>;

} // namespace gpu
} // namespace mc
} // namespace nest

