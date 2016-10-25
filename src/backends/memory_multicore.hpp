#pragma once

#include <common_types.hpp>
#include <memory/memory.hpp>

namespace nest {
namespace mc {
namespace multicore {

template <typename T, typename I>
struct memory_traits_generic {
    // define basic types
    using value_type = T;
    using size_type  = I;

    // define storage types
    using array  = memory::host_vector<value_type>;
    using iarray   = memory::host_vector<size_type>;

    using view       = typename array::view_type;
    using const_view = typename array::const_view_type;

    using iview       = typename iarray::view_type;
    using const_iview = typename iarray::const_view_type;

    using host_array = array;
    using host_iarray  = iarray;
};

using memory_traits = memory_traits_generic<double, nest::mc::cell_lid_type>;

} // namespace multicore
} // namespace mc
} // namespace nest

