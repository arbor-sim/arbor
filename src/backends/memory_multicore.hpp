#pragma once

namespace nest {
namespace mc {
namespace multicore {

template <typename T, typename I>
struct memory_policy {
    // define basic types
    using value_type = T;
    using size_type  = I;

    // define storage types
    using vector_type  = memory::HostVector<value_type>;
    using index_type   = memory::HostVector<size_type>;

    using view       = typename vector_type::view_type;
    using const_view = typename vector_type::const_view_type;

    using iview       = typename index_type::view_type;
    using const_iview = typename index_type::const_view_type;

    using host_vector_type = vector_type;
    using host_index_type  = index_type;
} // namespace multicore
} // namespace mc
} // namespace nest
