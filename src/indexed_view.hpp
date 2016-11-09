#pragma once

#include <memory/memory.hpp>

namespace nest {
namespace mc {

template <typename MemoryTraits>
struct indexed_view {
    using memory_traits = MemoryTraits;
    using value_type  = typename memory_traits::value_type;
    using size_type   = typename memory_traits::size_type;
    using view        = typename memory_traits::view;
    using const_iview = typename memory_traits::const_iview;
    using reference   = typename view::reference;
    using const_reference = typename view::const_reference;

    view data;
    const_iview index;

    indexed_view(view v, const_iview i):
        data(v), index(i)
    {}

    std::size_t size() const {
        return index.size();
    }

    reference operator[] (std::size_t i) {
        return data[index[i]];
    }

    const_reference operator[] (std::size_t i) const {
        return data[index[i]];
    }
};

} // namespace mc
} // namespace nest
