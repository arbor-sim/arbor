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

    view data;
    const_iview index;

    indexed_view(view v, const_iview i) :
        data(v), index(i)
    {}

    std::size_t size() const {
        return index.size();
    }

    // TODO
    //
    //  these should either
    //      - return the internal reference type implementations from the containers
    //          e.g. the GPU reference that does a copy-on-read from GPU memory
    //      - or ensure that the result of dereferencing these is properly handled
    //          i.e. by just returning a value for the const version

    value_type& operator[] (std::size_t i) {
        return data[index[i]];
    }

    value_type operator[] (std::size_t i) const {
        return data[index[i]];
    }
};

} // namespace mc
} // namespace nest
