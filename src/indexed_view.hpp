#pragma once

#include <vector/Vector.hpp>

namespace nest {
namespace mc {

template <typename T, typename I>
struct indexed_view {
    using value_type      = T;
    using size_type       = I;
    using view_type       = typename memory::HostVector<T>::view_type;
    using index_view_type = typename memory::HostVector<I>::view_type;

    view_type       view;
    index_view_type index; // TODO make this a const view

    indexed_view(view_type v, index_view_type i)
    :   view(v),
        index(i)
    {}

    size_type size() const
    {
        return index.size();
    }

    // TODO
    //
    //  these should either
    //      - return the internal reference type implementations from the containers
    //          e.g. the GPU reference that does a copy-on-read from GPU memory
    //      - or ensure that the result of dereferencing these is properly handled
    //          i.e. by just returning a value for the const version

    value_type&
    operator[] (const size_type i)
    {
        return view[index[i]];
    }

    value_type const&
    operator[] (const size_type i) const
    {
        return view[index[i]];
    }
};

#ifdef WITH_CUDA
namespace gpu {

template <typename T, typename I>
struct indexed_view {
    using value_type      = T;
    using size_type       = I;
    using view_type       = typename memory::DeviceVector<T>::view_type;
    using index_view_type = typename memory::DeviceVector<I>::view_type;

    view_type       view;
    index_view_type index; // TODO make this a const view

    indexed_view(view_type v, index_view_type i)
    :   view(v),
        index(i)
    {}

    size_type size() const {
        return index.size();
    }

    // TODO
    //
    //  these should either
    //      - return the internal reference type implementations from the containers
    //          e.g. the GPU reference that does a copy-on-read from GPU memory
    //      - or ensure that the result of dereferencing these is properly handled
    //          i.e. by just returning a value for the const version

    value_type&
    operator[] (const size_type i)
    {
        return view[index[i]];
    }

    value_type const&
    operator[] (const size_type i) const
    {
        return view[index[i]];
    }
};

} // namespace gpu
#endif

} // namespace mc
} // namespace nest
