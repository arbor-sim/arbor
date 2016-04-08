#pragma once

#include "vector/include/Vector.hpp"


/*
using memory::util::red;
using memory::util::yellow;
using memory::util::green;
using memory::util::white;
using memory::util::blue;
using memory::util::cyan;
*/

#include <memory>
#include <ostream>
#include <utility>
#include <vector>

template <typename T>
std::ostream&
operator << (std::ostream &o, std::vector<T>const& v)
{
    o << "[";
    for(auto const& i: v) {
        o << i << ", ";
    }
    o << "]";
    return o;
}

namespace nest {
namespace mc {
namespace util {

    // just because we aren't using C++14, doesn't mean we shouldn't go
    // without make_unique
    template <typename T, typename... Args>
    std::unique_ptr<T> make_unique(Args&&... args) {
        return std::unique_ptr<T>(new T(std::forward<Args>(args) ...));
    }

    /// helper for taking the first value in a std::pair
    template <typename L, typename R>
    L const& left(std::pair<L, R> const& p)
    {
        return p.first;
    }

    /// helper for taking the second value in a std::pair
    template <typename L, typename R>
    R const& right(std::pair<L, R> const& p)
    {
        return p.second;
    }

} // namespace util
} // namespace mc
} // namespace nest

