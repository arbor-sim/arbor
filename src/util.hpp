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

namespace util {
    // just because we aren't using C++14, doesn't mean we shouldn't go
    // without make_unique
    template <typename T, typename... Args>
    std::unique_ptr<T> make_unique(Args&&... args) {
        return std::unique_ptr<T>(new T(std::forward<Args>(args) ...));
    }
}

