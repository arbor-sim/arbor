#pragma once

#include "vector/include/Vector.hpp"

using memory::util::red;
using memory::util::yellow;
using memory::util::green;
using memory::util::white;
using memory::util::blue;
using memory::util::cyan;

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

