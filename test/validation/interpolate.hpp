#pragma once

#include <cmath>

template <typename T, typename U>
inline T lerp(T a, T b, U u) {
    return std::fma(u, b, std::fma(-u, a, a));
}

// Piece-wise linear interpolation across a sequence of points (u_i, x_i),
// monotonically increasing in u.
// 
// Parameters get_u and get_x provide the accessors for the point sequence;
// consider moving to structured bindings in C++17 instead.

template <typename U, typename Seq, typename GetU, typename GetX>
auto pw_linear_interpolate(U u, const Seq& seq, GetU get_u, GetX get_x) {
    using std::begin;
    using std::end;
    using value_type = decltype(get_x(*begin(seq)));

    auto i = begin(seq);
    auto e = end(seq);

    if (i==e) {
        return value_type(NAN);
    }

    auto u0 = get_u(*i);
    auto x0 = get_x(*i);

    if (u<u0) {
        return x0;
    }

    while (++i!=e) {
        auto u1 = get_u(*i);
        auto x1 = get_x(*i);

        if (u<u1) {
            return lerp(x0, x1, (u-u0)/(u1-u0));
        }

        u0 = u1;
        x0 = x1;
    }

    return x0;
}

