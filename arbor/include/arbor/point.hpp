#pragma once

#include <cmath>
#include <limits>
#include <ostream>

namespace arb {

template <typename T>
struct point {
    using value_type = T;
    value_type x;
    value_type y;
    value_type z;

    constexpr point(T a, T b, T c) :
        x{a}, y{b}, z{c}
    {}

    // initialize to NaN by default
    constexpr point() :
        x(std::numeric_limits<T>::quiet_NaN()),
        y(std::numeric_limits<T>::quiet_NaN()),
        z(std::numeric_limits<T>::quiet_NaN())
    {}

    constexpr bool is_set() const {
        return (x==x && y==y && z==z);
    }
};

template <typename T>
constexpr point<T>
operator+(point<T> const& lhs, point<T> const& rhs) {
    return point<T>(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z);
}

template <typename T>
constexpr point<T>
operator-(point<T> const& lhs, point<T> const& rhs) {
    return point<T>(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z);
}

template <typename T>
constexpr point<T>
operator*(T alpha, point<T> const& p) {
    return point<T>(alpha*p.x, alpha*p.y, alpha*p.z);
}

template <typename T>
T norm(point<T> const& p) {
    return std::sqrt(p.x*p.x + p.y*p.y + p.z*p.z);
}

template <typename T>
constexpr T
dot(point<T> const& lhs, point<T> const& rhs) {
    return lhs.x*rhs.x + lhs.y*rhs.y + lhs.z*rhs.z;
}

template<typename T>
bool operator==(const point<T> &rhs, const point<T> &lhs) {
    return (rhs.x==lhs.x) && (rhs.y==lhs.y) && (rhs.z==lhs.z);
}

template<typename T>
bool operator!=(const point<T> &rhs, const point<T> &lhs) {
    return !(rhs == lhs);
}

} // namespace arb

template <typename T>
std::ostream& operator << (std::ostream& o, arb::point<T> const& p) {
    return o << "[" << p.x << ", " << p.y << ", " << p.z << "]";
}
