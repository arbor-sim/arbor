#pragma once

#include <cmath>
#include <limits>
#include <utility>

namespace nest {
namespace mc {
namespace math {

template <typename T>
T constexpr pi() {
    return T(3.1415926535897932384626433832795l);
}

template <typename T = float>
T constexpr infinity() {
    return std::numeric_limits<T>::infinity();
}

template <typename T>
T constexpr mean(T a, T b) {
    return (a+b) / T(2);
}

template <typename T>
T constexpr mean(std::pair<T,T> const& p) {
    return (p.first+p.second) / T(2);
}

template <typename T>
T constexpr square(T a) {
    return a*a;
}

template <typename T>
T constexpr cube(T a) {
    return a*a*a;
}

// Area of circle radius r.
template <typename T>
T constexpr area_circle(T r) {
    return pi<T>() * square(r);
}

// Surface area of conic frustrum excluding the discs at each end,
// with length L, end radii r1, r2.
template <typename T>
T constexpr area_frustrum(T L, T r1, T r2) {
    return pi<T>() * (r1+r2) * sqrt(square(L) + square(r1-r2));
}

// Volume of conic frustrum of length L, end radii r1, r2.
template <typename T>
T constexpr volume_frustrum(T L, T r1, T r2) {
    return pi<T>()/T(3) * (square(r1+r2) - r1*r2) * L;
}

// Volume of a sphere radius r.
template <typename T>
T constexpr volume_sphere(T r) {
    return T(4)/T(3) * pi<T>() * cube(r);
}

// Surface area of a sphere radius r.
template <typename T>
T constexpr area_sphere(T r) {
    return T(4) * pi<T>() * square(r);
}

// Linear interpolation by u in interval [a,b]: (1-u)*a + u*b.
template <typename T, typename U>
T constexpr lerp(T a, T b, U u) {
    return std::fma(u, b, std::fma(-u, a, a));
}

// Return -1, 0 or 1 according to sign of parameter.
template <typename T>
int signum(T x) {
    return (x<T(0)) - (x>T(0));
}

} // namespace math
} // namespace mc
} // namespace nest

