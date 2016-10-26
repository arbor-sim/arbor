#pragma once

#include <cmath>
#include <limits>
#include <utility>

namespace nest {
namespace mc {
namespace math {

template <typename T>
T constexpr pi()
{
    return T(3.1415926535897932384626433832795);
}

template <typename T = float>
T constexpr infinity()
{
    return std::numeric_limits<T>::infinity();
}

template <typename T>
T constexpr mean(T a, T b)
{
    return (a+b) / T(2);
}

template <typename T>
T constexpr mean(std::pair<T,T> const& p)
{
    return (p.first+p.second) / T(2);
}

template <typename T>
T constexpr square(T a)
{
    return a*a;
}

template <typename T>
T constexpr cube(T a)
{
    return a*a*a;
}

// area of a circle
//   pi r-squared
template <typename T>
T constexpr area_circle(T r)
{
    return pi<T>() * square(r);
}

// volume of a conic frustrum
template <typename T>
T constexpr area_frustrum(T L, T r1, T r2)
{
    return pi<T>() * (r1+r2) * sqrt(square(L) + square(r1-r2));
}

// surface area of conic frustrum, not including the area of the
// circles at either end (i.e. just the area of the surface of rotation)
template <typename T>
T constexpr volume_frustrum(T L, T r1, T r2)
{
    // this formulation uses one less multiplication
    return pi<T>()/T(3) * (square(r1+r2) - r1*r2) * L;
    //return pi<T>()/T(3) * (square(r1) + r1*r2 + square(r2)) * L;
}

// volume of a sphere
//   four-thirds pi r-cubed
template <typename T>
T constexpr volume_sphere(T r)
{
    return T(4)/T(3) * pi<T>() * cube(r);
}

// surface area of a sphere
//   four pi r-squared
template <typename T>
T constexpr area_sphere(T r)
{
    return T(4) * pi<T>() * square(r);
}

// linear interpolation in interval [a,b]
template <typename T, typename U>
T constexpr lerp(T a, T b, U u) {
    // (1-u)*a + u*b
    return std::fma(u, b, std::fma(-u, a, a));
}

// -1, 0 or 1 according to sign of parameter
template <typename T>
int signum(T x) {
    return (x<T(0)) - (x>T(0));
}

} // namespace math
} // namespace mc
} // namespace nest

