#pragma once

#include <cmath>
#include <limits>
#include <type_traits>
#include <utility>

#include <arbor/util/compat.hpp>

namespace arb {
namespace math {

template <typename T>
T constexpr pi = 3.1415926535897932384626433832795l;

template <typename T = float>
T constexpr infinity = std::numeric_limits<T>::infinity();

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
    return pi<T> * square(r);
}

// Surface area of conic frustrum excluding the discs at each end,
// with length L, end radii r1, r2.
template <typename T>
T constexpr area_frustrum(T L, T r1, T r2) {
    return pi<T> * (r1+r2) * std::sqrt(square(L) + square(r1-r2));
}

// Volume of conic frustrum of length L, end radii r1, r2.
template <typename T>
T constexpr volume_frustrum(T L, T r1, T r2) {
    return pi<T>/T(3) * (square(r1+r2) - r1*r2) * L;
}

// Linear interpolation by u in interval [a,b]: (1-u)*a + u*b.
template <typename T, typename U>
T constexpr lerp(T a, T b, U u) {
    return compat::fma(T(u), b, compat::fma(T(-u), a, a));
}

// Return -1, 0 or 1 according to sign of parameter.
template <typename T>
int signum(T x) {
    return (x>T(0)) - (x<T(0));
}

// Next integral power of 2 for unsigned integers:
//
// next_pow2(x) returns 0 if x==0, else returns smallest 2^k such
// that 2^k>=x.

template <typename U, typename = std::enable_if_t<std::is_unsigned<U>::value>>
U next_pow2(U x) {
    --x;
    for (unsigned s=1; s<std::numeric_limits<U>::digits; s<<=1) {
        x|=(x>>s);
    }
    return ++x;
}

namespace impl {
    template <typename T>
    T abs_if_signed(const T& x, std::true_type) {
        return std::abs(x);
    }

    template <typename T>
    T abs_if_signed(const T& x, std::false_type) {
        return x;
    }
}

// round_up(v, b) returns r, the smallest magnitude multiple of b
// such that v lies between 0 and r inclusive.
//
// Examples:
//     round_up( 7,  3) ==  9
//     round_up( 7, -3) ==  9
//     round_up(-7,  3) == -9
//     round_up(-7, -3) == -9
//     round_up( 8,  4) ==  8

template <
    typename T,
    typename U,
    typename C = std::common_type_t<T, U>,
    typename Signed = std::is_signed<C>
>
C round_up(T v, U b) {
    C m = v%b;
    return v-m+signum(m)*impl::abs_if_signed(b, Signed{});
}

// Returns 1/x if x != 0; 0 otherwise
template <typename T>
inline
T safeinv(T x) {
    // If abs(x) is less than epsilon return epsilon, else calculate the result directly.
    return (T(1)==T(1)+x)? 1/std::numeric_limits<T>::epsilon(): 1/x;
}

// Value of x/(exp(x)-1) with care taken to handle x=0 case
template <typename T>
inline
T exprelr(T x) {
    // If abs(x) is less than epsilon return 1, else calculate the result directly.
    return (T(1)==T(1)+x)? T(1): x/std::expm1(x);
}

// Quaternion implementation.
// Represents w + x.i + y.j + z.k.

struct quaternion {
    double w, x, y, z;

    constexpr quaternion(): w(0), x(0), y(0), z(0) {}

    // scalar
    constexpr quaternion(double w): w(w), x(0), y(0), z(0) {}

    // vector (pure imaginary)
    constexpr quaternion(double x, double y, double z): w(0), x(x), y(y), z(z) {}

    // all 4-components
    constexpr quaternion(double w, double x, double y, double z): w(w), x(x), y(y), z(z) {}

    // equality testing
    constexpr bool operator==(quaternion q) const {
        return w==q.w && x==q.x && y==q.y && z==q.z;
    }

    constexpr bool operator!=(quaternion q) const {
        return !(*this==q);
    }

    constexpr quaternion conj() const {
        return {w, -x, -y, -z};
    }

    constexpr quaternion operator*(quaternion q) const {
        return {w*q.w-x*q.x-y*q.y-z*q.z,
                w*q.x+x*q.w+y*q.z-z*q.y,
                w*q.y-x*q.z+y*q.w+z*q.x,
                w*q.z+x*q.y-y*q.x+z*q.w};
    }

    quaternion& operator*=(quaternion q) {
        return (*this=*this*q);
    }

    constexpr quaternion operator*(double d) const {
        return {w*d, x*d, y*d, z*d};
    }

    quaternion& operator*=(double d) {
        return (*this=*this*d);
    }

    friend constexpr quaternion operator*(double d, quaternion q) {
        return q*d;
    }

    constexpr quaternion operator+(quaternion q) const {
        return {w+q.w, x+q.x, y+q.y, z+q.z};
    }

    quaternion& operator+=(quaternion q) {
        w += q.w;
        x += q.x;
        y += q.y;
        z += q.z;
        return *this;
    }

    constexpr quaternion operator-() const {
        return {-w, -x, -y, -z};
    }

    constexpr quaternion operator-(quaternion q) const {
        return {w-q.w, x-q.x, y-q.y, z-q.z};
    }

    quaternion& operator-=(quaternion q) {
        w -= q.w;
        x -= q.x;
        y -= q.y;
        z -= q.z;
        return *this;
    }

    constexpr double sqnorm() const {
        return w*w+x*x+y*y+z*z;
    }

    double norm() const {
        return std::sqrt(sqnorm());
    }

    // Conjugation a ^ b = b a b*.
    constexpr quaternion operator^(quaternion b) const {
        return b*(*this)*b.conj();
    }

    quaternion& operator^=(quaternion b) {
        return *this = b*(*this)*b.conj();
    }

    // add more as required...
};

// Quaternionic representations of axis rotations.
// Given a vector v = (x, y, z), and r a quaternion representing
// a rotation as below, then then (0, x, y, z) ^ r = (0, x', y', z')
// represents the rotated vector.

inline quaternion rotation_x(double phi) {
    return {std::cos(phi/2), std::sin(phi/2), 0, 0};
}

inline quaternion rotation_y(double theta) {
    return {std::cos(theta/2), 0, std::sin(theta/2), 0};
}

inline quaternion rotation_z(double psi) {
    return {std::cos(psi/2), 0, 0, std::sin(psi/2)};
}

} // namespace math
} // namespace arb

