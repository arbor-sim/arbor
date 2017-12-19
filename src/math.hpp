#pragma once

#include <cmath>
#include <limits>
#include <utility>

namespace arb {
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
    return (x>T(0)) - (x<T(0));
}

// Return minimum of the two values
template <typename T>
T min(const T& lhs, const T& rhs) {
    return lhs<rhs? lhs: rhs;
}

// Return maximum of the two values
template <typename T>
T max(const T& lhs, const T& rhs) {
    return lhs<rhs? rhs: lhs;
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

