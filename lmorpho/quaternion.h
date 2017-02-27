#pragma once

#include <cmath>

struct quaternion {
    double w, x, y, z;

    constexpr quaternion(): w(0), x(0), y(0), z(0) {}

    // scalar
    constexpr quaternion(double w): w(w), x(0), y(0), z(0) {}

    // vector (pure imaginary)
    constexpr quaternion(double x, double y, double z): w(0), x(x), y(y), z(z) {}

    // all 4-components
    constexpr quaternion(double w, double x, double y, double z): w(w), x(x), y(y), z(z) {}

    quaternion conj() const {
        return {w, -x, -y, -z};
    }

    quaternion operator*(quaternion q) const {
        return {w*q.w-x*q.x-y*q.y-z*q.z,
                w*q.x+x*q.w+y*q.z-z*q.y,
                w*q.y-x*q.z+y*q.w+z*q.x,
                w*q.z+x*q.y-y*q.x+z*q.w};
    }

    quaternion& operator*=(quaternion q) {
        return (*this=*this*q);
    }

    quaternion& operator*=(double d) {
        w *= d;
        x *= d;
        y *= d;
        z *= d;
        return *this;
    }

    quaternion operator*(double d) const {
        quaternion q=*this;
        return q*=d;
    }

    friend quaternion operator*(double d, quaternion q) {
        return q*d;
    }

    quaternion operator+(quaternion q) const {
        return {w+q.w, x+q.x, y+q.y, z+q.z};
    }

    quaternion& operator+=(quaternion q) {
        w += q.w;
        x += q.x;
        y += q.y;
        z += q.z;
        return *this;
    }

    quaternion operator-(quaternion q) const {
        return {w-q.w, x-q.x, y-q.y, z-q.z};
    }

    quaternion& operator-=(quaternion q) {
        w -= q.w;
        x -= q.x;
        y -= q.y;
        z -= q.z;
        return *this;
    }

    double sqnorm() const {
        return w*w+x*x+y*y+z*z;
    }

    double norm() const {
        return std::sqrt(sqnorm());
    }

    // conjugation a ^ b = b a b*
    quaternion operator^(quaternion b) const {
        return b*(*this)*b.conj();
    }

    quaternion& operator^=(quaternion b) {
        return *this = b*(*this)*b.conj();
    }

    // add more as required...
};

inline quaternion rotation_x(double phi) {
    return {std::cos(phi/2), std::sin(phi/2), 0, 0};
}

inline quaternion rotation_y(double theta) {
    return {std::cos(theta/2), 0, std::sin(theta/2), 0};
}

inline quaternion rotation_z(double psi) {
    return {std::cos(psi/2), 0, 0, std::sin(psi/2)};
}
