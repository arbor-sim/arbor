#pragma once

#include <arbor/export.hpp>
#include <arbor/math.hpp>

namespace arb {
// Represent a 3-d isometry as a rotation (via quaterion q_)
// and a subsequent translation (tx_, ty_, tz_).
struct ARB_ARBOR_API isometry {

    isometry() = default;

    // Composition: rotations are interpreted as applied to intrinsic
    // coordinates (composed by right multiplication), and translations
    // as applied to absolute coordinates (composed by addition).

    friend isometry operator*(const isometry& a, const isometry& b) {
        return isometry(b.q_*a.q_, a.tx_+b.tx_, a.ty_+b.ty_, a.tz_+b.tz_);
    }

    template <typename PointLike>
    PointLike apply(PointLike p) const {
        auto w = quaternion(p.x, p.y, p.z)^q_;
        p.x = w.x+tx_;
        p.y = w.y+ty_;
        p.z = w.z+tz_;
        return p;
    }

private:
    using quaternion = arb::math::quaternion;
    quaternion q_{1, 0, 0, 0};
    double tx_ = 0, ty_ = 0, tz_ = 0;

    isometry(quaternion q, double tx, double ty, double tz):
        q_(std::move(q)), tx_(tx), ty_(ty), tz_(tz) {}

public:
    static isometry translate(double x, double y, double z) {
        return isometry(quaternion{1, 0, 0, 0}, x, y, z);
    }

    template <typename PointLike>
    static isometry translate(const PointLike& p) {
        return translate(p.x, p.y, p.z);
    }

    // Rotate about axis in direction (ax, ay, az) by theta radians.
    static isometry rotate(double theta, double ax, double ay, double az) {
        double norm = std::sqrt(ax*ax+ay*ay+az*az);
        double vscale = std::sin(theta/2)/norm;

        return isometry(quaternion{std::cos(theta/2), ax*vscale, ay*vscale, az*vscale}, 0, 0, 0);
    }

    template <typename PointLike>
    static isometry rotate(double theta, const PointLike& p) {
        return rotate(theta, p.x, p.y, p.z);
    }
};

} // arb
