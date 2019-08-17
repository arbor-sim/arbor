#include <ostream>

#include <arbor/math.hpp>
#include <arbor/morph/primitives.hpp>

#include "io/sepval.hpp"
#include "util/span.hpp"
#include "util/rangeutil.hpp"

namespace arb {

// interpolate between two points.
mpoint lerp(const mpoint& a, const mpoint& b, double u) {
    return { math::lerp(a.x, b.x, u),
             math::lerp(a.y, b.y, u),
             math::lerp(a.z, b.z, u),
             math::lerp(a.radius, b.radius, u) };
}

// test if two morphology sample points share the same location.
bool is_collocated(const mpoint& a, const mpoint& b) {
    return a.x==b.x && a.y==b.y && a.z==b.z;
}

// calculate the distance between two morphology sample points.
double distance(const mpoint& a, const mpoint& b) {
    double dx = a.x - b.x;
    double dy = a.y - b.y;
    double dz = a.z - b.z;

    return std::sqrt(dx*dx + dy*dy * dz*dz);
}

bool is_collocated(const msample& a, const msample& b) {
    return is_collocated(a.loc, b.loc);
}

double distance(const msample& a, const msample& b) {
    return distance(a.loc, b.loc);
}

std::ostream& operator<<(std::ostream& o, const mpoint& p) {
    return o << "mpoint(" << p.x << "," << p.y << "," << p.z << "," << p.radius << ")";
}

std::ostream& operator<<(std::ostream& o, const msample& s) {
    return o << "msample(" << s.loc << ", " << s.tag << ")";
}

} // namespace arb
