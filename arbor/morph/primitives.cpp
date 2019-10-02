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

    return std::sqrt(dx*dx + dy*dy + dz*dz);
}

bool is_collocated(const msample& a, const msample& b) {
    return is_collocated(a.loc, b.loc);
}

double distance(const msample& a, const msample& b) {
    return distance(a.loc, b.loc);
}

bool test_invariants(const mlocation& l) {
    return (0.<=l.pos && l.pos<=1.) && l.branch!=mnpos;
}

bool test_invariants(const mcable& c) {
    return (0.<=c.prox_pos && c.prox_pos<=c.dist_pos && c.dist_pos<=1.) && c.branch!=mnpos;
}

mlocation prox_loc(const mcable& c) {
    return {c.branch, c.prox_pos};
}

mlocation dist_loc(const mcable& c) {
    return {c.branch, c.dist_pos};
}

bool test_invariants(const mcable_list& l) {
    return std::is_sorted(l.begin(), l.end())
        && l.end()==std::find_if(l.begin(), l.end(), [](const mcable& c) {return !test_invariants(c);});
}

std::ostream& operator<<(std::ostream& o, const mpoint& p) {
    return o << "(point " << p.x << " " << p.y << " " << p.z << " " << p.radius << ")";
}

std::ostream& operator<<(std::ostream& o, const msample& s) {
    return o << "(sample " << s.loc << " " << s.tag << ")";
}

std::ostream& operator<<(std::ostream& o, const mlocation& l) {
    return o << "(location " << l.branch << " " << l.pos << ")";
}

std::ostream& operator<<(std::ostream& o, const mlocation_list& l) {
    return o << "(list " << io::sepval(l, ' ') << ")";
}

std::ostream& operator<<(std::ostream& o, const mcable& c) {
    return o << "(cable " << c.branch << " " << c.prox_pos << " " << c.dist_pos << ")";
}

std::ostream& operator<<(std::ostream& o, const mcable_list& c) {
    return o << "(list " << io::sepval(c, ' ') << ")";
}

} // namespace arb
