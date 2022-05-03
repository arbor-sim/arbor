#include <algorithm>
#include <ostream>

#include <arbor/math.hpp>
#include <arbor/morph/primitives.hpp>

#include "io/sepval.hpp"
#include "util/span.hpp"
#include "util/rangeutil.hpp"
#include "util/unique.hpp"

namespace arb {

namespace {

// Advance an iterator to the first value that is not equal to its current
// value, or end, whichever comes first.
template <typename T>
T next_unique(T& it, T end) {
    const auto& x = *it;
    ++it;
    while (it!=end && *it==x) ++it;
    return it;
};

// Return the number of times that the value at it is repeated. Advances the
// iterator to the first value not equal to its current value, or end,
// whichever comse first.
template <typename T>
int multiplicity(T& it, T end) {
    const auto b = it;
    return std::distance(b, next_unique(it, end));
};

} // anonymous namespace


// interpolate between two points.
ARB_ARBOR_API mpoint lerp(const mpoint& a, const mpoint& b, double u) {
    return { math::lerp(a.x, b.x, u),
             math::lerp(a.y, b.y, u),
             math::lerp(a.z, b.z, u),
             math::lerp(a.radius, b.radius, u) };
}

// test if two morphology sample points share the same location.
ARB_ARBOR_API bool is_collocated(const mpoint& a, const mpoint& b) {
    return a.x==b.x && a.y==b.y && a.z==b.z;
}

// calculate the distance between two morphology sample points.
ARB_ARBOR_API double distance(const mpoint& a, const mpoint& b) {
    double dx = a.x - b.x;
    double dy = a.y - b.y;
    double dz = a.z - b.z;

    return std::sqrt(dx*dx + dy*dy + dz*dz);
}

ARB_ARBOR_API bool test_invariants(const mlocation& l) {
    return (0.<=l.pos && l.pos<=1.) && l.branch!=mnpos;
}

ARB_ARBOR_API mlocation_list sum(const mlocation_list& lhs, const mlocation_list& rhs) {
    mlocation_list v;
    v.resize(lhs.size() + rhs.size());
    std::merge(lhs.begin(), lhs.end(), rhs.begin(), rhs.end(), v.begin());
    return v;
}

ARB_ARBOR_API mlocation_list join(const mlocation_list& lhs, const mlocation_list& rhs) {
    mlocation_list L;
    L.reserve(lhs.size()+rhs.size());

    auto l    = lhs.begin();
    auto lend = lhs.end();
    auto r    = rhs.begin();
    auto rend = rhs.end();

    auto at_end = [&]() { return l==lend || r==rend; };
    while (!at_end()) {
        auto x = (*l<*r) ? *l: *r;
        auto count = (*l<*r)? multiplicity(l, lend):
                     (*r<*l)? multiplicity(r, rend):
                     std::max(multiplicity(l, lend), multiplicity(r, rend));
        L.insert(L.end(), count, x);
    }
    L.insert(L.end(), l, lend);
    L.insert(L.end(), r, rend);

    return L;
}

ARB_ARBOR_API mlocation_list intersection(const mlocation_list& lhs, const mlocation_list& rhs) {
    mlocation_list L;
    L.reserve(lhs.size()+rhs.size());

    auto l    = lhs.begin();
    auto lend = lhs.end();
    auto r    = rhs.begin();
    auto rend = rhs.end();

    auto at_end = [&]() { return l==lend || r==rend; };
    while (!at_end()) {
        if (*l==*r) {
            auto x = *l;
            auto count = std::min(multiplicity(l, lend), multiplicity(r, rend));
            L.insert(L.end(), count, x);
        }
        else if (*l<*r) {
            next_unique(l, lend);
        }
        else {
            next_unique(r, rend);
        }
    }

    return L;
}

ARB_ARBOR_API mlocation_list support(mlocation_list L) {
    util::unique_in_place(L);
    return L;
}

ARB_ARBOR_API bool test_invariants(const mcable& c) {
    return (0.<=c.prox_pos && c.prox_pos<=c.dist_pos && c.dist_pos<=1.) && c.branch!=mnpos;
}

ARB_ARBOR_API bool test_invariants(const mcable_list& l) {
    return std::is_sorted(l.begin(), l.end())
        && l.end()==std::find_if(l.begin(), l.end(), [](const mcable& c) {return !test_invariants(c);});
}

ARB_ARBOR_API std::ostream& operator<<(std::ostream& o, const mpoint& p) {
    return o << "(point " << p.x << " " << p.y << " " << p.z << " " << p.radius << ")";
}

ARB_ARBOR_API std::ostream& operator<<(std::ostream& o, const msegment& s) {
    return o << "(segment " << s.id << " " << s.prox << " " << s.dist << " " << s.tag << ")";
}

ARB_ARBOR_API std::ostream& operator<<(std::ostream& o, const mlocation& l) {
    return o << "(location " << l.branch << " " << l.pos << ")";
}

ARB_ARBOR_API std::ostream& operator<<(std::ostream& o, const mlocation_list& l) {
    return o << "(list " << io::sepval(l, ' ') << ")";
}

ARB_ARBOR_API std::ostream& operator<<(std::ostream& o, const mcable& c) {
    return o << "(cable " << c.branch << " " << c.prox_pos << " " << c.dist_pos << ")";
}

ARB_ARBOR_API std::ostream& operator<<(std::ostream& o, const mcable_list& c) {
    return o << "(list " << io::sepval(c, ' ') << ")";
}

} // namespace arb
