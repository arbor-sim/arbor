#pragma once

#include <algorithm>
#include <cstdlib>
#include <ostream>
#include <vector>

#include <arbor/export.hpp>
#include <arbor/util/hash_def.hpp>
#include <arbor/util/lexcmp_def.hpp>

//
//  Types used to identify concrete locations.
//

namespace arb {

using msize_t = std::uint32_t;
constexpr msize_t mnpos = msize_t(-1);

// a morphology sample point: a 3D location and radius.
struct ARB_SYMBOL_VISIBLE mpoint {
    double x, y, z;  // [µm]
    double radius;   // [μm]
    friend std::ostream& operator<<(std::ostream&, const mpoint&);
};

ARB_DEFINE_LEXICOGRAPHIC_ORDERING(mpoint, (a.x,a.y,a.z,a.radius), (b.x,b.y,b.z,b.radius));

ARB_ARBOR_API mpoint lerp(const mpoint& a, const mpoint& b, double u);
ARB_ARBOR_API bool is_collocated(const mpoint& a, const mpoint& b);
ARB_ARBOR_API double distance(const mpoint& a, const mpoint& b);

// Indicate allowed comparison operations for classifying regions
enum class comp_op {
    lt,
    le,
    gt,
    ge
};

// Describe a cable segment between two adjacent samples.
struct ARB_SYMBOL_VISIBLE msegment {
    msize_t id;
    mpoint prox;
    mpoint dist;
    int tag;

    friend std::ostream& operator<<(std::ostream&, const msegment&);
};

ARB_DEFINE_LEXICOGRAPHIC_ORDERING(msegment, (a.id,a.prox,a.dist,a.tag),  (b.id,b.prox,b.dist,b.tag));

// Describe a specific location on a morpholology.
struct ARB_SYMBOL_VISIBLE mlocation {
    // The id of the branch.
    msize_t branch = 0;
    // The relative position on the branch ∈ [0,1].
    double pos = 0.0;

    friend std::ostream& operator<<(std::ostream&, const mlocation&);
};

// branch ≠ npos and 0 ≤ pos ≤ 1
ARB_ARBOR_API bool test_invariants(const mlocation&);
ARB_DEFINE_LEXICOGRAPHIC_ORDERING(mlocation, (a.branch,a.pos), (b.branch,b.pos));

using mlocation_list = std::vector<mlocation>;
ARB_ARBOR_API std::ostream& operator<<(std::ostream& o, const mlocation_list& l);

//// Tests whether each location in the list satisfies the invariants for a location,
//// and that the locations in the vector are ordered.
//bool test_invariants(const mlocation_list&);

// Multiset operations on location lists.
ARB_ARBOR_API mlocation_list sum(const mlocation_list&, const mlocation_list&);
ARB_ARBOR_API mlocation_list join(const mlocation_list&, const mlocation_list&);
ARB_ARBOR_API mlocation_list intersection(const mlocation_list&, const mlocation_list&);
ARB_ARBOR_API mlocation_list support(mlocation_list);

// Describe an unbranched cable in the morphology.
//
// Cables are a representation of a closed interval of a branch in a morphology.
// They may be zero-length, and fork points in the morphology may have multiple,
// equivalent zero-length cable representations.

struct ARB_SYMBOL_VISIBLE mcable {
    // The id of the branch on which the cable lies.
    msize_t branch;

    // Relative location of the end points on the branch.
    // 0 ≤ prox_pos ≤ dist_pos ≤ 1
    double prox_pos; // ∈ [0,1]
    double dist_pos; // ∈ [0,1]

    friend mlocation prox_loc(const mcable& c) {
        return {c.branch, c.prox_pos};
    }
    friend mlocation dist_loc(const mcable& c) {
        return {c.branch, c.dist_pos};
    }

    // branch ≠ npos, and 0 ≤ prox_pos ≤ dist_pos ≤ 1
    friend bool test_invariants(const mcable&);
    friend std::ostream& operator<<(std::ostream&, const mcable&);
};

ARB_DEFINE_LEXICOGRAPHIC_ORDERING(mcable, (a.branch,a.prox_pos,a.dist_pos), (b.branch,b.prox_pos,b.dist_pos));

using mcable_list = std::vector<mcable>;
ARB_ARBOR_API std::ostream& operator<<(std::ostream& o, const mcable_list& c);
// Tests whether each cable in the list satisfies the invariants for a cable,
// and that the cables in the vector are ordered.
ARB_ARBOR_API bool test_invariants(const mcable_list&);

} // namespace arb

ARB_DEFINE_HASH(arb::mcable, a.branch, a.prox_pos, a.dist_pos);
ARB_DEFINE_HASH(arb::mlocation, a.branch, a.pos);
ARB_DEFINE_HASH(arb::mpoint, a.x, a.y, a.z, a.radius);
