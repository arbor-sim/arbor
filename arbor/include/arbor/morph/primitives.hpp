#pragma once

#include <algorithm>
#include <cstdlib>
#include <ostream>
#include <vector>

#include <arbor/util/lexcmp_def.hpp>

//
//  Types used to identify concrete locations.
//

namespace arb {

using msize_t = std::uint32_t;
constexpr msize_t mnpos = msize_t(-1);

// a morphology sample point: a 3D location and radius.
struct mpoint {
    double x, y, z;  // [µm]
    double radius;   // [μm]

    friend std::ostream& operator<<(std::ostream&, const mpoint&);
};

mpoint lerp(const mpoint& a, const mpoint& b, double u);
bool is_collocated(const mpoint& a, const mpoint& b);
double distance(const mpoint& a, const mpoint& b);

// Indicate allowed comparison operations for classifying regions
enum class comp_op {
    lt,
    le,
    gt,
    ge
};

// A morphology sample consists of a location and an integer tag.
// When loaded from an SWC file, the tag will correspond to the SWC label,
// which are standardised as follows:
//  1 - soma
//  2 - axon
//  3 - (basal) dendrite
//  4 - apical dendrite
// http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html
// However, any positive integer tag can be provided and labelled dynamically.

struct msample {
    mpoint loc;
    int tag;

    friend std::ostream& operator<<(std::ostream&, const msample&);
};

bool is_collocated(const msample& a, const msample& b);
double distance(const msample& a, const msample& b);

// Describe a specific location on a morpholology.
struct mlocation {
    // The id of the branch.
    msize_t branch;
    // The relative position on the branch ∈ [0,1].
    double pos;

    friend std::ostream& operator<<(std::ostream&, const mlocation&);
};

// branch ≠ npos and 0 ≤ pos ≤ 1
bool test_invariants(const mlocation&);
ARB_DEFINE_LEXICOGRAPHIC_ORDERING(mlocation, (a.branch,a.pos), (b.branch,b.pos));

using mlocation_list = std::vector<mlocation>;
std::ostream& operator<<(std::ostream& o, const mlocation_list& l);

// Tests whether each location in the list satisfies the invariants for a location,
// and that the locations in the vector are ordered.
bool test_invariants(const mlocation_list&);

// Multiset operations on location lists.
mlocation_list sum(const mlocation_list&, const mlocation_list&);
mlocation_list join(const mlocation_list&, const mlocation_list&);
mlocation_list intersection(const mlocation_list&, const mlocation_list&);

// Describe an unbranched cable in the morphology.
//
// Cables are a representation of a closed interval of a branch in a morphology.
// They may be zero-length, and fork points in the morphology may have multiple,
// equivalent zero-length cable representations.

struct mcable {
    // The id of the branch on which the cable lies.
    msize_t branch;

    // Relative location of the end points on the branch.
    // 0 ≤ prox_pos ≤ dist_pos ≤ 1
    double prox_pos; // ∈ [0,1]
    double dist_pos; // ∈ [0,1]

    friend mlocation prox_loc(const mcable&);
    friend mlocation dist_loc(const mcable&);

    // branch ≠ npos, and 0 ≤ prox_pos ≤ dist_pos ≤ 1
    friend bool test_invariants(const mcable&);
    friend std::ostream& operator<<(std::ostream&, const mcable&);
};

ARB_DEFINE_LEXICOGRAPHIC_ORDERING(mcable, (a.branch,a.prox_pos,a.dist_pos), (b.branch,b.prox_pos,b.dist_pos));

using mcable_list = std::vector<mcable>;
std::ostream& operator<<(std::ostream& o, const mcable_list& c);
// Tests whether each cable in the list satisfies the invariants for a cable,
// and that the cables in the vector are ordered.
bool test_invariants(const mcable_list&);

using point_prop = std::uint8_t;
enum point_prop_mask: point_prop {
    point_prop_mask_none = 0,
    point_prop_mask_root = 1,
    point_prop_mask_fork = 2,
    point_prop_mask_terminal = 4,
    point_prop_mask_collocated = 8
};

#define ARB_PROP(prop) \
constexpr bool is_##prop(point_prop p) {\
    return p&point_prop_mask_##prop;\
} \
inline void set_##prop(point_prop& p) {\
    p |= point_prop_mask_##prop;\
} \
inline void unset_##prop(point_prop& p) {\
    p &= ~point_prop_mask_##prop;\
}

ARB_PROP(root)
ARB_PROP(fork)
ARB_PROP(terminal)
ARB_PROP(collocated)

} // namespace arb
