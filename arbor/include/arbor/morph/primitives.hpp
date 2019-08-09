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

// a morphology sample point: a 3D location and radius.
struct mpoint {
    double x, y, z;  // [µm]
    double radius;   // [μm]

    friend std::ostream& operator<<(std::ostream&, const mpoint&);
};

mpoint lerp(const mpoint& a, const mpoint& b, double u);
bool is_collocated(const mpoint& a, const mpoint& b);
double distance(const mpoint& a, const mpoint& b);


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

// An unbranched cable with root, branch or terminal points at each end.
// A branch's prox sample can be either:
// (1) the root of the tree.
// (2) the spherical soma as parent.
// (3) a fork point at the end of a parent branch.
// For (1) & (2) second==prox+1, and the sequence is the half-open interval [prox:dist)
// For       (3) second!=prox, and the sequence is the half-open interval [prox,second:dist)
struct mbranch {
    // branch index at the root of the morphology
    static constexpr size_t npos = -1;

    size_t prox;                // sample index
    size_t second;              // sample index
    size_t dist;                // sample index
    size_t parent_id = npos;    // branch index

    mbranch(size_t p, size_t f, size_t d, size_t parent): prox(p), second(f), dist(d), parent_id(parent) {}
    bool is_sphere() const { return size()==1u; }
    size_t size() const { return 1 + dist-second; }
    bool has_parent() const {return parent_id!=npos;}

    friend bool operator==(const mbranch& l, const mbranch& r);
    friend std::ostream& operator<<(std::ostream& o, const mbranch& b);
};

using point_prop = std::uint8_t;
enum class point_prop_mask:
    point_prop {root=1, fork=2, terminal=4, collocated=8, sphere=16};

std::ostream& operator<<(std::ostream& o, const point_prop& p);

#define IS_PROP(prop) constexpr bool is_##prop(point_prop p) {\
    return p&static_cast<point_prop>(point_prop_mask::prop);\
}

IS_PROP(root)
IS_PROP(fork)
IS_PROP(terminal)
IS_PROP(collocated)
IS_PROP(sphere)

#define SET_PROP(prop) constexpr point_prop set_##prop(point_prop p) {\
    return p|static_cast<point_prop>(point_prop_mask::prop);\
}

SET_PROP(root)
SET_PROP(fork)
SET_PROP(terminal)
SET_PROP(collocated)
SET_PROP(sphere)

} // namespace arb
