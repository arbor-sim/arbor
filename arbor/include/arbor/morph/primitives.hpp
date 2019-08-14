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

// An unbranched cable segment that has root, terminal or fork point at each end.
struct mbranch {
    // branch index at the root of the morphology
    static constexpr size_t npos = -1;

    std::vector<size_t> index;  // sample index
    size_t parent_id = npos;    // branch index

    mbranch() = default;
    mbranch(std::vector<size_t> idx, size_t parent):
        index(std::move(idx)), parent_id(parent) {}

    bool is_sphere()  const { return size()==1u; }
    size_t size()     const { return index.size(); }
    bool has_parent() const {return parent_id!=npos;}

    friend bool operator==(const mbranch& l, const mbranch& r);
    friend std::ostream& operator<<(std::ostream& o, const mbranch& b);
};

using point_prop = std::uint8_t;
constexpr point_prop point_prop_mask_none = 0;
constexpr point_prop point_prop_mask_root = 1;
constexpr point_prop point_prop_mask_fork = 2;
constexpr point_prop point_prop_mask_terminal = 4;
constexpr point_prop point_prop_mask_collocated = 8;

std::ostream& operator<<(std::ostream& o, const point_prop& p);

#define PROP(prop) \
constexpr bool is_##prop(point_prop p) {\
    return p&point_prop_mask_##prop;\
} \
inline void set_##prop(point_prop& p) {\
    p |= point_prop_mask_##prop;\
} \
inline void unset_##prop(point_prop& p) {\
    p &= ~point_prop_mask_##prop;\
}

PROP(root)
PROP(fork)
PROP(terminal)
PROP(collocated)

} // namespace arb
