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
