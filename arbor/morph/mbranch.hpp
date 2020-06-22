#pragma once

/*
 * Definitions and prototypes used in the morphology implementation.
 * This header is used to simplify unit testing of the morphology implementation.
 */

#include <cmath>
#include <utility>
#include <vector>

#include <arbor/morph/primitives.hpp>

namespace arb {
namespace impl{

// An unbranched cable segment that has root, terminal or fork point at each end.
struct mbranch {
    std::vector<msegment> segments;  // explicit segments.
    msize_t parent_id = mnpos;       // branch index.

    mbranch() = default;
    mbranch(std::vector<msegment> segs, msize_t parent):
        segments(std::move(segs)), parent_id(parent) {}

    msize_t size()    const { return segments.size()+1; }
    bool has_parent() const { return parent_id!=mnpos;}

    friend std::ostream& operator<<(std::ostream& o, const mbranch& b);
};

} // namespace impl
} // namespace arb
