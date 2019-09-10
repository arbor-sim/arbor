#pragma once

/*
 * Definitions and prototypes used in the morphology implementation.
 * This header is used to simplify unit testing of the morphology implementation.
 */

#include <cmath>
#include <vector>

#include <arbor/morph/primitives.hpp>

namespace arb {
namespace impl{

// An unbranched cable segment that has root, terminal or fork point at each end.
struct mbranch {
    std::vector<msize_t> index;  // sample index
    msize_t parent_id = mnpos;   // branch index

    mbranch() = default;
    mbranch(std::vector<msize_t> idx, msize_t parent):
        index(std::move(idx)), parent_id(parent) {}

    bool is_sphere()  const { return size()==1u; }
    msize_t size()    const { return index.size(); }
    bool has_parent() const { return parent_id!=mnpos;}

    friend bool operator==(const mbranch& l, const mbranch& r);
    friend std::ostream& operator<<(std::ostream& o, const mbranch& b);
};

std::vector<mbranch> branches_from_parent_index(const std::vector<msize_t>& parents,
                                                const std::vector<point_prop>& props,
                                                bool spherical_root);

} // namespace impl
} // namespace arb
