#pragma once

#include <memory>
#include <ostream>
#include <vector>

#include <arbor/util/lexcmp_def.hpp>
#include <arbor/morph/primitives.hpp>
#include <arbor/morph/sample_tree.hpp>

namespace arb {

struct morphology_impl;

using mindex_range = std::pair<const msize_t*, const msize_t*>;

class morphology {
    // Hold an immutable copy of the morphology implementation.
    std::shared_ptr<const morphology_impl> impl_;

public:
    morphology(sample_tree m, bool use_spherical_root);
    morphology(sample_tree m);
    morphology();

    // Empty/default-constructed morphology?
    bool empty() const;

    // Whether the root of the morphology is spherical.
    bool spherical_root() const;

    // The number of branches in the morphology.
    msize_t num_branches() const;

    // The number of samples in the morphology.
    msize_t num_samples() const;

    // The parent branch of branch b.
    msize_t branch_parent(msize_t b) const;

    // The child branches of branch b.
    const std::vector<msize_t>& branch_children(msize_t b) const;

    // Range of indexes into the sample points in branch b.
    mindex_range branch_indexes(msize_t b) const;

    // All of the samples in the morphology.
    const std::vector<msample>& samples() const;

    // The parent sample of sample i.
    const std::vector<msize_t>& sample_parents() const;

    // Point properties of samples in the morphology.
    const std::vector<point_prop>& sample_props() const;

    friend std::ostream& operator<<(std::ostream&, const morphology&);
};

} // namespace arb
