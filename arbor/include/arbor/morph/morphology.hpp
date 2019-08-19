#pragma once

#include <ostream>
#include <vector>

#include <arbor/util/lexcmp_def.hpp>
#include <arbor/morph/primitives.hpp>
#include <arbor/morph/sample_tree.hpp>

namespace arb {

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

class morphology {
    // The sample tree of sample points and their parent-child relationships.
    sample_tree samples_;

    // Indicates whether the soma is a sphere.
    bool spherical_root_ = false;

    // Branch state.
    std::vector<mbranch> branches_;
    std::vector<msize_t> branch_parents_;
    std::vector<std::vector<msize_t>> branch_children_;

    using index_range = std::pair<const msize_t*, const msize_t*>;

    void init();

public:
    morphology(sample_tree m, bool use_spherical_root);
    morphology(sample_tree m);

    // Whether the root of the morphology is spherical.
    bool spherical_root() const;

    // The number of branches in the morphology.
    msize_t num_branches() const;

    // The parent sample of sample i.
    const std::vector<msize_t>& sample_parents() const;

    // The parent branch of branch b.
    msize_t branch_parent(msize_t b) const;

    // The child branches of branch b.
    const std::vector<msize_t>& branch_children(msize_t b) const;

    // Range of indexes into the sample points in branch b.
    index_range branch_sample_span(msize_t b) const;

    // Range of the samples in branch b.
    const std::vector<msample>& samples() const;

    friend std::ostream& operator<<(std::ostream&, const morphology&);
};

} // namespace arb
