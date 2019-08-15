#pragma once

#include <ostream>
#include <vector>

#include <arbor/util/lexcmp_def.hpp>
#include <arbor/morph/primitives.hpp>
#include <arbor/morph/sample_tree.hpp>

namespace arb {

class morphology {
    // The sample tree of sample points and their parent-child relationships.
    sample_tree sample_tree_;

    // Indicates whether the soma is a sphere.
    bool spherical_root_;

    // Branch state.
    std::vector<mbranch> branches_;
    std::vector<size_t> branch_parents_;
    std::vector<std::vector<size_t>> branch_children_;

    // Meta data about sample point properties.
    std::vector<size_t> fork_points_;
    std::vector<size_t> terminal_points_;
    std::vector<point_prop> point_props_;

    using index_range = std::pair<const size_t*, const size_t*>;

    void init();

public:
    morphology(sample_tree m, bool use_spherical_root);
    morphology(sample_tree m);

    // Whether the root of the morphology is spherical.
    bool spherical_root() const;

    // The number of branches in the morphology.
    size_t num_branches() const;

    // List the ids of fork points in the morphology.
    const std::vector<size_t>& fork_points() const;

    // List the ids of terminal points in the morphology.
    const std::vector<size_t>& terminal_points() const;

    // The parent sample of sample i.
    const std::vector<size_t>& sample_parents() const;

    // The parent branch of branch b.
    size_t branch_parent(size_t b) const;

    // The child branches of branch b.
    const std::vector<size_t>& branch_children(size_t b) const;

    // Range of indexes into the sample points in branch b.
    index_range branch_sample_span(size_t b) const;

    // Range of the samples in branch b.
    const std::vector<msample>& samples() const;

    friend std::ostream& operator<<(std::ostream&, const morphology&);
};

} // namespace arb
