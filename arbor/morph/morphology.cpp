#include <cmath>
#include <iostream>
#include <unordered_map>
#include <utility>

#include <arbor/math.hpp>
#include <arbor/morph/error.hpp>
#include <arbor/morph/morphology.hpp>
#include <arbor/morph/sample_tree.hpp>
#include <arbor/morph/primitives.hpp>

#include "algorithms.hpp"
#include "util/span.hpp"
#include "util/strprintf.hpp"

namespace arb {

namespace impl{

std::vector<mbranch> branches_from_parent_index(const std::vector<size_t>& parents, const std::vector<point_prop>& props, bool spherical_root) {
    using util::make_span;

    const char* errstr_single_sample_root =
        "A morphology with only one sample must have a spherical root";
    const char* errstr_incomplete_cable =
        "A branch must contain at least two samples";

    auto npos = mbranch::npos;

    if (!parents.size()) return {};

    auto nsamp = parents.size();

    // Enforce that a morphology with one sample has a spherical root.
    if (!spherical_root && nsamp==1u) {
        throw morphology_error(errstr_single_sample_root);
    }

    std::vector<int> bids(nsamp);
    int nbranches = spherical_root? 1: 0;
    for (auto i: make_span(1, nsamp)) {
        size_t p = parents[i];
        bool first = is_root(props[p]) || is_fork(props[p]);
        bids[i] = first? nbranches++: bids[p];
    }

    std::vector<mbranch> branches(nbranches);
    for (auto i: make_span(nsamp)) {
        auto p = parents[i];
        auto& branch = branches[bids[i]];

        // Determine the id of the parent branch if this is the first sample in
        // the branch, and include the fork/root point at the end of the parent
        // branch where applicable.
        // This is icky, but of all the solutions, this is the cleanest (that I
        // could find).
        if (!branch.size()) {
            // A branch has null root if either:
            //      ∃ a spherical root and branch id is 0
            //      ∄ a spherical root and parent id is 0
            auto null_root = spherical_root? !i: !p;
            branch.parent_id = null_root? npos: bids[parents[i]];

            // Add the first sample to a branch that is attached to
            // non-spherical root if the branch is not branch 0.
            if ((null_root && i) || is_fork(props[p])) {
                branch.index.push_back(p);
            }
        }
        branch.index.push_back(i);
    }

    // Enforce that all cable branches that are potentially connected to a spherical
    // root contain at least two samples.
    if (spherical_root) {
        for (auto i: make_span(1, nbranches)) { // skip the root.
            if (branches[i].size()<2u) {
                throw morphology_error(errstr_incomplete_cable);
            }
        }
    }

    return branches;
}

} // namespace impl

//
// morphology implementation
//

morphology::morphology(sample_tree m):
    sample_tree_(std::move(m))
{
    using util::make_span;
    using util::count_along;

    auto nsamp = sample_tree_.size();

    // Treat the root sample as a sphere if it does not have the same tag as
    // any of its children.
    spherical_root_ = sample_tree_.single_root_tag();

    // Cache the fork and terminal points.
    auto& props = sample_tree_.properties();
    for (auto i: make_span(nsamp)) {
        if (is_fork(props[i])) {
            fork_points_.push_back(i);
        }
        if (is_terminal(props[i])) {
            terminal_points_.push_back(i);
        }
    }

    // Generate branches.
    branches_ = impl::branches_from_parent_index(sample_tree_.parents(), props, spherical_root_);
    auto nbranch = branches_.size();

    // Generate branch tree.
    branch_children_.resize(nbranch);
    branch_parents_.reserve(nbranch);
    for (auto i: make_span(nbranch)) {
        auto id = branches_[i].parent_id;
        branch_parents_.push_back(id);
        if (id!=mbranch::npos) {
            branch_children_[id].push_back(i);
        }
    }
}

// The parent branch of branch b.
size_t morphology::branch_parent(size_t b) const {
    return branch_parents_[b];
}

// The child branches of branch b.
const std::vector<size_t>& morphology::branch_children(size_t b) const {
    return branch_children_[b];
}

// Whether the root of the morphology is spherical.
bool morphology::spherical_root() const {
    return spherical_root_;
}

morphology::index_range morphology::branch_sample_span(size_t b) const {
    const auto& idx = branches_[b].index;
    return std::make_pair(idx.data(), idx.data()+idx.size());
}

const std::vector<msample>& morphology::samples() const {
    return sample_tree_.samples();
}

size_t morphology::num_branches() const {
    return branches_.size();
}

const std::vector<size_t>& morphology::fork_points() const {
    return fork_points_;
}

const std::vector<size_t>& morphology::terminal_points() const {
    return terminal_points_;
}

std::ostream& operator<<(std::ostream& o, const morphology& m) {
    o << "morphology: "
      << m.sample_tree_.size() << " samples, "
      << m.num_branches() << " branches.";
    for (auto i: util::make_span(m.num_branches()))
        o << "\n  branch " << i << ": " << m.branches_[i];

    return o;
}

} // namespace arb

