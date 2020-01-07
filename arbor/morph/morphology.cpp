#include <iostream>
#include <stack>
#include <utility>

#include <arbor/morph/morphexcept.hpp>
#include <arbor/morph/morphology.hpp>
#include <arbor/morph/sample_tree.hpp>
#include <arbor/morph/primitives.hpp>

#include "morph/mbranch.hpp"
#include "util/rangeutil.hpp"
#include "util/span.hpp"

using arb::util::make_span;

namespace arb {
namespace impl{

std::vector<mbranch> branches_from_parent_index(const std::vector<msize_t>& parents,
                                                const std::vector<point_prop>& props,
                                                bool spherical_root)
{
    auto nsamp = parents.size();
    if (!nsamp) return {};

    // Enforce that a morphology with one sample has a spherical root.
    if (!spherical_root && nsamp==1u) {
        throw incomplete_branch(0);
    }

    std::vector<int> bids(nsamp);
    int nbranches = spherical_root? 1: 0;
    for (auto i: make_span(1, nsamp)) {
        auto p = parents[i];
        bool first = is_root(props[p]) || is_fork(props[p]);
        bids[i] = first? nbranches++: bids[p];
    }

    std::vector<mbranch> branches(nbranches);
    for (auto i: make_span(nsamp)) {
        auto p = parents[i];
        auto& branch = branches[bids[i]];
        if (!branch.size()) {
            bool null_root = spherical_root? p==mnpos: p==mnpos||p==0;
            branch.parent_id = null_root? mnpos: bids[p];

            // Add the distal sample from the parent branch if the parent is not
            // a spherical root or mnpos.
            if (p!=mnpos && !(p==0 && spherical_root)) {
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
                throw incomplete_branch(i);
            }
        }
    }

    return branches;
}

// Returns false if one of the root's children has the same tag as the root.
bool root_sample_tag_differs_from_children(const sample_tree& st) {
    if (st.empty()) return false;
    auto& P = st.parents();
    auto& S = st.samples();
    auto root_tag = S.front().tag;
    for (auto i: util::make_span(1, st.size())) {
        if (0u==P[i] && S[i].tag==root_tag) {
            return false;
        }
    }
    return true;
}

} // namespace impl

//
//  morphology_impl definition and implementation
//

struct morphology_impl {
    // The sample tree of sample points and their parent-child relationships.
    sample_tree samples_;

    // Indicates whether the soma is a sphere.
    bool spherical_root_ = false;

    // Branch state.
    std::vector<impl::mbranch> branches_;
    std::vector<msize_t> branch_parents_;
    std::vector<msize_t> root_children_;
    std::vector<msize_t> terminal_branches_;
    std::vector<std::vector<msize_t>> branch_children_;

    morphology_impl(sample_tree m, bool use_spherical_root);
    morphology_impl(sample_tree m);

    void init();

    friend std::ostream& operator<<(std::ostream&, const morphology_impl&);
};

morphology_impl::morphology_impl(sample_tree m, bool use_spherical_root):
    samples_(std::move(m)),
    spherical_root_(use_spherical_root)
{
    init();
}

morphology_impl::morphology_impl(sample_tree m):
    samples_(std::move(m)),
    spherical_root_(impl::root_sample_tag_differs_from_children(samples_))
{
    init();
}

void morphology_impl::init() {
    auto nsamp = samples_.size();
    if (!nsamp) return;

    // Generate branches.
    branches_ = impl::branches_from_parent_index(samples_.parents(), samples_.properties(), spherical_root_);
    auto nbranch = branches_.size();

    // Generate branch tree.
    branch_children_.resize(nbranch);
    branch_parents_.reserve(nbranch);
    for (auto i: make_span(nbranch)) {
        auto id = branches_[i].parent_id;
        branch_parents_.push_back(id);
        if (id!=mnpos) {
            branch_children_[id].push_back(i);
        }
        else {
            root_children_.push_back(i);
        }
    }

    // Collect terminal branches.
    terminal_branches_.reserve(nbranch);
    for (auto i: make_span(nbranch)) {
        if (branch_children_[i].empty()) {
            terminal_branches_.push_back(i);
        }
    }
    terminal_branches_.shrink_to_fit();
}

std::ostream& operator<<(std::ostream& o, const morphology_impl& m) {
    o << "morphology: "
      << m.samples_.size() << " samples, "
      << m.branches_.size() << " branches.";
    for (auto i: util::make_span(m.branches_.size()))
        o << "\n  branch " << i << ": " << m.branches_[i];

    return o;
}

//
// morphology implementation
//

morphology::morphology(sample_tree m, bool use_spherical_root):
    impl_(std::make_shared<const morphology_impl>(std::move(m), use_spherical_root))
{}

morphology::morphology(sample_tree m):
    impl_(std::make_shared<const morphology_impl>(std::move(m)))
{}

morphology::morphology():
    morphology(sample_tree())
{}

bool morphology::empty() const {
    return impl_->branches_.empty();
}

// The parent branch of branch b.
msize_t morphology::branch_parent(msize_t b) const {
    return impl_->branch_parents_[b];
}

// The parent sample of sample i.
const std::vector<msize_t>& morphology::sample_parents() const {
    return impl_->samples_.parents();
}

// The child branches of branch b.
const std::vector<msize_t>& morphology::branch_children(msize_t b) const {
    return b==mnpos? impl_->root_children_: impl_->branch_children_[b];
}

const std::vector<msize_t>& morphology::terminal_branches() const {
    return impl_->terminal_branches_;
}

// Whether the root of the morphology is spherical.
bool morphology::spherical_root() const {
    return impl_->spherical_root_;
}

mindex_range morphology::branch_indexes(msize_t b) const {
    const auto& idx = impl_->branches_[b].index;
    return std::make_pair(idx.data(), idx.data()+idx.size());
}

const std::vector<msample>& morphology::samples() const {
    return impl_->samples_.samples();
}

// Point properties of samples in the morphology.
const std::vector<point_prop>& morphology::sample_props() const {
    return impl_->samples_.properties();
}

msize_t morphology::num_samples() const {
    return impl_->samples_.size();
}

msize_t morphology::num_branches() const {
    return impl_->branches_.size();
}

std::ostream& operator<<(std::ostream& o, const morphology& m) {
    return o << *m.impl_;
}

// Utilities.

mlocation_list minset(const morphology& m, const mlocation_list& in) {
    mlocation_list L;

    std::stack<msize_t> stack;

    // All root branches must be searched.
    for (auto c: m.branch_children(mnpos)) {
        stack.push(c);
    }

    // Depth-first traversal of the branch tree.
    while (!stack.empty()) {
        auto branch = stack.top();
        stack.pop();

        // Search for a location on branch.
        auto it = std::lower_bound(in.begin(), in.end(), mlocation{branch, 0});

        // If found, insert to the minset and skip the rest of this sub-tree.
        if (it!=in.end() && it->branch==branch) {
            L.push_back(*it);
            continue;
        }

        // No location on this branch, so continue searching in this sub-tree.
        for (auto c: m.branch_children(branch)) {
            stack.push(c);
        }
    }

    util::sort(L);
    return L;
}

mlocation canonical(const morphology& m, mlocation loc) {
    if (loc.pos==0) {
        msize_t parent = m.branch_parent(loc.branch);
        return parent==mnpos? mlocation{0, 0.}: mlocation{parent, 1.};
    }
    return loc;
}

} // namespace arb

