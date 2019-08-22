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
#include "io/sepval.hpp"
#include "morph/morphology_impl.hpp"
#include "util/span.hpp"
#include "util/strprintf.hpp"

namespace arb {

namespace impl{

//
//  mbranch implementation
//

bool operator==(const mbranch& l, const mbranch& r) {
    return l.parent_id==r.parent_id && l.index==r.index;
}

std::ostream& operator<<(std::ostream& o, const mbranch& b) {
    o <<"mbranch([" << io::csv(b.index) << "], ";
    if (b.parent_id==mnpos) o << "none)";
    else  o << b.parent_id << ")";
    return o;
}

std::vector<mbranch> branches_from_parent_index(const std::vector<msize_t>& parents,
                                                const std::vector<point_prop>& props,
                                                bool spherical_root)
{
    using util::make_span;

    const char* errstr_single_sample_root =
        "A morphology with only one sample must have a spherical root";
    const char* errstr_incomplete_cable =
        "A branch must contain at least two samples";

    if (parents.empty()) return {};

    auto nsamp = parents.size();

    // Enforce that a morphology with one sample has a spherical root.
    if (!spherical_root && nsamp==1u) {
        throw morphology_error(errstr_single_sample_root);
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
                throw morphology_error(errstr_incomplete_cable);
            }
        }
    }

    return branches;
}

// Returns false if one of the root's children has the same tag as the root.
bool root_sample_has_same_tag_as_child(const sample_tree& st) {
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
    std::vector<std::vector<msize_t>> branch_children_;

    using index_range = std::pair<const msize_t*, const msize_t*>;

    void init();

    morphology_impl(sample_tree m, bool use_spherical_root);
    morphology_impl(sample_tree m);

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
    index_range branch_indexes(msize_t b) const;

    // Range of the samples in branch b.
    const std::vector<msample>& samples() const;

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
    spherical_root_(impl::root_sample_has_same_tag_as_child(samples_))
{
    init();
}

void morphology_impl::init() {
    using util::make_span;
    using util::count_along;

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
    }
}

// The parent branch of branch b.
msize_t morphology_impl::branch_parent(msize_t b) const {
    return branch_parents_[b];
}

// The parent sample of sample i.
const std::vector<msize_t>& morphology_impl::sample_parents() const {
    return samples_.parents();
}

// The child branches of branch b.
const std::vector<msize_t>& morphology_impl::branch_children(msize_t b) const {
    return branch_children_[b];
}

// Whether the root of the morphology is spherical.
bool morphology_impl::spherical_root() const {
    return spherical_root_;
}

mindex_range morphology_impl::branch_indexes(msize_t b) const {
    const auto& idx = branches_[b].index;
    return std::make_pair(idx.data(), idx.data()+idx.size());
}

const std::vector<msample>& morphology_impl::samples() const {
    return samples_.samples();
}

msize_t morphology_impl::num_branches() const {
    return branches_.size();
}

std::ostream& operator<<(std::ostream& o, const morphology_impl& m) {
    o << "morphology: "
      << m.samples_.size() << " samples, "
      << m.num_branches() << " branches.";
    for (auto i: util::make_span(m.num_branches()))
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
    return impl_->branch_children_[b];
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

msize_t morphology::num_branches() const {
    return impl_->branches_.size();
}

std::ostream& operator<<(std::ostream& o, const morphology& m) {
    return o << *m.impl_;
}

} // namespace arb

