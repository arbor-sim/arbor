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
#include "util/span.hpp"
#include "util/strprintf.hpp"

namespace arb {

namespace impl{

std::vector<mbranch> branches_from_parent_index(const std::vector<size_t>& parents, const std::vector<point_prop>& props, bool spherical_root) {
    const char* errstr_single_sample_root =
        "A morphology with only one sample must have a spherical root";
    const char* errstr_self_root =
        "Parent of root node must be itself, i.e. parents[0]==0";
    const char* errstr_incomplete_cable =
        "A branch must contain at least two samples";

    auto fork_or_term = [](point_prop p) {return is_terminal(p) || is_fork(p);};
    if (!parents.size()) return {};
    if (parents[0]) throw morphology_error(errstr_self_root);

    auto nsamp = parents.size();

    // Handle the single sample case.
    if (nsamp==1u) {
        if (!spherical_root) throw morphology_error(errstr_single_sample_root);
        return {mbranch{0,1,1,mbranch::npos}};
    }

    unsigned nbranch = spherical_root? 1: 0;
    nbranch += std::count_if(props.begin()+1, props.end(), fork_or_term);

    std::vector<mbranch> branches;
    branches.reserve(nbranch);

    // For lookup of branch id using the branch's distal sample as key.
    std::unordered_map<size_t, size_t> bp;
    // Tracks the id of the first sample in the current branch: start with sample 0.
    size_t first = 0;
    // Add the spherical root.
    if (spherical_root) {
        branches.push_back({0, 1, 1, mbranch::npos});
        bp[0] = 0; // connections to the root node are treated as children of the spherical root.
        ++first;
    }
    else {
        bp[0] = mbranch::npos; // first sample point is marked "none"
    }
    for (auto i: util::make_span(1, nsamp)) {
        // Fork and terminal points mark the end of a branch.
        if (fork_or_term(props[i])) {
            auto p = parents[first]; // parent sample of the first non-fork sample in this branch
            // Cable section has to be attached to a spherical root.
            if (p==0 && spherical_root) {
                branches.push_back({first, first+1, i+1, bp[p]});
                // Catch the case where a single terminal sample has a spherical root as a parent.
                if (branches.back().size()<2) throw morphology_error(errstr_incomplete_cable);
            }
            else {
                branches.push_back({p, p==first? first+1: first, i+1, bp[p]});
            }
            first = i+1;
            bp[i] = branches.size()-1;
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

    auto& samples= sample_tree_.samples();
    auto& parents= sample_tree_.parents();
    auto& props= sample_tree_.properties();
    auto nsamp = sample_tree_.size();

    // Determine whether the root is spherical by counting how many
    // times the tag assigned to the root appears.
    auto root_tag = samples[0].tag;
    auto tags = util::transform_view(samples, [](const msample& s){return s.tag;});
    spherical_root_ = 1==std::count_if(tags.begin(), tags.end(),
                                       [root_tag](auto t){return t==root_tag;});

    // Cache the fork and terminal points.
    for (auto i: make_span(nsamp)) {
        if (is_fork(props[i])) {
            fork_points_.push_back(i);
        }
        if (is_terminal(props[i])) {
            terminal_points_.push_back(i);
        }
    }

    // Generate branches.
    branches_ = impl::branches_from_parent_index(parents, props, spherical_root_);
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

struct branch_indexer {
    const size_t parent;
    const size_t first;

    branch_indexer(const mbranch& b):
        parent(b.prox), first(b.second) {}

    size_t operator()(size_t i) const {
        return first==parent? parent+i: i? first+i-1: parent;
    }
};

// Return a range of the sample points in a branch
morphology::index_range morphology::branch_sample_span(size_t b) const {
    auto it = index_counter(branch_indexer{branches_[b]});
    return std::make_pair(it, it+branches_[b].size());
}

struct sample_branch_indexer {
    branch_indexer map;
    const std::vector<msample>& samples;

    sample_branch_indexer(const mbranch& b, const std::vector<msample>& s):
        map(b), samples(s) {}

    msample operator()(size_t i) const {
        return samples[map(i)];
    }
};

morphology::sample_range morphology::branch_sample_view(size_t b) const {
    auto it = sample_counter(sample_branch_indexer(branches_[b], sample_tree_.samples()));
    return {it, it+branches_[b].size()};
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

