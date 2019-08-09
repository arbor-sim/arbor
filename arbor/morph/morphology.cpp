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

// Generate a point_kind value for each sample point in a tree described
// only by parent indexes. The point_kind values can be used to quickly look
// up whether a point is root, fork or terminal.
// Implemented out of line from the point_props function below to facilitate
// unit testing of this step, and of branches_from_parent_index.
std::vector<point_prop> mark_branch_props(const std::vector<size_t>& parents) {
    auto n = parents.size();
    std::vector<int> counts(n);
    for (auto p: parents) {
        ++counts[p];
    }
    std::vector<point_prop> props(n, 0);
    props[0] = set_root(props[0]);
    for (auto i: util::make_span(1, n)) {
        auto c = counts[i];
        if (c==0) props[i] = set_terminal(props[i]);
        if (c>1)  props[i] = set_fork(props[i]);
    }
    return props;
}

// Generate a point properties each sample point in a tree described.
// The point_kind values can be used to quickly look up
// up whether a point is
//  * a terminal point (has no children)
//  * a fork pint (has more than one child)
//  * the root (has no parent)
//  * collocated (has the same location (x,y,z) as its parent)
std::vector<point_prop> point_props(const sample_tree& sm) {
    auto& S = sm.samples();
    auto& P = sm.parents();
    auto n = P.size();

    auto props = mark_branch_props(P);

    for (auto i: util::make_span(1, n)) {
        if (is_collocated(S[i], S[P[i]])) {
            props[i] = set_collocated(props[i]);
        }
    }

    return props;
}

std::vector<mbranch> branches_from_parent_index(const std::vector<size_t>& parents, bool spherical_root) {
    const char* errstr_single_sample_soma =
        "A morphology with only one sample must have a spherical root";
    const char* errstr_self_root =
        "Parent of root node must be itself, i.e. parents[0]==0";
    const char* errstr_incomplete_cable =
        "A cable segment requires at least two non-collocated samples";

    auto fork_or_term = [](point_prop p) {return is_terminal(p) || is_fork(p);};
    if (!parents.size()) return {};
    if (parents[0]) throw morphology_error(errstr_self_root);

    auto nsamp = parents.size();

    // Handle the single sample case.
    if (nsamp==1u) {
        if (!spherical_root) throw morphology_error(errstr_single_sample_soma);
        return {mbranch{0,1,1,mbranch::npos}};
    }

    auto props = mark_branch_props(parents);
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
            if (p==0 && spherical_root) {
                branches.push_back({first, first+1, i+1, bp[p]});
            }
            else {
                branches.push_back({p, p==first? first+1: first, i+1, bp[p]});
            }
            first = i+1;
            bp[i] = branches.size()-1;
            if (branches.back().size()<2) throw morphology_error(errstr_incomplete_cable);
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

    auto& ss= sample_tree_.samples();
    auto& pp= sample_tree_.parents();
    auto nsamp = ss.size();

    // Determine whether the root is spherical by counting how many
    // times the tag assigned to the root appears.
    auto soma_tag = ss[0].tag;
    auto tags = util::transform_view(ss, [](const msample& s){return s.tag;});
    spherical_root_ = 1==std::count_if(tags.begin(), tags.end(),
                                       [soma_tag](auto t){return t==soma_tag;});

    // Find properties of sample points.
    point_props_ = impl::point_props(sample_tree_);
    for (auto i: count_along(pp)) {
        const auto prop = point_props_[i];
        if (is_fork(prop)) {
            fork_points_.push_back(i);
        }
        if (is_terminal(prop)) {
            terminal_points_.push_back(i);
        }
    }

    // Generate branches.
    branches_ = impl::branches_from_parent_index(pp, spherical_root_);
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

    // Generate segments.
    segments_.reserve(nsamp);
    segment_part_.reserve(nbranch+1);
    segment_part_.push_back(0);
    for (auto i:make_span(0, nbranch)) {
        auto b = util::make_range(branch_sample_span(i));
        if (b.size()==1) { // spherical segment
            segments_.push_back(segment{0,0});
        }
        else {             // line segment
            int nseg = 0;
            bool prev_collocated = false; // records if the last sample point was collocated
            for (auto j:make_span(1, b.size())) {
                auto l = b[j-1];
                auto r = b[j];
                bool this_collocated = is_collocated(point_props_[r]);
                if (prev_collocated && this_collocated) {
                    throw morphology_error(
                        "Morphology has more than two points at the same location.");
                }
                if (!this_collocated) {
                    segments_.push_back({l, r});
                    ++nseg;
                }
                prev_collocated = this_collocated;
            }
            if (nseg==0)
                throw morphology_error(
                    "A branch must have at least one segment. This error is caused by collocated points");
            if (prev_collocated && is_terminal(point_props_[b.back()]))
                throw morphology_error(
                    "A terminal point can't be collocated'");
        }
        segment_part_.push_back(segment_part_.size());
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

size_t morphology::num_segments() const {
    return segments_.size();
}

const std::vector<size_t>& morphology::fork_points() const {
    return fork_points_;
}

const std::vector<size_t>& morphology::terminal_points() const {
    return terminal_points_;
}

size_t morphology::seg2bid(size_t i) const {
    // Return branch that segment i lies on
    return std::distance(segment_part_.begin(),
                         std::lower_bound(segment_part_.begin(), segment_part_.end(), i));
}

std::ostream& operator<<(std::ostream& o, const morphology& m) {
    auto print_seg = [](const morphology::segment& s) {return util::pprintf("[{},{}]", s.prox, s.dist);};
    o << "morphology: "
      << m.sample_tree_.size() << " samples, "
      << m.num_branches() << " branches, "
      << m.num_segments() << " segments.";
    for (auto i: util::make_span(m.num_branches()))
        o << "\n  branch " << i << ": " << m.branches_[i];
    o << "\n  segments: [" << io::csv(util::transform_view(m.segments_, print_seg)) << "]";

    return o;
}

} // namespace arb

