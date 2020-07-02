#include <iostream>
#include <stack>
#include <unordered_map>
#include <utility>

#include <arbor/morph/morphexcept.hpp>
#include <arbor/morph/morphology.hpp>
#include <arbor/morph/sample_tree.hpp>
#include <arbor/morph/primitives.hpp>

#include "morph/mbranch.hpp"
#include "util/mergeview.hpp"
#include "util/rangeutil.hpp"
#include "util/span.hpp"

using arb::util::make_span;

namespace arb {
namespace impl {

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

mlocation_list maxset(const morphology& m, const mlocation_list& in_) {
    mlocation_list L;

    // Sort the input in reverse order, so that more distal locations
    // come first.
    mlocation_list in = in_;
    util::sort(in, [](const auto& l, const auto& r) {return r<l;});

    // List of branches that have had a more distal location found.
    std::unordered_set<msize_t> br;
    for (auto loc: in) {
        auto b = loc.branch;

        // A child of this branch has already been visited: a more distal
        // location has already been found, so we can skip.
        if (br.count(b)) continue;

        // Add the location to the maxset.
        L.push_back(loc);

        // Mark the branch and its parents.
        while (b!=mnpos) {
            br.insert(b);
            b = m.branch_parent(b);
        }
    }

    std::reverse(L.begin(), L.end());
    return L;
}

mlocation canonical(const morphology& m, mlocation loc) {
    if (loc.pos==0) {
        msize_t parent = m.branch_parent(loc.branch);
        return parent==mnpos? mlocation{0, 0.}: mlocation{parent, 1.};
    }
    return loc;
}

// Merge overlapping cables so that none of the cables in the output overlap.
// Used by the mextent constructor.
mcable_list build_mextent_cables(const mcable_list& cables) {
    arb_assert(arb::test_invariants(cables));

    mcable_list cs;
    for (auto& c: cables) {
        mcable* prev = cs.empty()? nullptr: &cs.back();

        if (prev && prev->branch==c.branch && prev->dist_pos>=c.prox_pos) {
            prev->dist_pos = std::max(prev->dist_pos, c.dist_pos);
        }
        else {
            cs.push_back(c);
        }
    }

    return cs;
}

mextent::mextent(const mcable_list& cables):
    cables_(build_mextent_cables(cables)) {}

bool mextent::test_invariants() const {
    // Checks for:
    //   * validity of each cables.
    //   * sortedness of cables in list.
    if (!arb::test_invariants(cables_)) return false;

    // Check for intersections:
    for (auto i: util::count_along(cables_)) {
        if (!i) continue;

        const auto& c = cables_[i];
        const auto& p = cables_[i-1];
        if (p.branch==c.branch && p.dist_pos>=c.prox_pos) return false;
    }

    return true;
}

bool mextent::test_invariants(const morphology& m) const {
    // Check for sortedness, intersections:
    if (!test_invariants()) return false;

    // Too many branches?
    if (!empty() && cables_.back().branch>=m.num_branches()) return false;

    return true;
}

bool mextent::intersects(const mcable_list& a) const {
    arb_assert(arb::test_invariants(a));

    // Early exit?
    if (empty() || a.empty() ||
        cables_.front().branch>a.back().branch ||
        cables_.back().branch<a.front().branch)
    {
        return false;
    }

    auto from = cables_.begin();
    for (auto& c: a) {
        auto i = std::lower_bound(from, cables_.end(), c);

        if (i!=cables_.end() && i->branch==c.branch) {
            arb_assert(i->prox_pos>=c.prox_pos);
            if (i->prox_pos<=c.dist_pos) return true;
        }

        if (i!=cables_.begin()) {
           auto j = std::prev(i);
           if (j->branch==c.branch) {
               arb_assert(j->prox_pos<c.prox_pos);
               if (j->dist_pos>=c.prox_pos) return true;
           }
        }

        from = i;
    }

    return false;
}

mextent intersect(const mextent& a, const mextent& b) {
    auto precedes = [](mcable x, mcable y) {
        return x.branch<y.branch || (x.branch==y.branch && x.dist_pos<y.prox_pos);
    };

    mextent m;
    auto ai = a.cables().begin();
    auto ae = a.cables().end();
    auto bi = b.cables().begin();
    auto be = b.cables().end();

    while (ai!=ae && bi!=be) {
        if (precedes(*ai, *bi)) {
            ++ai;
        }
        else if (precedes(*bi, *ai)) {
            ++bi;
        }
        else {
            m.cables_.push_back(mcable{ai->branch,
                std::max(ai->prox_pos, bi->prox_pos),
                std::min(ai->dist_pos, bi->dist_pos)});
            if (ai->dist_pos<bi->dist_pos) {
                ++ai;
            }
            else {
                ++bi;
            }
        }
    }
    return m;
}

mextent join(const mextent& a, const mextent& b) {
    mextent m;
    mcable_list& cs = m.cables_;

    for (auto c: util::merge_view(a.cables(), b.cables())) {
        mcable* prev = cs.empty()? nullptr: &cs.back();

        if (prev && prev->branch==c.branch && prev->dist_pos>=c.prox_pos) {
            prev->dist_pos = std::max(prev->dist_pos, c.dist_pos);
        }
        else {
            cs.push_back(c);
        }
    }
    return m;
}

std::vector<mextent> components(const morphology& m, const mextent& ex) {
    std::unordered_map<mlocation, unsigned> component_index;
    std::vector<mcable_list> component_cables;

    for (mcable c: ex) {
        mlocation head = canonical(m, prox_loc(c));

        unsigned index;
        if (component_index.count(head)) {
            index = component_index.at(head);
        }
        else {
            index = component_cables.size();
            component_cables.push_back({});
        }

        component_cables[index].push_back(c);
        component_index[dist_loc(c)] = index;
    }

    std::vector<mextent> components;
    for (auto& cl: component_cables) {
        components.emplace_back(std::move(cl));
    }
    return components;
}

} // namespace arb

