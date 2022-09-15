#include <iostream>
#include <stack>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include <arbor/morph/morphexcept.hpp>
#include <arbor/morph/morphology.hpp>
#include <arbor/morph/segment_tree.hpp>
#include <arbor/morph/primitives.hpp>

#include "io/sepval.hpp"
#include "util/mergeview.hpp"
#include "util/rangeutil.hpp"
#include "util/span.hpp"
#include "util/transform.hpp"

using arb::util::make_span;

namespace arb {
namespace impl {

auto branches_from_segment_tree(const segment_tree& tree) {
    using rtype = std::pair<std::vector<msize_t>,
                            std::vector<std::vector<msegment>>>;
    rtype branches;
    auto& branch_parents = branches.first;
    auto& branch_segs = branches.second;
    auto& seg_parents = tree.parents();
    auto& segs = tree.segments();

    auto nsegs = seg_parents.size();
    if (!nsegs) return branches;

    // Determine which branch each segment belongs to while counting the number
    // of branches in the morphology.
    std::vector<msize_t> bids(nsegs);
    int nbranches = 1;
    bids[0] = 0;
    for (auto i: make_span(1, nsegs)) {
        if (tree.is_root(i)) {
            bids[i] = nbranches++;
        }
        else {
            auto p = seg_parents[i];
            bids[i] = tree.is_fork(p)? nbranches++: bids[p];
        }
    }

    // A working vector used to track whether the first segment in a branch has been visited.
    std::vector<char> visited(nbranches);

    branch_segs.resize(nbranches);
    branch_parents.resize(nbranches);
    // Construct all of the cable segments for all of the branches.
    for (auto i: make_span(nsegs)) {
        auto p = seg_parents[i];
        auto b = bids[i];
        // If this is the first sample in the branch, set the branch's parent branch.
        if (!visited[b]) {
            branch_parents[b] = p==mnpos? mnpos: bids[p];
            visited[b] = 1;
        }
        branch_segs[b].push_back(segs[i]);
    }

    return branches;
}

} // namespace impl

//
//  morphology_impl definition and implementation
//

struct morphology_impl {
    // Branch state.
    std::vector<std::vector<msegment>> branches_;
    std::vector<msize_t> branch_parents_;
    std::vector<msize_t> root_children_;
    std::vector<msize_t> terminal_branches_;
    std::vector<std::vector<msize_t>> branch_children_;

    morphology_impl(const segment_tree& m);

    void init();

    friend std::ostream& operator<<(std::ostream&, const morphology_impl&);
};

morphology_impl::morphology_impl(const segment_tree& tree) {
    auto nsamp = tree.size();
    if (!nsamp) return;

    // Generate branches.
    auto B = impl::branches_from_segment_tree(tree);
    branches_ = std::move(B.second);
    branch_parents_ = std::move(B.first);
    auto nbranch = branches_.size();

    // Generate branch tree.
    branch_children_.resize(nbranch);
    branch_parents_.reserve(nbranch);
    for (auto i: make_span(nbranch)) {
        auto id = branch_parents_[i];
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
    if (m.branches_.empty()) {
        return o << "(morphology ())";
    }
    bool first = true;
    o << "(morphology\n  (";
    for (auto i: util::make_span(m.branches_.size())) {
        if (!first) o << "\n  ";
        o << "(" << m.branch_parents_[i] << " (" << io::sepval(m.branches_[i], " ") << "))";
        first = false;
    }
    return o << "))";
}

//
// morphology implementation
//

morphology::morphology(segment_tree m):
    impl_(std::make_shared<const morphology_impl>(std::move(m)))
{}

morphology::morphology():
    morphology(segment_tree())
{}

bool morphology::empty() const {
    return impl_->branches_.empty();
}

// The parent branch of branch b.
msize_t morphology::branch_parent(msize_t b) const {
    return impl_->branch_parents_[b];
}

// The child branches of branch b.
const std::vector<msize_t>& morphology::branch_children(msize_t b) const {
    return b==mnpos? impl_->root_children_: impl_->branch_children_[b];
}

const std::vector<msize_t>& morphology::terminal_branches() const {
    return impl_->terminal_branches_;
}

const std::vector<msegment>& morphology::branch_segments(msize_t b) const {
    return impl_->branches_[b];
}

msize_t morphology::num_branches() const {
    return impl_->branches_.size();
}

ARB_ARBOR_API segment_tree morphology::to_segment_tree() const {
    segment_tree st;
    const auto& branches = impl_->branches_;

    for (auto bi: make_span(branches.size()) ) {
        const auto& branch = branches[bi];
        for (auto si: make_span(branch.size())) {
            const auto& seg = branch[si];
            auto p = mnpos;
            if (si == 0) {
                auto bp = impl_->branch_parents_[bi];
                p = bp == mnpos ? mnpos : branches[bp].back().id;
            } else {
                p = branch[si-1].id;
            }
            st.append(p, seg.prox, seg.dist, seg.tag);
        }
    }

    return st;
}


ARB_ARBOR_API std::ostream& operator<<(std::ostream& o, const morphology& m) {
    return o << *m.impl_;
}

// Utilities.

ARB_ARBOR_API mlocation_list minset(const morphology& m, const mlocation_list& in) {
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

ARB_ARBOR_API mlocation_list maxset(const morphology& m, const mlocation_list& in_) {
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

ARB_ARBOR_API mlocation canonical(const morphology& m, mlocation loc) {
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

ARB_ARBOR_API mextent intersect(const mextent& a, const mextent& b) {
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

ARB_ARBOR_API std::vector<mextent> components(const morphology& m, const mextent& ex) {
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

