#pragma once

#include <memory>
#include <ostream>
#include <vector>

#include <arbor/export.hpp>
#include <arbor/morph/primitives.hpp>
#include <arbor/morph/segment_tree.hpp>
#include <arbor/util/lexcmp_def.hpp>

namespace arb {

struct morphology_impl;

class ARB_ARBOR_API morphology {
    // Hold an immutable copy of the morphology implementation.
    std::shared_ptr<const morphology_impl> impl_;

public:
    morphology(segment_tree m);
    morphology();

    // Empty/default-constructed morphology?
    bool empty() const;

    // The number of branches in the morphology.
    msize_t num_branches() const;

    // The parent branch of branch b.
    // Return mnpos if branch has no parent.
    msize_t branch_parent(msize_t b) const;

    // The child branches of branch b.
    // If b is mnpos, return root branches.
    const std::vector<msize_t>& branch_children(msize_t b) const;

    // Branches with no children.
    const std::vector<msize_t>& terminal_branches() const;

    // Range of segments in a branch.
    const std::vector<msegment>& branch_segments(msize_t b) const;

    // Convert to segment tree
    // Note: tree == arb::morphology(tree).to_segment_tree() is not guaranteed
    // to be true.
    segment_tree to_segment_tree() const;

    friend std::ostream& operator<<(std::ostream&, const morphology&);
};

// Represent a (possibly empty or disconnected) region on a morphology.
//
// Wraps an mcable_list, and satisfies the additional constraint that
// any two cables on the same branch are strictly disjoint, i.e.
// for cables p and q on the same branch, either p.prox_pos > q.dist_pos
// or p.dist_pos < q.prox_pos.
//
// Union, intersection, and location membership operations can be performed
// without a morphology.
// A morphology is required to assert the invariant that an mextent does
// not contain branches not in the morphology.
struct ARB_ARBOR_API mextent {
    mextent() = default;
    mextent(const mextent&) = default;
    mextent(mextent&&) = default;

    mextent& operator=(const mextent&) = default;
    mextent& operator=(mextent&&) = default;

    mextent(const mcable_list&);

    // Check that the cable segments are valid, and that cables are strictly disjoint.
    bool test_invariants() const;
    // Checks the above, along with asserting that only cables in the morphology
    // are present.
    bool test_invariants(const morphology&) const;

    const mcable_list& cables() const {
        return cables_;
    }

    bool operator==(const mextent& a) const { return cables_==a.cables_; }
    bool operator!=(const mextent& a) const { return cables_!=a.cables_; }

    bool intersects(const mcable_list& a) const;
    bool intersects(const mcable& a) const { return intersects(mcable_list{a}); }

    bool intersects(const mextent& a) const {
        return intersects(a.cables());
    }
    bool intersects(mlocation loc) const {
        return intersects(mcable{loc.branch, loc.pos, loc.pos});
    }

    friend mextent intersect(const mextent& a, const mextent& b);
    friend mextent join(const mextent& a, const mextent& b);

    // Forward const container operations:
    decltype(auto) cbegin() const { return cables_.cbegin(); }
    decltype(auto) begin() const { return cables_.begin(); }
    decltype(auto) cend() const { return cables_.cend(); }
    decltype(auto) end() const { return cables_.end(); }

    bool empty() const { return cables_.empty(); }
    std::size_t size() const { return cables_.size(); }
    const mcable& front() const { return cables_.front(); }
    const mcable& back() const { return cables_.back(); }

private:
    mcable_list cables_;
};

// Morphology utility functions.

ARB_ARBOR_API mlocation canonical(const morphology&, mlocation);

// Find the set of locations in an mlocation_list for which there
// are no other locations that are more proximal in that list.
ARB_ARBOR_API mlocation_list minset(const morphology&, const mlocation_list&);

// Find the set of locations in an mlocation_list for which there
// are no other locations that are more distal in the list.
ARB_ARBOR_API mlocation_list maxset(const morphology&, const mlocation_list&);

// Determine the components of an extent.
//
// Let T be the topological tree described by a morphology and C be the
// cover, comprising the disjoint union of unit intervals, one per branch.
//
// Let π be the projection from C onto T.
//
// Locations in C are ordered by distality: (b1, x1) < (b2, x2) if branches b1
// and b2 are the same and x1<x2, or else if b1 is a more proximal branch than
// b2.
//
// Locations in T are ordered by distality: given points a and b in C,
// π(a) < π(b) if a<b and π(a) is not equal to π(b).
//
// (NOTE: the notion of the cover may be extended in the future to include
// a 'most proximal point' (-1, 1) which projects to the root of the tree,
// and which is strictly more proximal than any other point in the cover.)
//
// Let two locations a,b in an extent X of C be directed-path-connected if
// there is an order-preserving map p: [0, 1] -> C such that π∘p is a
// path in T, with p(0) = a and p(1) = b.
//
// The components E_i of an extent X are subsets such that for all x and y
// in E_i, there exists a location a with both a, x and a, y
// directed-path-connected in X, and such that for all x in E_i and all y in
// E_j, with i not equal to j, x and y are not directed-path-connected in X.

ARB_ARBOR_API std::vector<mextent> components(const morphology& m, const mextent&);


} // namespace arb
