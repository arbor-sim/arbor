#pragma once

#include <memory>
#include <ostream>
#include <vector>

#include <arbor/util/lexcmp_def.hpp>
#include <arbor/morph/primitives.hpp>
#include <arbor/morph/sample_tree.hpp>

namespace arb {

struct morphology_impl;

using mindex_range = std::pair<const msize_t*, const msize_t*>;

class morphology {
    // Hold an immutable copy of the morphology implementation.
    std::shared_ptr<const morphology_impl> impl_;

public:
    morphology(sample_tree m, bool use_spherical_root);
    morphology(sample_tree m);
    morphology();

    // Empty/default-constructed morphology?
    bool empty() const;

    // Whether the root of the morphology is spherical.
    bool spherical_root() const;

    // The number of branches in the morphology.
    msize_t num_branches() const;

    // The number of samples in the morphology.
    msize_t num_samples() const;

    // The parent branch of branch b.
    // Return mnpos if branch has no parent.
    msize_t branch_parent(msize_t b) const;

    // The child branches of branch b.
    // If b is mnpos, return root branches.
    const std::vector<msize_t>& branch_children(msize_t b) const;

    // Branches with no children.
    const std::vector<msize_t>& terminal_branches() const;

    // Range of indexes into the sample points in branch b.
    mindex_range branch_indexes(msize_t b) const;

    // All of the samples in the morphology.
    const std::vector<msample>& samples() const;

    // The parent sample of sample i.
    const std::vector<msize_t>& sample_parents() const;

    // Point properties of samples in the morphology.
    const std::vector<point_prop>& sample_props() const;

    friend std::ostream& operator<<(std::ostream&, const morphology&);
};

// Morphology utility functions.
mlocation_list minset(const morphology&, const mlocation_list&);

mlocation canonical(const morphology&, mlocation);

// Represent a (possibly empty or disconnected) region on a morphology.
// Wraps an mcable_list, and satisfies the additional constraints:
//
//    I.  Any two cables on the same branch are strictly disjoint, i.e.
//        for cables p and q on the same branch, either p.prox_pos > q.dist_pos
//        or p.dist_pos < q.prox_pos.
//
//    II. For any branch b on the morphology tree that intersects the subset
//        described by the extent, there exists a cable in the extent defined
//        on this branch b.
//
// While an mextent can only be built from an mcable_list in conjunction with
// a morphology, union, intersection, and location membership operations can
// be performed without one.
struct mextent {
    mextent() = default;
    mextent(const mextent&) = default;
    mextent(mextent&&) = default;

    mextent& operator=(const mextent&) = default;
    mextent& operator=(mextent&&) = default;

    mextent(const morphology&, const mcable_list&);

    bool test_invariants() const; // check invariant (I) above.
    bool test_invariants(const morphology&) const; // check invariants (I) and (II) above.

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

// Reduced representation of an extent, excluding zero-length cables
// that are covered by more proximal or non-zero-length cables.
mcable_list canonical(const morphology& m, const mextent& a);

} // namespace arb
