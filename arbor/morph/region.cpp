#include <set>
#include <string>
#include <vector>

#include <arbor/morph/error.hpp>
#include <arbor/morph/locset.hpp>
#include <arbor/morph/primitives.hpp>
#include <arbor/morph/region.hpp>

#include "morph/em_morphology.hpp"
#include "util/span.hpp"
#include "util/strprintf.hpp"
#include "util/range.hpp"
#include "util/rangeutil.hpp"

namespace arb {
namespace reg {

// Returns true iff cable sections a and b:
//  1. are on the same branch
//  2. overlap, i.e. their union is not empty.
bool is_disjoint(const mcable& a, const mcable& b) {
    if (a.branch!=b.branch) return true;
    return a<b? a.dist_pos<b.prox_pos: b.dist_pos<a.prox_pos;
}

// Take the union of two cable sections that are not disjoint.
mcable make_union(const mcable& a, const mcable& b) {
    assert(!is_disjoint(a,b));
    return mcable{a.branch, std::min(a.prox_pos, b.prox_pos), std::max(a.dist_pos, b.dist_pos)};
}

// Take the intersection of two cable sections that are not disjoint.
mcable make_intersection(const mcable& a, const mcable& b) {
    assert(!is_disjoint(a,b));

    return mcable{a.branch, std::max(a.prox_pos, b.prox_pos), std::min(a.dist_pos, b.dist_pos)};
}

mcable_list merge(const mcable_list& v) {
    if (v.size()<2) return v;
    std::vector<mcable> L;
    L.reserve(v.size());
    auto c = v.front();
    auto it = v.begin()+1;
    while (it!=v.end()) {
        if (!is_disjoint(c, *it)) {
            c = make_union(c, *it);
        }
        else {
            L.push_back(c);
            c = *it;
        }
        ++it;
    }
    L.push_back(c);
    return L;
}

// Insert a zero-length region at the start of each child branch for every cable
// that includes the end of a branch.
// Insert a zero-length region at the end of the parent branch for each cable
// that includes the start of the branch.
mcable_list cover(mcable_list cables, const em_morphology& m) {
    mcable_list L;
    for (auto& c: cables) {
        if (c.prox_pos==0) {
            for (auto& x: m.cover(mlocation{c.branch, 0}, false)) {
                L.push_back({x.branch, x.pos, x.pos});
            }
        }
        if (c.dist_pos==1) {
            for (auto& x: m.cover(mlocation{c.branch, 1}, false)) {
                L.push_back({x.branch, x.pos, x.pos});
            }
        }
    }
    L.insert(L.end(), cables.begin(), cables.end());
    util::sort(L);

    return L;
}

mcable_list remove_cover(mcable_list cables, const em_morphology& m) {
    // Find all zero-length cables at the end of cables, and convert to
    // their canonical representation.
    for (auto& c: cables) {
        if (c.dist_pos==0 || c.prox_pos==1) {
            auto cloc = m.canonicalize({c.branch, c.prox_pos});
            c = {cloc.branch, cloc.pos, cloc.pos};
        }
    }
    // Some new zero-length cables may be out of order, so sort
    // the cables.
    util::sort(cables);

    // Remove multiple copies of zero-length cables if present.
    return merge(cables);
}

//
// Null/empty region
//
struct nil_ {};

region nil() {
    return region{nil_{}};
}

mcable_list thingify_(const nil_& x, const em_morphology& m) {
    return {};
}

std::ostream& operator<<(std::ostream& o, const nil_& x) {
    return o << "nil";
}

//
// Explicit cable section
//

struct cable_ {
    mcable cable;
};

region cable(mcable c) {
    if (!test_invariants(c)) {
        throw morphology_error(util::pprintf("Invalid cable section {}", c));
    }
    return region(cable_{c});
}

region branch(msize_t bid) {
    return cable({bid, 0, 1});
}

mcable_list thingify_(const cable_& reg, const em_morphology& em) {
    auto& m = em.morph();

    if (reg.cable.branch>=m.num_branches()) {
        throw morphology_error(util::pprintf("Branch {} does not exist in morpology", reg.cable.branch));
    }

    return {reg.cable};
}

std::ostream& operator<<(std::ostream& o, const cable_& c) {
    return o << c.cable;
}

//
// region with all segments with the same numeric tag
//
struct tagged_ {
    int tag;
};

region tagged(int id) {
    return region(tagged_{id});
}

mcable_list thingify_(const tagged_& reg, const em_morphology& em) {
    auto& m = em.morph();
    size_t nb = m.num_branches();

    std::vector<mcable> L;
    L.reserve(nb);
    auto& samples = m.samples();
    auto matches     = [reg, &samples](msize_t i) {return samples[i].tag==reg.tag;};
    auto not_matches = [&matches](msize_t i) {return !matches(i);};

    for (msize_t i: util::make_span(nb)) {
        auto ids = util::make_range(m.branch_indexes(i)); // range of sample ids in branch.
        size_t ns = util::size(ids);        // number of samples in branch.

        if (ns==1) {
            // The branch is a spherical soma
            if (samples[0].tag==reg.tag) {
                L.push_back({0,0,1});
            }
            continue;
        }

        // The branch has at least 2 samples.
        // Start at begin+1 because a segment gets its tag from its distal sample.
        auto beg = std::cbegin(ids);
        auto end = std::cend(ids);

        // Find the next sample that matches reg.tag.
        auto start = std::find_if(beg+1, end, matches);
        while (start!=end) {
            // find the next sample that does not match reg.tag
            auto first = start-1;
            auto last = std::find_if(start, end, not_matches);

            auto l = first==beg? 0.: em.sample2loc(*first).pos;
            auto r = last==end?  1.: em.sample2loc(*(last-1)).pos;
            L.push_back({i, l, r});

            // Find the next sample in the branch that matches reg.tag.
            start = std::find_if(last, end, matches);
        }
    }
    if (L.size()<L.capacity()/4) {
        L.shrink_to_fit();
    }
    return L;
}

std::ostream& operator<<(std::ostream& o, const tagged_& t) {
    return o << "(tag " << t.tag << ")";
}

//
// region with all segments in a cell
//
struct all_ {};

region all() {
    return region(all_{});
}

mcable_list thingify_(const all_&, const em_morphology& m) {
    auto nb = m.morph().num_branches();
    mcable_list branches;
    branches.reserve(nb);
    for (auto i: util::make_span(nb)) {
        branches.push_back({i,0,1});
    }
    return branches;
}

std::ostream& operator<<(std::ostream& o, const all_& t) {
    return o << "all";
}

//
// intersection of two regions.
//
struct reg_and {
    region lhs;
    region rhs;
    reg_and(region lhs, region rhs): lhs(std::move(lhs)), rhs(std::move(rhs)) {}
};

mcable_list thingify_(const reg_and& P, const em_morphology& m) {
    using cable_it = std::vector<mcable>::const_iterator;
    using cable_it_pair = std::pair<cable_it, cable_it>;

    auto lhs = cover(thingify(P.lhs, m), m);
    auto rhs = cover(thingify(P.rhs, m), m);

    // Perform intersection
    cable_it_pair it{lhs.begin(), rhs.begin()};
    cable_it_pair end{lhs.end(), rhs.end()};
    std::vector<mcable> L;

    bool at_end = it.first==end.first || it.second==end.second;
    while (!at_end) {
        bool first_less = *(it.first) < *(it.second);
        auto& lhs = first_less? it.first: it.second;
        auto& rhs = first_less? it.second: it.first;
        if (!is_disjoint(*lhs, *rhs)) {
            L.push_back(make_intersection(*lhs, *rhs));
        }
        if (dist_loc(*lhs) < dist_loc(*rhs)) {
            ++lhs;
        }
        else {
            ++rhs;
        }
        at_end = it.first==end.first || it.second==end.second;
    }

    return remove_cover(L, m);
}

std::ostream& operator<<(std::ostream& o, const reg_and& x) {
    return o << "(intersect " << x.lhs << " " << x.rhs << ")";
}

//
// union of two point sets
//
struct reg_or {
    region lhs;
    region rhs;
    reg_or(region lhs, region rhs): lhs(std::move(lhs)), rhs(std::move(rhs)) {}
};

mcable_list thingify_(const reg_or& P, const em_morphology& m) {
    auto lhs = thingify(P.lhs, m);
    auto rhs = thingify(P.rhs, m);
    mcable_list L;
    L.resize(lhs.size() + rhs.size());

    std::merge(lhs.begin(), lhs.end(), rhs.begin(), rhs.end(), L.begin());

    return merge(L);
}

std::ostream& operator<<(std::ostream& o, const reg_or& x) {
    return o << "(join " << x.lhs << " " << x.rhs << ")";
}

} // namespace reg

// The intersect and join operations in the arb:: namespace with region so that
// ADL allows for construction of expressions with regions without having
// to namespace qualify the intersect/join.

region intersect(region l, region r) {
    return region{reg::reg_and(std::move(l), std::move(r))};
}

region join(region l, region r) {
    return region{reg::reg_or(std::move(l), std::move(r))};
}

region::region() {
    *this = reg::nil();
}

} // namespace arb
