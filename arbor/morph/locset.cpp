#include <algorithm>
#include <iostream>
#include <numeric>

#include <arbor/morph/locset.hpp>
#include <arbor/morph/morphexcept.hpp>
#include <arbor/morph/morphology.hpp>
#include <arbor/morph/mprovider.hpp>
#include <arbor/morph/primitives.hpp>

#include "util/rangeutil.hpp"
#include "util/strprintf.hpp"

namespace arb {
namespace ls {

// Throw on invalid mlocation.
void assert_valid(mlocation x) {
    if (!test_invariants(x)) {
        throw invalid_mlocation(x);
    }
}

// Empty locset.

struct nil_: locset_tag {};

locset nil() {
    return locset{nil_{}};
}

mlocation_list thingify_(const nil_& x, const mprovider&) {
    return {};
}

std::ostream& operator<<(std::ostream& o, const nil_& x) {
    return o << "nil";
}

// An explicit location.

struct location_: locset_tag {
    explicit location_(mlocation loc): loc(loc) {}
    mlocation loc;
};

locset location(msize_t branch, double pos) {
    mlocation loc{branch, pos};
    assert_valid(loc);
    return locset{location_{loc}};
}

mlocation_list thingify_(const location_& x, const mprovider& p) {
    assert_valid(x.loc);
    if (x.loc.branch>=p.morphology().num_branches()) {
        throw no_such_branch(x.loc.branch);
    }
    return {x.loc};
}

std::ostream& operator<<(std::ostream& o, const location_& x) {
    return o << "(location " << x.loc.branch << " " << x.loc.pos << ")";
}


// Location corresponding to a sample id.

struct sample_: locset_tag {
    explicit sample_(msize_t index): index(index) {}
    msize_t index;
};

locset sample(msize_t index) {
    return locset{sample_{index}};
}

mlocation_list thingify_(const sample_& x, const mprovider& p) {
    return {canonical(p.morphology(), p.embedding().sample_location(x.index))};
}

std::ostream& operator<<(std::ostream& o, const sample_& x) {
    return o << "(sample " << x.index << ")";
}

// Set of terminal points (most distal points).

struct terminal_: locset_tag {};

locset terminal() {
    return locset{terminal_{}};
}

mlocation_list thingify_(const terminal_&, const mprovider& p) {
    mlocation_list locs;
    util::assign(locs, util::transform_view(p.morphology().terminal_branches(),
        [](msize_t bid) { return mlocation{bid, 1.}; }));

    return locs;
}

std::ostream& operator<<(std::ostream& o, const terminal_& x) {
    return o << "(terminal)";
}

// Root location (most proximal point).

struct root_: locset_tag {};

locset root() {
    return locset{root_{}};
}

mlocation_list thingify_(const root_&, const mprovider& p) {
    return {mlocation{0, 0.}};
}

std::ostream& operator<<(std::ostream& o, const root_& x) {
    return o << "(root)";
}

// Proportional location on every branch.

struct on_branches_ { double pos; };

locset on_branches(double pos) {
    return locset{on_branches_{pos}};
}

mlocation_list thingify_(const on_branches_& ob, const mprovider& p) {
    msize_t n_branch = p.morphology().num_branches();

    mlocation_list locs;
    locs.reserve(n_branch);
    for (msize_t b = 0; b<n_branch; ++b) {
        locs.push_back({b, ob.pos});
    }
    return locs;
}

std::ostream& operator<<(std::ostream& o, const on_branches_& x) {
    return o << "(on_branchs " << x.pos << ")";
}

// Named locset.

struct named_: locset_tag {
    explicit named_(std::string name): name(std::move(name)) {}
    std::string name;
};

locset named(std::string name) {
    return locset(named_{std::move(name)});
}

mlocation_list thingify_(const named_& n, const mprovider& p) {
    return p.locset(n.name);
}

std::ostream& operator<<(std::ostream& o, const named_& x) {
    return o << "(locset \"" << x.name << "\")";
}


// Intersection of two point sets.

struct land: locset_tag {
    locset lhs;
    locset rhs;
    land(locset lhs, locset rhs): lhs(std::move(lhs)), rhs(std::move(rhs)) {}
};

mlocation_list thingify_(const land& P, const mprovider& p) {
    return intersection(thingify(P.lhs, p), thingify(P.rhs, p));
}

std::ostream& operator<<(std::ostream& o, const land& x) {
    return o << "(intersect " << x.lhs << " " << x.rhs << ")";
}

// Union of two point sets.

struct lor: locset_tag {
    locset lhs;
    locset rhs;
    lor(locset lhs, locset rhs): lhs(std::move(lhs)), rhs(std::move(rhs)) {}
};

mlocation_list thingify_(const lor& P, const mprovider& p) {
    return join(thingify(P.lhs, p), thingify(P.rhs, p));
}

std::ostream& operator<<(std::ostream& o, const lor& x) {
    return o << "(join " << x.lhs << " " << x.rhs << ")";
}

// Sum of two point sets.

struct lsum: locset_tag {
    locset lhs;
    locset rhs;
    lsum(locset lhs, locset rhs): lhs(std::move(lhs)), rhs(std::move(rhs)) {}
};

mlocation_list thingify_(const lsum& P, const mprovider& p) {
    return sum(thingify(P.lhs, p), thingify(P.rhs, p));
}

std::ostream& operator<<(std::ostream& o, const lsum& x) {
    return o << "(sum " << x.lhs << " " << x.rhs << ")";
}

} // namespace ls

// The intersect and join operations in the arb:: namespace with locset so that
// ADL allows for construction of expressions with locsets without having
// to namespace qualify the intersect/join.

locset intersect(locset lhs, locset rhs) {
    return locset(ls::land(std::move(lhs), std::move(rhs)));
}

locset join(locset lhs, locset rhs) {
    return locset(ls::lor(std::move(lhs), std::move(rhs)));
}

locset sum(locset lhs, locset rhs) {
    return locset(ls::lsum(std::move(lhs), std::move(rhs)));
}

// Implicit constructors.

locset::locset() {
    *this = ls::nil();
}

locset::locset(mlocation loc) {
    *this = ls::location(loc.branch, loc.pos);
}

locset::locset(const mlocation_list& ll) {
    *this = std::accumulate(ll.begin(), ll.end(), ls::nil(),
        [](auto& ls, auto& p) { return sum(ls, locset(p)); });
}

locset::locset(std::string name) {
    *this = ls::named(std::move(name));
}

locset::locset(const char* name) {
    *this = ls::named(name);
}

} // namespace arb
