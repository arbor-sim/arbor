#include <algorithm>
#include <iostream>
#include <numeric>

#include <arbor/morph/error.hpp>
#include <arbor/morph/locset.hpp>
#include <arbor/morph/region.hpp>

#include "util/rangeutil.hpp"
#include "util/strprintf.hpp"

#include "morph/em_morphology.hpp"

namespace arb {
namespace ls {

//
// Functions for taking the sum, union and intersection of location_lists (multisets).
//

using it_t = mlocation_list::iterator;
using const_it_t = mlocation_list::const_iterator;

// Advance an iterator to the first value that is not equal to its current
// value, or end, whichever comes first.
template <typename T>
T next_unique(T& it, T end) {
    const auto& x = *it;
    ++it;
    while (it!=end && *it==x) ++it;
    return it;
};

// Return the number of times that the value at it is repeated. Advances the
// iterator to the first value not equal to its current value, or end,
// whichever comse first.
template <typename T>
int multiplicity(T& it, T end) {
    const auto b = it;
    return std::distance(b, next_unique(it, end));
};

mlocation_list sum(const mlocation_list& lhs, const mlocation_list& rhs) {
    mlocation_list v;
    v.resize(lhs.size() + rhs.size());
    std::merge(lhs.begin(), lhs.end(), rhs.begin(), rhs.end(), v.begin());
    return v;
}

mlocation_list join(const mlocation_list& lhs, const mlocation_list& rhs) {
    mlocation_list L;
    L.reserve(lhs.size()+rhs.size());

    auto l    = lhs.begin();
    auto lend = lhs.end();
    auto r    = rhs.begin();
    auto rend = rhs.end();

    auto at_end = [&]() { return l==lend || r==rend; };
    while (!at_end()) {
        auto x = (*l<*r) ? *l: *r;
        auto count = (*l<*r)? multiplicity(l, lend):
                     (*r<*l)? multiplicity(r, rend):
                     std::max(multiplicity(l, lend), multiplicity(r, rend));
        L.insert(L.end(), count, x);
    }
    L.insert(L.end(), l, lend);
    L.insert(L.end(), r, rend);

    return L;
}

mlocation_list intersection(const mlocation_list& lhs, const mlocation_list& rhs) {
    mlocation_list L;
    L.reserve(lhs.size()+rhs.size());

    auto l    = lhs.begin();
    auto lend = lhs.end();
    auto r    = rhs.begin();
    auto rend = rhs.end();

    auto at_end = [&]() { return l==lend || r==rend; };
    while (!at_end()) {
        if (*l==*r) {
            auto x = *l;
            auto count = std::min(multiplicity(l, lend), multiplicity(r, rend));
            L.insert(L.end(), count, x);
        }
        else if (*l<*r) {
            next_unique(l, lend);
        }
        else {
            next_unique(r, rend);
        }
    }

    return L;
}

// Null set
struct nil_ {};

locset nil() {
    return locset{nil_{}};
}

mlocation_list thingify_(const nil_& x, const em_morphology& m) {
    return {};
}

std::ostream& operator<<(std::ostream& o, const nil_& x) {
    return o << "nil";
}

// An explicit location
struct location_ {
    mlocation loc;
};

locset location(mlocation loc) {
    if (!test_invariants(loc)) {
        throw morphology_error(util::pprintf("invalid location {}", loc));
    }
    return locset{location_{loc}};
}

mlocation_list thingify_(const location_& x, const em_morphology& m) {
    m.assert_valid_location(x.loc);
    return {x.loc};
}

std::ostream& operator<<(std::ostream& o, const location_& x) {
    return o << "(location " << x.loc.branch << " " << x.loc.pos << ")";
}


// Location corresponding to a sample id
struct sample_ {
    msize_t index;
};

locset sample(msize_t index) {
    return locset{sample_{index}};
}

mlocation_list thingify_(const sample_& x, const em_morphology& m) {
    return {m.sample2loc(x.index)};
}

std::ostream& operator<<(std::ostream& o, const sample_& x) {
    return o << "(sample " << x.index << ")";
}

// set of terminal nodes on a morphology
struct terminal_ {};

locset terminal() {
    return locset{terminal_{}};
}

mlocation_list thingify_(const terminal_&, const em_morphology& m) {
    return m.terminals();
}

std::ostream& operator<<(std::ostream& o, const terminal_& x) {
    return o << "terminal";
}

// the root node of a morphology
struct root_ {};

locset root() {
    return locset{root_{}};
}

mlocation_list thingify_(const root_&, const em_morphology& m) {
    return {m.root()};
}

std::ostream& operator<<(std::ostream& o, const root_& x) {
    return o << "root";
}

// intersection of two point sets
struct land {
    locset lhs;
    locset rhs;
    land(locset lhs, locset rhs): lhs(std::move(lhs)), rhs(std::move(rhs)) {}
};

mlocation_list thingify_(const land& P, const em_morphology& m) {
    return intersection(thingify(P.lhs, m), thingify(P.rhs, m));
}

std::ostream& operator<<(std::ostream& o, const land& x) {
    return o << "(intersect " << x.lhs << " " << x.rhs << ")";
}

// union of two point sets
struct lor {
    locset lhs;
    locset rhs;
    lor(locset lhs, locset rhs): lhs(std::move(lhs)), rhs(std::move(rhs)) {}
};

mlocation_list thingify_(const lor& P, const em_morphology& m) {
    return join(thingify(P.lhs, m), thingify(P.rhs, m));
}

std::ostream& operator<<(std::ostream& o, const lor& x) {
    return o << "(join " << x.lhs << " " << x.rhs << ")";
}

// sum of two point sets
struct lsum {
    locset lhs;
    locset rhs;
    lsum(locset lhs, locset rhs): lhs(std::move(lhs)), rhs(std::move(rhs)) {}
};

mlocation_list thingify_(const lsum& P, const em_morphology& m) {
    return sum(thingify(P.lhs, m), thingify(P.rhs, m));
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

locset::locset() {
    *this = ls::nil();
}

locset::locset(mlocation other) {
    *this = ls::location(other);
}

} // namespace arb
