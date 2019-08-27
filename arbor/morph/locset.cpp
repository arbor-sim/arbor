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

mlocation_list merge(const mlocation_list& lhs, const mlocation_list& rhs) {
    mlocation_list v;
    v.resize(lhs.size() + rhs.size());
    std::merge(lhs.begin(), lhs.end(), rhs.begin(), rhs.end(), v.begin());
    return v;
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

std::set<std::string> named_dependencies_(const location_&) {
    return {};
}

locset replace_named_dependencies_(const location_& s, const region_dictionary& r, const locset_dictionary& p) {
    return locset(s);
}

mlocation_list concretise_(const location_& x, const em_morphology& m) {
    // canonicalize will throw if the location is not present.
    return {m.canonicalize(x.loc)};
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

std::set<std::string> named_dependencies_(const sample_&) {
    return {};
}

locset replace_named_dependencies_(const sample_& s, const region_dictionary& r, const locset_dictionary& p) {
    return locset(s);
}

mlocation_list concretise_(const sample_& x, const em_morphology& m) {
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

std::set<std::string> named_dependencies_(const terminal_&) {
    return {};
}

locset replace_named_dependencies_(const terminal_& ps, const region_dictionary& r, const locset_dictionary& p) {
    return locset(ps);
}

mlocation_list concretise_(const terminal_&, const em_morphology& m) {
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

std::set<std::string> named_dependencies_(const root_&) {
    return {};
}

locset replace_named_dependencies_(const root_& ps, const region_dictionary& r, const locset_dictionary& p) {
    return locset(ps);
}

mlocation_list concretise_(const root_&, const em_morphology& m) {
    return {m.root()};
}

std::ostream& operator<<(std::ostream& o, const root_& x) {
    return o << "root";
}

// a named locset
struct named_ {
    named_(std::string n): name(std::move(n)) {}
    std::string name;
};

locset named(std::string n) {
    return locset{named_{std::move(n)}};
}

std::set<std::string> named_dependencies_(const named_& n) {
    return {n.name};
}

locset replace_named_dependencies_(const named_& ps, const region_dictionary& r, const locset_dictionary& p) {
    auto it = p.find(ps.name);
    if (it==p.end()) {
        throw morphology_error(
            util::pprintf("internal error: unable to replace label {}, unavailable in label dictionary", ps.name));
    }
    return it->second;
}

mlocation_list concretise_(const named_&, const em_morphology& m) {
    throw morphology_error("concretise_ not implemented for named mlocation_list");
    return {};
}

std::ostream& operator<<(std::ostream& o, const named_& x) {
    return o << "\"" <<  x.name << "\"";
}

// intersection of two point sets
struct land {
    locset lhs;
    locset rhs;
    land(locset lhs, locset rhs): lhs(std::move(lhs)), rhs(std::move(rhs)) {}
};

std::set<std::string> named_dependencies_(const land& p) {
    auto l = named_dependencies(p.lhs);
    auto r = named_dependencies(p.rhs);
    l.insert(r.begin(), r.end());
    return l;
}

locset replace_named_dependencies_(const land& ps, const region_dictionary& r, const locset_dictionary& p) {
    return locset(land(replace_named_dependencies(ps.lhs, r, p),
                       replace_named_dependencies(ps.rhs, r, p)));
}

mlocation_list concretise_(const land& P, const em_morphology& m) {
    auto locs = merge(concretise(P.lhs, m), concretise(P.rhs, m));
    auto beg = locs.begin();
    auto end = locs.end();
    auto pos = beg;
    auto it = std::adjacent_find(beg, end);
    while (it!=end) {
        std::swap(*pos, *it);
        it = std::adjacent_find(++it, end);
        ++pos;
    }
    locs.resize(pos-beg);
    return locs;
}

std::ostream& operator<<(std::ostream& o, const land& x) {
    return o << "(and " << x.lhs << " " << x.rhs << ")";
}

// union of two point sets
struct locor {
    locset lhs;
    locset rhs;
    locor(locset lhs, locset rhs): lhs(std::move(lhs)), rhs(std::move(rhs)) {}
};

std::set<std::string> named_dependencies_(const locor& p) {
    auto l = named_dependencies(p.lhs);
    auto r = named_dependencies(p.rhs);
    l.insert(r.begin(), r.end());
    return l;
}

locset replace_named_dependencies_(const locor& ps, const region_dictionary& r, const locset_dictionary& p) {
    return locset(locor(replace_named_dependencies(ps.lhs, r, p),
                          replace_named_dependencies(ps.rhs, r, p)));
}

mlocation_list concretise_(const locor& P, const em_morphology& m) {
    // Concatenate locations from lhs and rhs.
    auto locs = merge(concretise(P.lhs, m), concretise(P.rhs, m));

    // Remove duplicates.
    auto it = std::unique(locs.begin(), locs.end());
    locs.resize(std::distance(locs.begin(), it));

    return locs;
}

std::ostream& operator<<(std::ostream& o, const locor& x) {
    return o << "(or " << x.lhs << " " << x.rhs << ")";
}

} // namespace ls


// The and_ and or_ operations in the arb:: namespace with locset so that
// ADL allows for construction of expressions with locsets without having
// to namespace qualify the and_/or_.

locset and_(locset lhs, locset rhs) {
    return locset(ls::land(std::move(lhs), std::move(rhs)));
}

locset or_(locset lhs, locset rhs) {
    return locset(ls::locor(std::move(lhs), std::move(rhs)));
}

} // namespace arb
