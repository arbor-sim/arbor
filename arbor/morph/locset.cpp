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

// set of terminal nodes on a morphology
struct terminal_ {};

locset terminal() {
    return locset{terminal_{}};
}

std::set<std::string> do_named_dependencies(const terminal_&) {
    return {};
}

locset do_replace_named_dependencies(const terminal_& ps, const region_dictionary& r, const locset_dictionary& p) {
    return locset(ps);
}

mlocation_list do_concretise(const terminal_&, const em_morphology& m) {
    const auto& terms = m.terminals();
    mlocation_list pl;
    pl.reserve(terms.size());
    for (auto t: terms) {
        pl.push_back(t);
    }
    return pl;
}

std::ostream& operator<<(std::ostream& o, const terminal_& x) {
    return o << "terminal";
}

// the root node of a morphology
struct root_ {};

locset root() {
    return locset{root_{}};
}

std::set<std::string> do_named_dependencies(const root_&) {
    return {};
}

locset do_replace_named_dependencies(const root_& ps, const region_dictionary& r, const locset_dictionary& p) {
    return locset(ps);
}

mlocation_list do_concretise(const root_&, const em_morphology& m) {
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

std::set<std::string> do_named_dependencies(const named_& n) {
    return {n.name};
}

locset do_replace_named_dependencies(const named_& ps, const region_dictionary& r, const locset_dictionary& p) {
    auto it = p.find(ps.name);
    if (it==p.end()) {
        throw morphology_error(
            util::pprintf("internal error: unable to replace label {}, unavailable in label dictionary", ps.name));
    }
    return it->second;
}

mlocation_list do_concretise(const named_&, const em_morphology& m) {
    throw morphology_error("do_concretise not implemented for named mlocation_list");
    return {};
}

std::ostream& operator<<(std::ostream& o, const named_& x) {
    return o << "\"" <<  x.name << "\"";
}

// intersection of two point sets
struct and_ {
    locset lhs;
    locset rhs;
    and_(locset lhs, locset rhs): lhs(std::move(lhs)), rhs(std::move(rhs)) {}
};

std::set<std::string> do_named_dependencies(const and_& p) {
    auto l = named_dependencies(p.lhs);
    auto r = named_dependencies(p.rhs);
    l.insert(r.begin(), r.end());
    return l;
}

locset do_replace_named_dependencies(const and_& ps, const region_dictionary& r, const locset_dictionary& p) {
    return locset(and_(replace_named_dependencies(ps.lhs, r, p),
                       replace_named_dependencies(ps.rhs, r, p)));
}

mlocation_list do_concretise(const and_& P, const em_morphology& m) {
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

std::ostream& operator<<(std::ostream& o, const and_& x) {
    return o << "(and " << x.lhs << " " << x.rhs << ")";
}

// union of two point sets
struct or_ {
    locset lhs;
    locset rhs;
    or_(locset lhs, locset rhs): lhs(std::move(lhs)), rhs(std::move(rhs)) {}
};

std::set<std::string> do_named_dependencies(const or_& p) {
    auto l = named_dependencies(p.lhs);
    auto r = named_dependencies(p.rhs);
    l.insert(r.begin(), r.end());
    return l;
}

locset do_replace_named_dependencies(const or_& ps, const region_dictionary& r, const locset_dictionary& p) {
    return locset(or_(replace_named_dependencies(ps.lhs, r, p),
                          replace_named_dependencies(ps.rhs, r, p)));
}

mlocation_list do_concretise(const or_& P, const em_morphology& m) {
    // Concatenate locations from lhs and rhs.
    auto locs = merge(concretise(P.lhs, m), concretise(P.rhs, m));

    // Remove duplicates.
    auto it = std::unique(locs.begin(), locs.end());
    locs.resize(std::distance(locs.begin(), it));

    return locs;
}

std::ostream& operator<<(std::ostream& o, const or_& x) {
    return o << "(or " << x.lhs << " " << x.rhs << ")";
}

} // namespace ls

} // namespace arb
