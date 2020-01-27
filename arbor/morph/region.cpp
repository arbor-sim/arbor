#include <set>
#include <string>
#include <vector>
#include <stack>

#include <arbor/morph/locset.hpp>
#include <arbor/morph/primitives.hpp>
#include <arbor/morph/morphexcept.hpp>
#include <arbor/morph/mprovider.hpp>
#include <arbor/morph/region.hpp>

#include "util/span.hpp"
#include "util/strprintf.hpp"
#include "util/range.hpp"
#include "util/rangeutil.hpp"

namespace arb {
namespace reg {

// Head and tail of an mcable as mlocations.
inline mlocation head(mcable c) {
    return mlocation{c.branch, c.prox_pos};
}

inline mlocation tail(mcable c) {
    return mlocation{c.branch, c.dist_pos};
}

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

// List of colocated mlocations, excluding the parameter.
mlocation_list colocated(mlocation loc, const morphology& m) {
    mlocation_list L{};

    // Note: if the location is not at the end of a branch, there are no
    // other colocated points.

    if (loc.pos==0) {
        // Include head of each branch with same parent,
        // and end of parent branch if not mnpos.

        auto p = m.branch_parent(loc.branch);
        if (p!=mnpos) L.push_back({p, 1});

        for (auto b: m.branch_children(p)) {
            if (b!=loc.branch) L.push_back({b, 0});
        }
    }
    else if (loc.pos==1) {
        // Include head of each child branch.

        for (auto b: m.branch_children(loc.branch)) {
            L.push_back({b, 0});
        }
    }

    return L;
}

// Insert a zero-length region at the start of each child branch for every cable
// that includes the end of a branch.
// Insert a zero-length region at the end of the parent branch for each cable
// that includes the start of the branch.
mcable_list cover(mcable_list cables, const morphology& m) {
    mcable_list L = cables;

    for (auto& c: cables) {
        for (auto& x: colocated(head(c), m)) {
            L.push_back({x.branch, x.pos, x.pos});
        }
        for (auto& x: colocated(tail(c), m)) {
            L.push_back({x.branch, x.pos, x.pos});
        }
    }

    util::sort(L);
    return L;
}

mcable_list remove_cover(mcable_list cables, const morphology& m) {
    // Find all zero-length cables at the end of cables, and convert to
    // their canonical representation.
    for (auto& c: cables) {
        if (c.dist_pos==0 || c.prox_pos==1) {
            auto cloc = canonical(m, head(c));
            c = {cloc.branch, cloc.pos, cloc.pos};
        }
    }
    // Some new zero-length cables may be out of order, so sort
    // the cables.
    util::sort(cables);

    // Remove multiple copies of zero-length cables if present.
    return merge(cables);
}

// Remove zero-length regions from a sorted cable_list that are in the cover of another region in the list.
mcable_list remove_covered_points(mcable_list cables, const morphology& m) {
    struct branch_index_pair {
        msize_t bid;
        unsigned lid;
    };
    std::vector<branch_index_pair> end_branches;
    std::vector<unsigned> erase_indices;

    for (unsigned i = 0; i < cables.size(); i++) {
        auto c = cables[i];
        // Save zero length cables at the distal end of a branch
        // Or at the proximal end of the soma
        if ((c.prox_pos==1 && c.dist_pos==1) ||
            (c.prox_pos==0 && c.dist_pos==0)) {
            end_branches.push_back({c.branch, i});
        }
        // Look for branches that are children of the cables saved in end_branches
        else if (c.prox_pos==0) {
            auto parent = m.branch_parent(c.branch);
            if (parent==mnpos) parent = 0;

            auto it = std::find_if(end_branches.begin(), end_branches.end(),
                                   [&](branch_index_pair& p) { return p.bid==parent;});

            if (it!=end_branches.end()) {
                erase_indices.push_back((*it).lid);
                end_branches.erase(it);
            }
        }
    }

    util::sort(erase_indices);
    for (auto it = erase_indices.rbegin(); it != erase_indices.rend(); it++) {
        cables.erase(cables.begin() + *it);
    }

    return cables;
}

// Empty region.

struct nil_: region_tag {};

region nil() {
    return region{nil_{}};
}

mcable_list thingify_(const nil_& x, const mprovider&) {
    return {};
}

std::ostream& operator<<(std::ostream& o, const nil_& x) {
    return o << "nil";
}


// Explicit cable section.

struct cable_: region_tag {
    explicit cable_(mcable c): cable(std::move(c)) {}
    mcable cable;
};

region cable(msize_t id, double prox, double dist) {
    mcable c{id, prox, dist};
    if (!test_invariants(c)) {
        throw invalid_mcable(c);
    }
    return region(cable_{c});
}

region branch(msize_t bid) {
    return cable(bid, 0, 1);
}

mcable_list thingify_(const cable_& reg, const mprovider& p) {
    if (reg.cable.branch>=p.morphology().num_branches()) {
        throw no_such_branch(reg.cable.branch);
    }
    return {reg.cable};
}

std::ostream& operator<<(std::ostream& o, const cable_& c) {
    return o << c.cable;
}


// Region with all segments with the same numeric tag.

struct tagged_: region_tag {
    explicit tagged_(int tag): tag(tag) {}
    int tag;
};

region tagged(int id) {
    return region(tagged_{id});
}

mcable_list thingify_(const tagged_& reg, const mprovider& p) {
    const auto& m = p.morphology();
    const auto& e = p.embedding();
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

            auto l = first==beg? 0.: e.sample_location(*first).pos;
            auto r = last==end?  1.: e.sample_location(*(last-1)).pos;
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

// Region comprising whole morphology.

struct all_: region_tag {};

region all() {
    return region(all_{});
}

mcable_list thingify_(const all_&, const mprovider& p) {
    auto nb = p.morphology().num_branches();
    mcable_list branches;
    branches.reserve(nb);
    for (auto i: util::make_span(nb)) {
        branches.push_back({i,0,1});
    }
    return branches;
}

std::ostream& operator<<(std::ostream& o, const all_& t) {
    return o << "(all)";
}

// Region with all segments distal from another region

struct distal_interval_ {
    locset start;
    double distance; //um
};

region distal_interval(locset start, double distance) {
    return region(distal_interval_{start, distance});
}

mcable_list thingify_(const distal_interval_& reg, const mprovider& p) {
    const auto& m = p.morphology();
    const auto& e = p.embedding();

    std::vector<mcable> L;

    auto start = thingify(reg.start, p);
    auto distance = reg.distance;

    struct branch_interval {
        msize_t bid;
        double distance;
    };

    for (auto c: start) {
        std::stack<branch_interval> branches_reached;
        bool first_branch = true;

        // Transform end point of branch into the start point of its parent
        if (c.pos == 0) {
            c = {m.branch_parent(c.branch),1};
        }

        // if we're starting at the end of a branch, start traversal with its children
        if (c.pos < 1) {
            branches_reached.push({c.branch, distance});
        } else {
            first_branch = false;
            for (auto child: m.branch_children(c.branch)) {
                branches_reached.push({child, distance});
            }
        }

        while (!branches_reached.empty()) {
            auto bi = branches_reached.top();
            branches_reached.pop();

            auto branch = bi.bid;
            auto rem_dist = bi.distance;

            auto branch_length = e.branch_length(branch);
            auto prox_pos = first_branch*c.pos;
            auto dist_pos = rem_dist / branch_length + prox_pos;

            if (dist_pos <= 1) {
                L.push_back({branch, prox_pos, dist_pos});
            } else {
                L.push_back({branch, prox_pos, 1});
                rem_dist = rem_dist - (1 - prox_pos)*branch_length;
                for (auto child: m.branch_children(branch)) {
                    branches_reached.push({child, rem_dist});
                }
            }
            first_branch = false;
        }
    }
    util::sort(L);
    return merge(L);
}

std::ostream& operator<<(std::ostream& o, const distal_interval_& d) {
    return o << "(distal_interval: " << d.start << ", " << d.distance << ")";
}

// Region with all segments proximal from another region

struct proximal_interval_ {
    locset end;
    double distance; //um
};

region proximal_interval(locset end, double distance) {
    return region(proximal_interval_{end, distance});
}

mcable_list thingify_(const proximal_interval_& reg, const mprovider& p) {
    const auto& m = p.morphology();
    const auto& e = p.embedding();

    std::vector<mcable> L;

    auto start = thingify(reg.end, p);
    auto distance = reg.distance;

    for (auto c: start) {
        auto branch = c.branch;
        auto branch_length = e.branch_length(branch);
        auto rem_dist = distance;

        auto dist_pos = c.pos;
        auto prox_pos = dist_pos - distance / branch_length;

        while (prox_pos < 0) {
            L.push_back({branch, 0, dist_pos});

            rem_dist = rem_dist - dist_pos*branch_length;

            branch = m.branch_parent(branch);
            if (branch == mnpos) {
                break;
            }

            dist_pos = 1;
            prox_pos = dist_pos - rem_dist / e.branch_length(branch);
        }
        if (branch != mnpos) {
            L.push_back({branch, prox_pos, dist_pos});
        }
    }
    util::sort(L);
    return merge(L);
}

std::ostream& operator<<(std::ostream& o, const proximal_interval_& d) {
    return o << "(distal_interval: " << d.end << ", " << d.distance << ")";
}

mcable_list radius_cmp(const mprovider& p, region r, double v, comp_op op) {
    const auto& e = p.embedding();

    std::vector<mcable> L;
    auto reg = thingify(r, p);
    auto val = v;
    for (auto c: reg) {
        for (auto r: e.radius_cmp(c.branch, val, op)) {
            if (is_disjoint(c, r)) continue;
            L.push_back(make_intersection(c, r));
        }
    }
    return remove_cover(L, p.morphology());
}

// Region with all segments with radius less than r
struct radius_lt_ {
    region reg;
    double val; //um
};

region radius_lt(region reg, double val) {
    return region(radius_lt_{reg, val});
}

mcable_list thingify_(const radius_lt_& r, const mprovider& p) {
    return radius_cmp(p, r.reg, r.val, comp_op::lt);
}

std::ostream& operator<<(std::ostream& o, const radius_lt_& r) {
    return o << "(radius_lt: " << r.reg << ", " << r.val << ")";
}

// Region with all segments with radius less than r
struct radius_le_ {
    region reg;
    double val; //um
};

region radius_le(region reg, double val) {
    return region(radius_le_{reg, val});
}

mcable_list thingify_(const radius_le_& r, const mprovider& p) {
    return radius_cmp(p, r.reg, r.val, comp_op::le);
}

std::ostream& operator<<(std::ostream& o, const radius_le_& r) {
    return o << "(radius_le: " << r.reg << ", " << r.val << ")";
}

// Region with all segments with radius greater than r
struct radius_gt_ {
    region reg;
    double val; //um
};

region radius_gt(region reg, double val) {
    return region(radius_gt_{reg, val});
}

mcable_list thingify_(const radius_gt_& r, const mprovider& p) {
    return radius_cmp(p, r.reg, r.val, comp_op::gt);
}

std::ostream& operator<<(std::ostream& o, const radius_gt_& r) {
    return o << "(radius_gt: " << r.reg << ", " << r.val << ")";
}

// Region with all segments with radius greater than or equal to r
struct radius_ge_ {
    region reg;
    double val; //um
};

region radius_ge(region reg, double val) {
    return region(radius_gt_{reg, val});
}

mcable_list thingify_(const radius_ge_& r, const mprovider& p) {
    return radius_cmp(p, r.reg, r.val, comp_op::ge);
}

std::ostream& operator<<(std::ostream& o, const radius_ge_& r) {
    return o << "(radius_ge: " << r.reg << ", " << r.val << ")";
}

mcable_list projection_cmp(const mprovider& p, double v, comp_op op) {
    const auto& m = p.morphology();
    const auto& e = p.embedding();

    std::vector<mcable> L;
    auto val = v;
    for (auto i: util::make_span(m.num_branches())) {
        util::append(L, e.projection_cmp(i, val, op));
    }
    return remove_cover(L, p.morphology());
}

// Region with all segments with projection less than val
struct projection_lt_{
    double val; //um
};

region projection_lt(double val) {
    return region(projection_lt_{val});
}

mcable_list thingify_(const projection_lt_& r, const mprovider& p) {
    return projection_cmp(p, r.val, comp_op::lt);
}

std::ostream& operator<<(std::ostream& o, const projection_lt_& r) {
    return o << "(projection_lt: " << r.val << ")";
}

// Region with all segments with projection less than or equal to val
struct projection_le_{
    double val; //um
};

region projection_le(double val) {
    return region(projection_le_{val});
}

mcable_list thingify_(const projection_le_& r, const mprovider& p) {
    return projection_cmp(p, r.val, comp_op::le);
}

std::ostream& operator<<(std::ostream& o, const projection_le_& r) {
    return o << "(projection_le: " << r.val << ")";
}

// Region with all segments with projection greater than val
struct projection_gt_ {
    double val; //um
};

region projection_gt(double val) {
    return region(projection_gt_{val});
}

mcable_list thingify_(const projection_gt_& r, const mprovider& p) {
    return projection_cmp(p, r.val, comp_op::gt);
}

std::ostream& operator<<(std::ostream& o, const projection_gt_& r) {
    return o << "(projection_gt: " << r.val << ")";
}

// Region with all segments with projection greater than val
struct projection_ge_ {
    double val; //um
};

region projection_ge(double val) {
    return region(projection_ge_{val});
}

mcable_list thingify_(const projection_ge_& r, const mprovider& p) {
    return projection_cmp(p, r.val, comp_op::ge);
}

std::ostream& operator<<(std::ostream& o, const projection_ge_& r) {
    return o << "(projection_ge: " << r.val << ")";
}

region z_dist_from_soma_lt(double r0) {
    if (r0 == 0) {
        return {};
    }
    region lt = reg::projection_lt(r0);
    region gt = reg::projection_gt(-r0);
    return intersect(std::move(lt), std::move(gt));
}

region z_dist_from_soma_le(double r0) {
    region le = reg::projection_le(r0);
    region ge = reg::projection_ge(-r0);
    return intersect(std::move(le), std::move(ge));
}

region z_dist_from_soma_gt(double r0) {
    region lt = reg::projection_lt(-r0);
    region gt = reg::projection_gt(r0);
    return region{join(std::move(lt), std::move(gt))};
}

region z_dist_from_soma_ge(double r0) {
    region lt = reg::projection_le(-r0);
    region gt = reg::projection_ge(r0);
    return region{join(std::move(lt), std::move(gt))};
}

// Named region.
struct named_: region_tag {
    explicit named_(std::string name): name(std::move(name)) {}
    std::string name;
};

region named(std::string name) {
    return region(named_{std::move(name)});
}

mcable_list thingify_(const named_& n, const mprovider& p) {
    return p.region(n.name);
}

std::ostream& operator<<(std::ostream& o, const named_& x) {
    return o << "(region \"" << x.name << "\")";
}


// Intersection of two regions.

struct reg_and: region_tag {
    region lhs;
    region rhs;
    reg_and(region lhs, region rhs): lhs(std::move(lhs)), rhs(std::move(rhs)) {}
};

mcable_list thingify_(const reg_and& P, const mprovider& p) {
    auto& m = p.morphology();

    using cable_it = std::vector<mcable>::const_iterator;
    using cable_it_pair = std::pair<cable_it, cable_it>;

    auto lhs = cover(thingify(P.lhs, p), m);
    auto rhs = cover(thingify(P.rhs, p), m);

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

    return remove_covered_points(remove_cover(L, m), m);
}

std::ostream& operator<<(std::ostream& o, const reg_and& x) {
    return o << "(intersect " << x.lhs << " " << x.rhs << ")";
}


// Union of two regions.

struct reg_or: region_tag {
    region lhs;
    region rhs;
    reg_or(region lhs, region rhs): lhs(std::move(lhs)), rhs(std::move(rhs)) {}
};

mcable_list thingify_(const reg_or& P, const mprovider& p) {
    auto lhs = thingify(P.lhs, p);
    auto rhs = thingify(P.rhs, p);
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

// Implicit constructors/converters.

region::region(std::string label) {
    *this = reg::named(std::move(label));
}

region::region(const char* label) {
    *this = reg::named(label);
}

region::region(mcable c) {
    *this = reg::cable(c.branch, c.prox_pos, c.dist_pos);
}

region::region(const mcable_list& cl) {
    *this = std::accumulate(cl.begin(), cl.end(), reg::nil(),
        [](auto& rg, auto& p) { return join(rg, region(p)); });
}

} // namespace arb
