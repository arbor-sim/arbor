#include <optional>
#include <stack>
#include <string>
#include <unordered_set>
#include <vector>

#include <arbor/morph/locset.hpp>
#include <arbor/morph/primitives.hpp>
#include <arbor/morph/morphexcept.hpp>
#include <arbor/morph/mprovider.hpp>
#include <arbor/morph/region.hpp>
#include <arbor/s_expr.hpp>

#include "util/mergeview.hpp"
#include "util/span.hpp"
#include "util/strprintf.hpp"
#include "util/range.hpp"
#include "util/rangeutil.hpp"

namespace arb {
namespace reg {

std::optional<mcable> intersect(const mcable& a, const mcable& b) {
    if (a.branch!=b.branch) return std::nullopt;

    double prox = std::max(a.prox_pos, b.prox_pos);
    double dist = std::min(a.dist_pos, b.dist_pos);
    return prox<=dist? std::optional(mcable{a.branch, prox, dist}): std::nullopt;
}

// Empty region.

struct nil_: region_tag {};

ARB_ARBOR_API region nil() {
    return region{nil_{}};
}

mextent thingify_(const nil_& x, const mprovider&) {
    return mextent{};
}

std::ostream& operator<<(std::ostream& o, const nil_& x) {
    return o << "(region-nil)";
}


// Explicit cable section.

struct cable_: region_tag {
    explicit cable_(mcable c): cable(std::move(c)) {}
    mcable cable;
};

ARB_ARBOR_API region cable(msize_t id, double prox, double dist) {
    mcable c{id, prox, dist};
    if (!test_invariants(c)) {
        throw invalid_mcable(c);
    }
    return region(cable_{c});
}

ARB_ARBOR_API region branch(msize_t bid) {
    return cable(bid, 0, 1);
}

mextent thingify_(const cable_& reg, const mprovider& p) {
    if (reg.cable.branch>=p.morphology().num_branches()) {
        throw no_such_branch(reg.cable.branch);
    }
    return mextent(mcable_list{{reg.cable}});
}

std::ostream& operator<<(std::ostream& o, const cable_& c) {
    return o << c.cable;
}

// Exlicit list of cables.
// (Not part of front-end API: used by region ctor.)

struct cable_list_: region_tag {
    explicit cable_list_(mcable_list cs): cables(std::move(cs)) {}
    mcable_list cables;
};

region cable_list(mcable_list cs) {
    if (!test_invariants(cs)) { throw invalid_mcable_list(); }
    return region(cable_list_{std::move(cs)});
}

mextent thingify_(const cable_list_& reg, const mprovider& p) {
    if (reg.cables.empty()) {
        return mextent{};
    }

    auto last_branch = reg.cables.back().branch;
    if (last_branch>=p.morphology().num_branches()) {
        throw no_such_branch(last_branch);
    }
    return mextent(reg.cables);
}

std::ostream& operator<<(std::ostream& o, const cable_list_& x) {
    o << "(cable_list";
    for (auto c: x.cables) { o << ' ' << c; }
    return o << ')';
}

// Explicit extent.
// (Not part of front-end API: used by region ctor.)

struct extent_: region_tag {
    explicit extent_(mextent x): extent(std::move(x)) {}
    mextent extent;
};

region extent(mextent x) {
    arb_assert(x.test_invariants());
    return region(extent_{std::move(x)});
}

mextent thingify_(const extent_& x, const mprovider& p) {
    arb_assert(x.extent.test_invariants(p.morphology()));
    return x.extent;
}

std::ostream& operator<<(std::ostream& o, const extent_& x) {
    o << "(extent";
    for (auto c: x.extent.cables()) { o << ' ' << c; }
    return o << ')';
}

// Region with all segments with the same numeric tag.

struct tagged_: region_tag {
    explicit tagged_(int tag): tag(tag) {}
    int tag;
};

ARB_ARBOR_API region tagged(int id) {
    return region(tagged_{id});
}

mextent thingify_(const tagged_& reg, const mprovider& p) {
    const auto& e = p.embedding();
    const auto& m = p.morphology();

    size_t nb = m.num_branches();
    std::vector<mcable> cables;

    for (msize_t i: util::make_span(nb)) {
        for (const auto& seg: m.branch_segments(i)) {
            if (seg.tag==reg.tag) {
                cables.push_back(e.segment(seg.id));
            }
        }
    }
    util::sort(cables);
    return mextent(cables);
}

std::ostream& operator<<(std::ostream& o, const tagged_& t) {
    return o << "(tag " << t.tag << ")";
}

// Region comprising a single segment.

struct segment_: region_tag {
    explicit segment_(int id): id(id) {}
    int id;
};

ARB_ARBOR_API region segment(int id) {
    return region(segment_{id});
}

mextent thingify_(const segment_& reg, const mprovider& p) {
    const auto& e = p.embedding();

    msize_t id(reg.id);
    if (id>=e.num_segments()) {
        throw no_such_segment(id);
    }

    mcable_list cables = {e.segment(id)};
    return mextent(cables);
}

std::ostream& operator<<(std::ostream& o, const segment_& reg) {
    return o << "(segment " << reg.id << ")";
}

// Region comprising whole morphology.

struct all_: region_tag {};

ARB_ARBOR_API region all() {
    return region(all_{});
}

mextent thingify_(const all_&, const mprovider& p) {
    auto nb = p.morphology().num_branches();
    mcable_list branches;
    branches.reserve(nb);
    for (auto i: util::make_span(nb)) {
        branches.push_back({i,0,1});
    }
    return mextent(branches);
}

std::ostream& operator<<(std::ostream& o, const all_& t) {
    return o << "(all)";
}

// Region comprising points up to `distance` distal to a point in `start`.

struct distal_interval_: region_tag {
    distal_interval_(locset ls, double d): start{std::move(ls)}, distance{d} {}
    locset start;
    double distance; //um
};

ARB_ARBOR_API region distal_interval(locset start, double distance) {
    return region(distal_interval_{std::move(start), distance});
}

mextent thingify_(const distal_interval_& reg, const mprovider& p) {
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

        // if we're starting at the end of a branch, start traversal with its children
        if (c.pos < 1) {
            branches_reached.push({c.branch, distance});
        } else {
            first_branch = false;
            L.push_back({c.branch,1,1});
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
    return mextent(L);
}

std::ostream& operator<<(std::ostream& o, const distal_interval_& d) {
    return d.distance==std::numeric_limits<double>::max()?
        o << "(distal-interval " << d.start << ")":
        o << "(distal-interval " << d.start << " " << d.distance << ")";
}

// Region comprising points up to `distance` proximal to a point in `end`.

struct proximal_interval_: region_tag {
    proximal_interval_(locset ls, double d): end{std::move(ls)}, distance{d} {}
    locset end;
    double distance; //um
};

ARB_ARBOR_API region proximal_interval(locset end, double distance) {
    return region(proximal_interval_{std::move(end), distance});
}

mextent thingify_(const proximal_interval_& reg, const mprovider& p) {
    const auto& m = p.morphology();
    const auto& e = p.embedding();

    std::vector<mcable> L;

    const auto start = thingify(reg.end, p);
    const auto distance = reg.distance;

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
    return mextent(L);
}

std::ostream& operator<<(std::ostream& o, const proximal_interval_& d) {
    return d.distance==std::numeric_limits<double>::max()?
        o << "(proximal-interval " << d.end << ")":
        o << "(proximal-interval " << d.end << " " << d.distance << ")";
}

mextent radius_cmp(const mprovider& p, region r, double val, comp_op op) {
    const auto& e = p.embedding();
    auto reg_extent = thingify(r, p);
    msize_t bid = mnpos;
    mcable_list cmp_cables;
    for (auto c: reg_extent) {
        if (bid != c.branch) {
            bid = c.branch;
            util::append(cmp_cables, e.radius_cmp(bid, val, op));
        }
    }
    return intersect(reg_extent, mextent(cmp_cables));
}

// Region with all segments with radius less than r
struct radius_lt_: region_tag {
    radius_lt_(region rg, double d): reg{std::move(rg)}, val{d} {}
    region reg;
    double val; //um
};

ARB_ARBOR_API region radius_lt(region reg, double val) {
    return region(radius_lt_{std::move(reg), val});
}

mextent thingify_(const radius_lt_& r, const mprovider& p) {
    return radius_cmp(p, r.reg, r.val, comp_op::lt);
}

std::ostream& operator<<(std::ostream& o, const radius_lt_& r) {
    return o << "(radius-lt " << r.reg << " " << r.val << ")";
}

// Region with all segments with radius less than r
struct radius_le_: region_tag {
    radius_le_(region rg, double d): reg{std::move(rg)}, val{d} {}
    region reg;
    double val; //um
};

ARB_ARBOR_API region radius_le(region reg, double val) {
    return region(radius_le_{std::move(reg), val});
}

mextent thingify_(const radius_le_& r, const mprovider& p) {
    return radius_cmp(p, r.reg, r.val, comp_op::le);
}

std::ostream& operator<<(std::ostream& o, const radius_le_& r) {
    return o << "(radius-le " << r.reg << " " << r.val << ")";
}

// Region with all segments with radius greater than r
struct radius_gt_: region_tag {
    radius_gt_(region rg, double d): reg{std::move(rg)}, val{d} {}
    region reg;
    double val; //um
};

ARB_ARBOR_API region radius_gt(region reg, double val) {
    return region(radius_gt_{std::move(reg), val});
}

mextent thingify_(const radius_gt_& r, const mprovider& p) {
    return radius_cmp(p, r.reg, r.val, comp_op::gt);
}

std::ostream& operator<<(std::ostream& o, const radius_gt_& r) {
    return o << "(radius-gt " << r.reg << " " << r.val << ")";
}

// Region with all segments with radius greater than or equal to r
struct radius_ge_: region_tag {
    radius_ge_(region rg, double d): reg{std::move(rg)}, val{d} {}
    region reg;
    double val; //um
};

ARB_ARBOR_API region radius_ge(region reg, double val) {
    return region(radius_ge_{std::move(reg), val});
}

mextent thingify_(const radius_ge_& r, const mprovider& p) {
    return radius_cmp(p, r.reg, r.val, comp_op::ge);
}

std::ostream& operator<<(std::ostream& o, const radius_ge_& r) {
    return o << "(radius-ge " << r.reg << " " << r.val << ")";
}

mextent projection_cmp(const mprovider& p, double v, comp_op op) {
    const auto& m = p.morphology();
    const auto& e = p.embedding();

    std::vector<mcable> L;
    auto val = v;
    for (auto i: util::make_span(m.num_branches())) {
        util::append(L, e.projection_cmp(i, val, op));
    }
    return mextent(L);
}

// Region with all segments with projection less than val
struct projection_lt_: region_tag {
    projection_lt_(double d): val{d} {}
    double val; //um
};

region projection_lt(double val) {
    return region(projection_lt_{val});
}

mextent thingify_(const projection_lt_& r, const mprovider& p) {
    return projection_cmp(p, r.val, comp_op::lt);
}

std::ostream& operator<<(std::ostream& o, const projection_lt_& r) {
    return o << "(projection-lt " << r.val << ")";
}

// Region with all segments with projection less than or equal to val
struct projection_le_: region_tag {
    projection_le_(double d): val{d} {}
    double val; //um
};

region projection_le(double val) {
    return region(projection_le_{val});
}

mextent thingify_(const projection_le_& r, const mprovider& p) {
    return projection_cmp(p, r.val, comp_op::le);
}

std::ostream& operator<<(std::ostream& o, const projection_le_& r) {
    return o << "(projection-le " << r.val << ")";
}

// Region with all segments with projection greater than val
struct projection_gt_: region_tag {
    projection_gt_(double d): val{d} {}
    double val; //um
};

region projection_gt(double val) {
    return region(projection_gt_{val});
}

mextent thingify_(const projection_gt_& r, const mprovider& p) {
    return projection_cmp(p, r.val, comp_op::gt);
}

std::ostream& operator<<(std::ostream& o, const projection_gt_& r) {
    return o << "(projection-gt " << r.val << ")";
}

// Region with all segments with projection greater than val
struct projection_ge_: region_tag {
    projection_ge_(double d): val{d} {}
    double val; //um
};

region projection_ge(double val) {
    return region(projection_ge_{val});
}

mextent thingify_(const projection_ge_& r, const mprovider& p) {
    return projection_cmp(p, r.val, comp_op::ge);
}

std::ostream& operator<<(std::ostream& o, const projection_ge_& r) {
    return o << "(projection-ge " << r.val << ")";
}

ARB_ARBOR_API region z_dist_from_root_lt(double r0) {
    if (r0 == 0) {
        return {};
    }
    region lt = reg::projection_lt(r0);
    region gt = reg::projection_gt(-r0);
    return intersect(lt, gt);
}

ARB_ARBOR_API region z_dist_from_root_le(double r0) {
    region le = reg::projection_le(r0);
    region ge = reg::projection_ge(-r0);
    return intersect(le, ge);
}

ARB_ARBOR_API region z_dist_from_root_gt(double r0) {
    region lt = reg::projection_lt(-r0);
    region gt = reg::projection_gt(r0);
    return region{join(std::move(lt), std::move(gt))};
}

ARB_ARBOR_API region z_dist_from_root_ge(double r0) {
    region lt = reg::projection_le(-r0);
    region gt = reg::projection_ge(r0);
    return region{join(std::move(lt), std::move(gt))};
}

// Named region.
struct named_: region_tag {
    explicit named_(std::string name): name(std::move(name)) {}
    std::string name;
};

ARB_ARBOR_API region named(std::string name) {
    return region(named_{std::move(name)});
}

mextent thingify_(const named_& n, const mprovider& p) {
    return p.region(n.name);
}

std::ostream& operator<<(std::ostream& o, const named_& x) {
    return o << "(region \"" << x.name << "\")";
}

// Adds all cover points to a region.
// Ensures that all valid representations of all fork points in the region are included.
struct super_: region_tag {
    explicit super_(const region& rg): reg{rg} {}
    region reg;
};

ARB_ARBOR_API region complete(region r) {
    return region(super_{std::move(r)});
}

mextent thingify_(const super_& r, const mprovider& p) {
    const auto& m = p.morphology();
    auto cables = thingify(r.reg, p).cables();
    std::unordered_set<msize_t> branch_tails;

    mcable_list cs;
    for (auto& c: cables) {
        mcable* prev = cs.empty()? nullptr: &cs.back();

        if (c.prox_pos==0) {
            branch_tails.insert(m.branch_parent(c.branch));
        }
        if (c.dist_pos==1) {
            branch_tails.insert(c.branch);
        }

        if (prev && prev->branch==c.branch && prev->dist_pos>=c.prox_pos) {
            prev->dist_pos = std::max(prev->dist_pos, c.dist_pos);
        }
        else {
            cs.push_back(c);
        }
    }

    if (!branch_tails.empty()) {
        std::vector<mcable> fork_covers;

        for (auto b: branch_tails) {
            if (b!=mnpos) fork_covers.push_back(mcable{b, 1., 1.});
            for (auto b_child: m.branch_children(b)) {
                fork_covers.push_back(mcable{b_child, 0., 0.});
            }
        }
        util::sort(fork_covers);

        // Merge cables in cs with 0-length cables corresponding to fork covers.
        mcable_list a;
        a.swap(cs);

        for (auto c: util::merge_view(a, fork_covers)) {
            mcable* prev = cs.empty()? nullptr: &cs.back();

            if (prev && prev->branch==c.branch && prev->dist_pos>=c.prox_pos) {
                prev->dist_pos = std::max(prev->dist_pos, c.dist_pos);
            }
            else {
                cs.push_back(c);
            }
        }
    }

    return {cs};
}

std::ostream& operator<<(std::ostream& o, const super_& r) {
    return o << "(complete " << r.reg << ")";
}


// Intersection of two regions.

struct reg_and: region_tag {
    region lhs;
    region rhs;
    reg_and(region lhs, region rhs): lhs(std::move(lhs)), rhs(std::move(rhs)) {}
};

mextent thingify_(const reg_and& P, const mprovider& p) {
    return intersect(thingify(P.lhs, p), thingify(P.rhs, p));
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

mextent thingify_(const reg_or& P, const mprovider& p) {
    return join(thingify(P.lhs, p), thingify(P.rhs, p));
}

std::ostream& operator<<(std::ostream& o, const reg_or& x) {
    return o << "(join " << x.lhs << " " << x.rhs << ")";
}


// Complement of a region.

struct reg_not: region_tag {
    region r;
    explicit reg_not(region r): r(std::move(r)) {}
};

mextent thingify_(const reg_not& P, const mprovider& p) {
    auto nb = p.morphology().num_branches();
    mcable_list result;

    mextent rex = thingify(P.r, p);
    auto rex_i = rex.begin();

    for (auto i: util::make_span(nb)) {
        while (rex_i!=rex.end() && rex_i->branch<i) ++rex_i;

        double x = 0;
        while (rex_i!=rex.end() && rex_i->branch==i) {
            double y = rex_i->prox_pos;
            if (y>x) {
                result.push_back(mcable{i, x, y});
            }

            x = rex_i->dist_pos;
            ++rex_i;
        }

        if (x<1) {
            result.push_back(mcable{i, x, 1});
        }
    }

    return mextent(result);
}

std::ostream& operator<<(std::ostream& o, const reg_not& x) {
    return o << "(complement " << x.r << ")";
}


// Closed set difference of two regions.

struct reg_minus: region_tag {
    region lhs;
    region rhs;
    reg_minus(region lhs, region rhs): lhs(std::move(lhs)), rhs(std::move(rhs)) {}
};

mextent thingify_(const reg_minus& P, const mprovider& p) {
    return thingify(intersect(P.lhs, complement(P.rhs)), p);
}

std::ostream& operator<<(std::ostream& o, const reg_minus& x) {
    return o << "(difference " << x.lhs << " " << x.rhs << ")";
}

} // namespace reg

// The intersect, join, complement and difference operations are in the arb::
// namespace with region so that ADL allows for construction of expressions
// with regions without having to namespace qualify these operations.

ARB_ARBOR_API region intersect(region l, region r) {
    return region{reg::reg_and(std::move(l), std::move(r))};
}

ARB_ARBOR_API region join(region l, region r) {
    return region{reg::reg_or(std::move(l), std::move(r))};
}

ARB_ARBOR_API region complement(region r) {
    return region{reg::reg_not(std::move(r))};
}

ARB_ARBOR_API region difference(region l, region r) {
    return region{reg::reg_minus(std::move(l), std::move(r))};
}

region::region() {
    *this = reg::nil();
}

// Implicit constructors/converters.

region::region(mcable c) {
    *this = reg::cable(c.branch, c.prox_pos, c.dist_pos);
}

region::region(mcable_list cl) {
    *this = reg::cable_list(std::move(cl));
}

region::region(mextent x) {
    *this = reg::extent(std::move(x));
}

} // namespace arb
