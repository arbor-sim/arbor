#include <cstddef>
#include <utility>
#include <vector>

#include <arbor/morph/embed_pwlin.hpp>
#include <arbor/morph/morphology.hpp>
#include <arbor/morph/primitives.hpp>

#include "util/piecewise.hpp"
#include "util/range.hpp"
#include "util/rangeutil.hpp"
#include "util/ratelem.hpp"
#include "util/span.hpp"

namespace arb {

// Each integrable or interpolated quantity is represented by
// a sequence of spans covering each branch, where a span is itself
// a piecewise polynomial or rational function.
//
// The piecewise functions are represented by util::pw_elements<util::rat_element<p, q>>
// objects. util::rat_element describes an order (p, q) rational function in terms of
// the values of that function at p+q+1 equally spaced points along the element, including
// the two endpoints.

using util::rat_element;
using util::pw_elements;

// Represents piecewise rational function over a subset of a branch.
// When the function represents an integral, the integral between two points
// in the domain of a pw_ratpoly can be computed by taking the difference
// in the interpolated value at the two points.

template <unsigned p, unsigned q>
using pw_ratpoly = util::pw_elements<rat_element<p, q>>;

// One branch can be covered by more than one pw_ratpoly (of the same degree).
// When the function represents an integral, the integral between two points
// that do not lie in the same domain of a pw_ratpoly must be summed by
// considering the contribution of each pw_ratpoly element separately. Multiple
// pw_ratpoly elements over the same branch are required to avoid cases of loss
// of precision and singularities.

template <unsigned p, unsigned q>
using branch_pw_spans = util::pw_elements<pw_ratpoly<p, q>>;

struct embed_pwlin_data {
    // Vectors below are all indexed by branch index.

    // Length and directed projection are piecewise linear, continuous, and
    // monotonically increasing. They are represented by a single piecewise
    // linear function per branch.

    std::vector<pw_ratpoly<1, 0>> length; // [µm]
    std::vector<pw_ratpoly<1, 0>> directed_projection; // [µm]

    // Radius is piecewise linear, but not necessarily continuous. Where the
    // morphology describes different radii at the same point, the interpolated
    // radius is regarded as being right-continuous, and corresponds to the
    // value described by the last segment added that covers the point.

    std::vector<pw_ratpoly<1, 0>> radius; // [µm]

    // Morphological surface area between points on a branch is given as
    // the difference between the interpolated value of a piecewise quadratic
    // function over the branch. The function is monotonically increasing,
    // but not necessarily continuous. Where the value jumps, the function
    // is interpreted as being right-continuous.

    std::vector<pw_ratpoly<2, 0>> area; // [µm²]

    // Integrated inverse cross-sectional area (ixa) between points on a branch
    // is used to compute the conductance between points on the morphology.
    // It is represented by an order (1, 1) rational function (i.e. a linear
    // function divided by a linear function).
    //
    // Because a small radius can lead to very large (or infinite, if the
    // radius is zero) values, one branch may be covered by multiple
    // piecewise functions, and computing ixa may require summing the
    // contributions from more than one.

    std::vector<branch_pw_spans<1, 1>> ixa;  // [1/µm]

    explicit embed_pwlin_data(msize_t n_branch):
        length(n_branch),
        directed_projection(n_branch),
        radius(n_branch),
        area(n_branch),
        ixa(n_branch)
    {}
};

// Interpolation

// Value at pos on a branch interpolated from a piecewise rational function
// which covers pos. pos is in [0, 1], and the bounds of the piecewise rational
// function is an interval [a, b] with 0 ≤ a ≤ pos ≤ b ≤ 1.

template <unsigned p, unsigned q>
double interpolate(double pos, const pw_ratpoly<p, q>& f) {
    auto [extent, poly] = f(pos);
    auto [left, right] = extent;

    return left==right? poly[0]: poly((pos-left)/(right-left));
}

// Integration

// Integral of piecwise constant function g with respect to a measure determined
// by a single piecewise monotonic right continuous rational function f, over
// an interval [r, s]: ∫[r,s] g(x) df(x) (where [r, s] is the domain of g).

template <unsigned p, unsigned q>
double integrate(const pw_constant_fn& g, const pw_ratpoly<p, q>& f) {
    double sum = 0;
    for (auto&& [extent, gval]: g) {
        sum += gval*(interpolate(extent.second, f)-interpolate(extent.first, f));
    }
    return sum;
}

// Integral of piecewise constant function g with respect to a measure determined
// by a contiguous sequence of piecewise monotonic right continuous rational
// functions fi with support [aⱼ, bⱼ], over an interval [r, s] with a₀ ≤ r ≤ s ≤ bₙ:
// Σⱼ ∫[r,s]∩[aⱼ,bⱼ] g(x)dfⱼ(x) (where [r, s] is the domain of g).

template <unsigned p, unsigned q>
double integrate(const pw_constant_fn& g, const pw_elements<pw_ratpoly<p, q>>& fs) {
    double sum = 0;
    for (auto&& [extent, pw_pair]: pw_zip_range(g, fs)) {
        auto [left, right] = extent;
        if (left==right) continue;

        double gval = pw_pair.first;
        pw_ratpoly<p, q> f = pw_pair.second;
        sum += gval*(interpolate(right, f)-interpolate(left, f));
    }
    return sum;
}

// Implementation of public embed_pwlin methods:

double embed_pwlin::radius(mlocation loc) const {
    return interpolate(loc.pos, data_->radius.at(loc.branch));
}

double embed_pwlin::directed_projection(arb::mlocation loc) const {
    return interpolate(loc.pos, data_->directed_projection.at(loc.branch));
}

// Point to point integration:

double embed_pwlin::integrate_length(mlocation proximal, mlocation distal) const {
    return interpolate(distal.pos, data_->length.at(distal.branch)) -
           interpolate(proximal.pos, data_->length.at(proximal.branch));
}

double embed_pwlin::integrate_area(mlocation proximal, mlocation distal) const {
    return interpolate(distal.pos, data_->area.at(distal.branch)) -
           interpolate(proximal.pos, data_->area.at(proximal.branch));
}

// Integrate piecewise function over a branch:

double embed_pwlin::integrate_length(msize_t bid, const pw_constant_fn& g) const {
    return integrate(g, data_->length.at(bid));
}

double embed_pwlin::integrate_area(msize_t bid, const pw_constant_fn& g) const {
    return integrate(g, data_->area.at(bid));
}

double embed_pwlin::integrate_ixa(msize_t bid, const pw_constant_fn& g) const {
    return integrate(g, data_->ixa.at(bid));
}

// Integrate over cable:

double embed_pwlin::integrate_length(const mcable& c) const {
    return integrate_length(c.branch, pw_constant_fn{{c.prox_pos, c.dist_pos}, {1.}});
}

double embed_pwlin::integrate_area(const mcable& c) const {
    return integrate_area(c.branch, pw_constant_fn{{c.prox_pos, c.dist_pos}, {1.}});
}

double embed_pwlin::integrate_ixa(const mcable& c) const {
    return integrate_ixa(c.branch, pw_constant_fn{{c.prox_pos, c.dist_pos}, {1.}});
}

// Integrate piecewise function over a cable:

static pw_constant_fn restrict(const pw_constant_fn& g, double left, double right) {
    return pw_zip_with(g, pw_elements<void>{{left, right}});
}

double embed_pwlin::integrate_length(const mcable& c, const pw_constant_fn& g) const {
    return integrate_length(c.branch, restrict(g, c.prox_pos, c.dist_pos));
}

double embed_pwlin::integrate_area(const mcable& c, const pw_constant_fn& g) const {
    return integrate_area(c.branch, restrict(g, c.prox_pos, c.dist_pos));
}

double embed_pwlin::integrate_ixa(const mcable& c, const pw_constant_fn& g) const {
    return integrate_ixa(c.branch, restrict(g, c.prox_pos, c.dist_pos));
}

// Subregions defined by geometric inequalities:

// Given a piecewise linear function f over a branch, a comparison operation op and a threshold v,
// determine the subset of the branch where the op(f(x), v) is true.
//
// Functions are supplied for each branch via a branch-index vector of pw_ratpoly<1, 0> functions;
// supplied branch id is the index into this vector, and is used to construct the cables
// corresponding to the matching subset.

template <typename operation>
mcable_list data_cmp(const std::vector<pw_ratpoly<1, 0>>& f_on_branch, msize_t bid, double val, operation op) {
    mcable_list L;
    for (auto&& piece: f_on_branch.at(bid)) {
        auto [left, right] = piece.extent;
        auto left_val = piece.value(0);
        auto right_val = piece.value(1);

        if (!op(left_val, val) && !op(right_val, val)) {
            continue;
        }
        if (op(left_val, val) && op(right_val, val)) {
            L.push_back({bid, left, right});
            continue;
        }

        auto cable_loc = (val - left_val)/(right_val - left_val);
        auto edge = math::lerp(left, right, cable_loc);

        if (op(left_val, val)) {
            L.push_back({bid, left, edge});
            continue;
        }
        if (!op(left_val, val)) {
            L.push_back({bid, edge, right});
            continue;
        }
    }
    return L;
}


mcable_list embed_pwlin::radius_cmp(msize_t bid, double val, comp_op op) const {
    switch (op) {
        case comp_op::lt: return data_cmp(data_->radius, bid, val, [](auto l, auto r){return l <  r;});
        case comp_op::le: return data_cmp(data_->radius, bid, val, [](auto l, auto r){return l <= r;});
        case comp_op::gt: return data_cmp(data_->radius, bid, val, [](auto l, auto r){return l >  r;});
        case comp_op::ge: return data_cmp(data_->radius, bid, val, [](auto l, auto r){return l >= r;});
        default: return {};
    }
}

mcable_list embed_pwlin::projection_cmp(msize_t bid, double val, comp_op op) const {
    switch (op) {
        case comp_op::lt: return data_cmp(data_->directed_projection, bid, val, [](auto l, auto r){return l <  r;});
        case comp_op::le: return data_cmp(data_->directed_projection, bid, val, [](auto l, auto r){return l <= r;});
        case comp_op::gt: return data_cmp(data_->directed_projection, bid, val, [](auto l, auto r){return l >  r;});
        case comp_op::ge: return data_cmp(data_->directed_projection, bid, val, [](auto l, auto r){return l >= r;});
        default: return {};
    }
}

// Initialization, creation of geometric data.

embed_pwlin::embed_pwlin(const arb::morphology& m) {
    constexpr double pi = math::pi<double>;
    msize_t n_branch = m.num_branches();
    data_ = std::make_shared<embed_pwlin_data>(n_branch);

    if (!n_branch) return;

    double proj_shift = m.branch_segments(0).front().prox.z;

    for (msize_t bid = 0; bid<n_branch; ++bid) {
        unsigned parent = m.branch_parent(bid);
        auto& segments = m.branch_segments(bid);
        arb_assert(segments.size());

        std::vector<double> seg_pos;
        seg_pos.reserve(segments.size()+1);
        seg_pos.push_back(0.);

        for (const auto &seg: segments) {
            double d = distance(seg.prox, seg.dist);
            seg_pos.push_back(seg_pos.back()+d);
        }

        double branch_length = seg_pos.back();
        if (branch_length!=0) {
            for (auto& d: seg_pos) {
                d /= branch_length;
                all_segment_ends_.push_back({bid, d});
            }
        }
        else {
            // In zero length branch, set all segment ends to be 0,
            // except for last, which is 1. This ensures that the
            // union of the cables corresponding to a branch cover
            // the branch.
            seg_pos.back() = 1;
            for (auto d: seg_pos) {
                all_segment_ends_.push_back({bid, d});
            }
        }

        // Second pass over segments to store associated cables.
        auto pos_iter = seg_pos.begin();
        for (const auto &seg: segments) {
            double pos0 = *pos_iter++;
            double pos1 = *pos_iter;

            if (seg.id>=segment_cables_.size()) {
                segment_cables_.resize(seg.id+1);
            }
            segment_cables_[seg.id] = mcable{bid, pos0, pos1};
        }

        double length_0 = parent==mnpos? 0: data_->length[parent].back().value[1];
        data_->length[bid].push_back(0., 1, rat_element<1, 0>(length_0, length_0+branch_length));

        double area_0 = parent==mnpos? 0: data_->area[parent].back().value[2];

        double ixa_last = 0;
        pw_ratpoly<1, 1> ixa_pw;

        for (auto i: util::count_along(segments)) {
            auto prox = segments[i].prox;
            auto dist = segments[i].dist;

            double p0 = seg_pos[i];
            double p1 = seg_pos[i+1];

            double z0 = prox.z - proj_shift;
            double z1 = dist.z - proj_shift;
            data_->directed_projection[bid].push_back(p0, p1, rat_element<1, 0>(z0, z1));

            double r0 = prox.radius;
            double r1 = dist.radius;
            data_->radius[bid].push_back(p0, p1, rat_element<1, 0>(r0, r1));

            double dx = (p1-p0)*branch_length;
            double dr = r1-r0;
            double c = pi*std::sqrt(dr*dr+dx*dx);
            double area_half = area_0 + (0.75*r0+0.25*r1)*c;
            double area_1 = area_0 + (r0+r1)*c;
            data_->area[bid].push_back(p0, p1, rat_element<2, 0>(area_0, area_half, area_1));
            area_0 = area_1;

            if (dx>0) {
                double ixa_0 = 0;
                double ixa_half = dx/(pi*r0*(r0+r1));
                double ixa_1 = dx/(pi*r0*r1);

                if (r0==0 && r1==0) {
                    // ixa is just not defined on this segment; represent with all
                    // infinite node values.
                    ixa_0 = INFINITY;
                    ixa_half = INFINITY;
                    ixa_1 = INFINITY;
                }
                else if (r0==0) {
                    ixa_0 = -INFINITY;
                    ixa_half = -dx/(pi*r1*r1);
                    ixa_1 = 0;
                }

                // Start a new piecewise function if last ixa value is
                // large compared to ixa_1 - ixa_0 (leading to loss
                // of precision), or if ixa_0 is already non-zero (something
                // above has declared we should start anew).

                constexpr double max_ixa_ratio = 1e5;
                if (!ixa_pw.empty() && ixa_0 == 0 && ixa_last<max_ixa_ratio*ixa_1) {
                    // Extend last pw representation on the branch.
                    ixa_0 += ixa_last;
                    ixa_half += ixa_last;
                    ixa_1 += ixa_last;
                }
                else {
                    // Start a new pw representation on the branch:
                    if (!ixa_pw.empty()) {
                        auto [left, right] = ixa_pw.bounds();
                        data_->ixa[bid].push_back(left, right, ixa_pw);
                    }
                    ixa_pw.clear();
                }
                ixa_pw.push_back(p0, p1, rat_element<1, 1>(ixa_0, ixa_half, ixa_1));
                ixa_last = ixa_1;
            }
        }
        // push last ixa pw function on the branch, if nonempty.
        if (!ixa_pw.empty()) {
            auto [left, right] = ixa_pw.bounds();
            data_->ixa[bid].push_back(left, right, ixa_pw);
        }

        arb_assert((data_->radius[bid].size()>0));
        if (branch_length!=0) {
            arb_assert((data_->radius[bid].bounds()==std::pair<double, double>(0., 1.)));
            arb_assert((data_->area[bid].bounds()==std::pair<double, double>(0., 1.)));
            arb_assert((data_->ixa[bid].bounds()==std::pair<double, double>(0., 1.)));
        }
    }
};

} // namespace arb
