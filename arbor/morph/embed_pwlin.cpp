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

using util::rat_element;

template <unsigned p, unsigned q>
using pw_ratpoly = util::pw_elements<rat_element<p, q>>;

template <unsigned p, unsigned q>
using branch_pw_ratpoly = std::vector<pw_ratpoly<p, q>>;

// Special case handling required for degenerate branches of length zero:
template <typename Elem>
static bool is_degenerate(const util::pw_elements<Elem>& pw) {
    return  pw.bounds().second==0;
}

template <unsigned p, unsigned q>
double interpolate(const branch_pw_ratpoly<p, q>& f, unsigned bid, double pos) {
    const auto& pw = f.at(bid);
    if (is_degenerate(pw)) pos = 0;

    auto [bounds, element] = pw(pos);

    if (bounds.first==bounds.second) return element[0];
    else {
        double x = (pos-bounds.first)/(bounds.second-bounds.first);
        return element(x);
    }
}

// Length, area, and ixa are polynomial or rational polynomial functions of branch position,
// continuos and monotonically increasing with respect to distance from root.
//
// Integration wrt a piecewise constant function is performed by taking the difference between
// interpolated values at the end points of each constant interval.

template <unsigned p, unsigned q>
double integrate(const branch_pw_ratpoly<p, q>& f, unsigned bid, const pw_constant_fn& g) {
    double accum = 0;
    for (msize_t i = 0; i<g.size(); ++i) {
        std::pair<double, double> interval = g.interval(i);
        accum += g.element(i)*(interpolate(f, bid, interval.second)-interpolate(f, bid, interval.first));
    }
    return accum;
}

// Performance note: when integrating over a cable within a branch, the code effectively
// performs a linear search for the starting interval. This can be replaced with a binary
// search for a small increase in code complexity.

template <unsigned p, unsigned q>
double integrate(const branch_pw_ratpoly<p, q>& f, mcable c, const pw_constant_fn& g) {
    msize_t bid = c.branch;
    double accum = 0;

    for (msize_t i = 0; i<g.size(); ++i) {
        std::pair<double, double> interval = g.interval(i);

        if (interval.second<c.prox_pos) {
            continue;
        }
        else if (interval.first>=c.dist_pos) {
            break;
        }
        else {
            interval.first = std::max(interval.first, c.prox_pos);
            interval.second = std::min(interval.second, c.dist_pos);

            if (interval.first<interval.second) {
                accum += g.element(i)*(interpolate(f, bid, interval.second)-interpolate(f, bid, interval.first));
            }
        }
    }
    return accum;
}

template <typename operation>
mcable_list data_cmp(const branch_pw_ratpoly<1, 0>& f, unsigned bid, double val, operation op) {
    mcable_list L;
    const auto& pw = f.at(bid);
    for (const auto& piece: pw) {
        auto extents = piece.interval;
        auto left_val = piece.element(0);
        auto right_val = piece.element(1);

        if (!op(left_val, val) && !op(right_val, val)) {
            continue;
        }
        if (op(left_val, val) && op(right_val, val)) {
            L.push_back({bid, extents.first, extents.second});
            continue;
        }

        auto cable_loc = (val - left_val)/(right_val - left_val);
        auto edge = math::lerp(extents.first, extents.second, cable_loc);

        if (op(left_val, val)) {
            L.push_back({bid, extents.first, edge});
            continue;
        }
        if (!op(left_val, val)) {
            L.push_back({bid, edge, extents.second});
            continue;
        }
    }
    return L;
}

struct embed_pwlin_data {
    branch_pw_ratpoly<1, 0> length; // [µm]
    branch_pw_ratpoly<1, 0> directed_projection; // [µm]
    branch_pw_ratpoly<1, 0> radius; // [µm]
    branch_pw_ratpoly<2, 0> area; // [µm²]
    branch_pw_ratpoly<1, 1> ixa;  // [1/µm]

    explicit embed_pwlin_data(msize_t n_branch):
        length(n_branch),
        directed_projection(n_branch),
        radius(n_branch),
        area(n_branch),
        ixa(n_branch)
    {}
};

double embed_pwlin::radius(mlocation loc) const {
    return interpolate(data_->radius, loc.branch, loc.pos);
}

double embed_pwlin::directed_projection(arb::mlocation loc) const {
    return interpolate(data_->directed_projection, loc.branch, loc.pos);
}

// Point to point integration:

double embed_pwlin::integrate_length(mlocation proximal, mlocation distal) const {
    return interpolate(data_->length, distal.branch, distal.pos) -
           interpolate(data_->length, proximal.branch, proximal.pos);
}

double embed_pwlin::integrate_area(mlocation proximal, mlocation distal) const {
    return interpolate(data_->area, distal.branch, distal.pos) -
           interpolate(data_->area, proximal.branch, proximal.pos);
}

// Integrate over cable:

double embed_pwlin::integrate_length(mcable c) const {
    return integrate_length(c.branch, pw_constant_fn{{c.prox_pos, c.dist_pos}, {1.}});
}

double embed_pwlin::integrate_area(mcable c) const {
    return integrate_area(c.branch, pw_constant_fn{{c.prox_pos, c.dist_pos}, {1.}});
}

double embed_pwlin::integrate_ixa(mcable c) const {
    return integrate_ixa(c.branch, pw_constant_fn{{c.prox_pos, c.dist_pos}, {1.}});
}

// Integrate piecewise function over a branch:

double embed_pwlin::integrate_length(msize_t bid, const pw_constant_fn& g) const {
    return integrate(data_->length, bid, g);
}

double embed_pwlin::integrate_area(msize_t bid, const pw_constant_fn& g) const {
    return integrate(data_->area, bid, g);
}

double embed_pwlin::integrate_ixa(msize_t bid, const pw_constant_fn& g) const {
    return integrate(data_->ixa, bid, g);
}

// Integrate piecewise function over a cable:

double embed_pwlin::integrate_length(mcable c, const pw_constant_fn& g) const {
    return integrate(data_->length, c, g);
}

double embed_pwlin::integrate_area(mcable c, const pw_constant_fn& g) const {
    return integrate(data_->area, c, g);
}

double embed_pwlin::integrate_ixa(mcable c, const pw_constant_fn& g) const {
    return integrate(data_->ixa, c, g);
}

// Subregions defined by geometric inequalities:

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

        double length_0 = parent==mnpos? 0: data_->length[parent].back().element[1];
        data_->length[bid].push_back(0., 1, rat_element<1, 0>(length_0, length_0+branch_length));

        double area_0 = parent==mnpos? 0: data_->area[parent].back().element[2];
        double ixa_0 = parent==mnpos? 0: data_->ixa[parent].back().element[2];

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

            // (Test for positive dx explicitly in case r0 is zero.)
            double ixa_half = ixa_0 + (dx>0? dx/(pi*r0*(r0+r1)): 0);
            double ixa_1 = ixa_0 + (dx>0? dx/(pi*r0*r1): 0);
            data_->ixa[bid].push_back(p0, p1, rat_element<1, 1>(ixa_0, ixa_half, ixa_1));
            ixa_0 = ixa_1;
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
