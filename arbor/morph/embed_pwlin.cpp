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

template <unsigned p, unsigned q>
double interpolate(const branch_pw_ratpoly<p, q>& f, unsigned bid, double pos) {
    const auto& pw = f.at(bid);
    unsigned index = pw.index_of(pos);

    const auto& element = pw.element(index);
    std::pair<double, double> bounds = pw.interval(index);

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
        auto extents = piece.first;
        auto left_val = piece.second(0);
        auto right_val = piece.second(1);

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
    branch_pw_ratpoly<1, 0> length;
    branch_pw_ratpoly<1, 0> directed_projection;
    branch_pw_ratpoly<1, 0> radius;
    branch_pw_ratpoly<2, 0> area;
    branch_pw_ratpoly<1, 1> ixa;

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

    const auto& samples = m.samples();
    sample_locations_.resize(m.num_samples());

    double proj_shift = samples.front().loc.z;

    for (msize_t bid = 0; bid<n_branch; ++bid) {
        unsigned parent = m.branch_parent(bid);
        auto sample_indices = util::make_range(m.branch_indexes(bid));
        if (bid==0 && m.spherical_root()) {
            arb_assert(sample_indices.size()==1);

            // Treat spherical root as area-equivalent cylinder.
            double r = samples[0].loc.radius;

            data_->directed_projection[bid].push_back(0., 1., rat_element<1, 0>(-r, r));
            data_->length[bid].push_back(0., 1., rat_element<1, 0>(0, r*2));
            data_->radius[bid].push_back(0., 1., rat_element<1, 0>(r, r));

            double cyl_area = 4*pi*r*r;
            data_->area[bid].push_back(0., 1., rat_element<2, 0>(0., cyl_area*0.5, cyl_area));

            double cyl_ixa = 2.0/(pi*r);
            data_->ixa[bid].push_back(0., 1., rat_element<1, 1>(0., cyl_ixa*0.5, cyl_ixa));

            sample_locations_[0] = mlocation{0, 0.5};
        }
        else {
            arb_assert(sample_indices.size()>1);

            std::vector<double> sample_distance;
            sample_distance.reserve(samples.size());
            sample_distance.push_back(0.);

            for (auto i: util::count_along(sample_indices)) {
                if (!i) continue;

                double d = distance(samples[sample_indices[i-1]], samples[sample_indices[i]]);
                sample_distance.push_back(sample_distance.back()+d);
            }

            double branch_length = sample_distance.back();
            double length_scale = branch_length>0? 1./branch_length: 0;

            for (auto i: util::count_along(sample_indices)) {
                sample_locations_[sample_indices[i]] = mlocation{bid, length_scale*sample_distance[i]};
            }
            sample_locations_[sample_indices.back()].pos = 1.; // Circumvent any rounding infelicities.

            double length_0 = parent==mnpos? 0: data_->length[parent].back().second[1];
            data_->length[bid].push_back(0., 1, rat_element<1, 0>(length_0, length_0+branch_length));

            double area_0 = parent==mnpos? 0: data_->area[parent].back().second[2];
            double ixa_0 = parent==mnpos? 0: data_->ixa[parent].back().second[2];

            if (length_scale==0) {
                // Zero-length branch? Weird, but make best show of it.
                double r = samples[sample_indices[0]].loc.radius;
                double z = samples[sample_indices[0]].loc.z;
                data_->radius[bid].push_back(0., 1., rat_element<1, 0>(r, r));
                data_->directed_projection[bid].push_back(0., 1., rat_element<1, 0>(z-proj_shift, z-proj_shift));
                data_->area[bid].push_back(0., 1., rat_element<2, 0>(area_0, area_0, area_0));
                data_->ixa[bid].push_back(0., 1., rat_element<1, 1>(ixa_0, ixa_0, ixa_0));
            }
            else {
                for (auto i: util::count_along(sample_indices)) {
                    if (!i) continue;

                    double p0 = i>1? sample_locations_[sample_indices[i-1]].pos: 0;
                    double p1 = sample_locations_[sample_indices[i]].pos;
                    if (p0==p1) continue;

                    double z0 = samples[sample_indices[i-1]].loc.z - proj_shift;
                    double z1 = samples[sample_indices[i]].loc.z - proj_shift;
                    data_->directed_projection[bid].push_back(p0, p1, rat_element<1, 0>(z0, z1));

                    double r0 = samples[sample_indices[i-1]].loc.radius;
                    double r1 = samples[sample_indices[i]].loc.radius;
                    data_->radius[bid].push_back(p0, p1, rat_element<1, 0>(r0, r1));

                    double dx = (p1-p0)*branch_length;
                    double dr = r1-r0;
                    double c = pi*std::sqrt(dr*dr+dx*dx);
                    double area_half = area_0 + (0.75*r0+0.25*r1)*c;
                    double area_1 = area_0 + (r0+r1)*c;
                    data_->area[bid].push_back(p0, p1, rat_element<2, 0>(area_0, area_half, area_1));
                    area_0 = area_1;

                    double ixa_half = ixa_0 + dx/(pi*r0*(r0+r1));
                    double ixa_1 = ixa_0 + dx/(pi*r0*r1);
                    data_->ixa[bid].push_back(p0, p1, rat_element<1, 1>(ixa_0, ixa_half, ixa_1));
                    ixa_0 = ixa_1;
                }
            }

            arb_assert((data_->radius[bid].size()>0));
            arb_assert((data_->radius[bid].bounds()==std::pair<double, double>(0., 1.)));
            arb_assert((data_->area[bid].bounds()==std::pair<double, double>(0., 1.)));
            arb_assert((data_->ixa[bid].bounds()==std::pair<double, double>(0., 1.)));
            arb_assert(sample_locations_.size()==m.samples().size());
        }
    }
};

} // namespace arb
