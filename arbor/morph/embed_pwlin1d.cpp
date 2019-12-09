#include <cstddef>
#include <utility>
#include <vector>

#include <arbor/morph/embed_pwlin1d.hpp>
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

template <unsigned p, unsigned q>
double integrate(const branch_pw_ratpoly<p, q>& f, unsigned bid, const pw_constant_fn& g) {
    double accum = 0;
    for (msize_t i = 0; i<g.size(); ++i) {
        std::pair<double, double> interval = g.interval(i);
        accum += g.element(i)*(interpolate(f, bid, interval.second)-interpolate(f, bid, interval.first));
    }
    return accum;
}

struct embed_pwlin1d_data {
    branch_pw_ratpoly<1, 0> length;
    branch_pw_ratpoly<1, 0> radius;
    branch_pw_ratpoly<2, 0> area;
    branch_pw_ratpoly<1, 1> ixa;

    explicit embed_pwlin1d_data(msize_t n_branch):
        length(n_branch),
        radius(n_branch),
        area(n_branch),
        ixa(n_branch)
    {}
};

double embed_pwlin1d::radius(mlocation loc) const {
    return interpolate(data_->radius, loc.branch, loc.pos);
}

double embed_pwlin1d::integrate_length(msize_t bid, const pw_constant_fn& g) const {
    return integrate(data_->length, bid, g);
}

double embed_pwlin1d::integrate_area(msize_t bid, const pw_constant_fn& g) const {
    return integrate(data_->area, bid, g);
}

double embed_pwlin1d::integrate_ixa(msize_t bid, const pw_constant_fn& g) const {
    return integrate(data_->ixa, bid, g);
}

// Cable versions of integration methods:

double embed_pwlin1d::integrate_length(mcable c) const {
    return integrate_length(c.branch, pw_constant_fn{{c.prox_pos, c.dist_pos}, {1.}});
}

double embed_pwlin1d::integrate_area(mcable c) const {
    return integrate_area(c.branch, pw_constant_fn{{c.prox_pos, c.dist_pos}, {1.}});
}

double embed_pwlin1d::integrate_ixa(mcable c) const {
    return integrate_ixa(c.branch, pw_constant_fn{{c.prox_pos, c.dist_pos}, {1.}});
}

// Initialization, creation of geometric data.

embed_pwlin1d::embed_pwlin1d(const arb::morphology& m) {
    constexpr double pi = math::pi<double>;
    msize_t n_branch = m.num_branches();
    data_ = std::make_shared<embed_pwlin1d_data>(n_branch);

    if (!n_branch) return;

    const auto& samples = m.samples();
    sample_locations_.resize(m.num_samples());


    for (msize_t bid = 0; bid<n_branch; ++bid) {
        unsigned parent = m.branch_parent(bid);
        auto sample_indices = util::make_range(m.branch_indexes(bid));

        if (bid==0 && m.spherical_root()) {
            arb_assert(sample_indices.size()==1);

            // Treat spherical root as area-equivalent cylinder.
            double r = samples[0].loc.radius;

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

            double area_0 = parent=mnpos? 0: data_->area[parent].back().second[1];
            double ixa_0 = parent=mnpos? 0: data_->ixa[parent].back().second[1];

            if (length_scale==0) {
                // Zero-length branch? Weird, but make best show of it.
                double r = samples[sample_indices[0]].loc.radius;
                data_->radius[bid].push_back(0., 1., rat_element<1, 0>(r, r));
                data_->area[bid].push_back(0., 1., rat_element<2, 0>(area_0, area_0, area_0));
            }
            else {
                for (auto i: util::count_along(sample_indices)) {
                    if (!i) continue;

                    double p0 = i>1? sample_locations_[sample_indices[i-1]].pos: 0;
                    double p1 = sample_locations_[sample_indices[i]].pos;
                    if (p0==p1) continue;

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
