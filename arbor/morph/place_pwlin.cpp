#include <cmath>
#include <memory>
#include <vector>

#include <arbor/morph/morphology.hpp>
#include <arbor/morph/place_pwlin.hpp>
#include <arbor/morph/primitives.hpp>

#include "morph/pwlin_common.hpp"
#include "util/piecewise.hpp"
#include "util/ratelem.hpp"
#include "util/span.hpp"

namespace arb {

using util::rat_element;

struct place_pwlin_data {
    branch_pw_ratpoly<1, 0> x, y, z, r; // [µm]

    explicit place_pwlin_data(msize_t n_branch):
        x(n_branch), y(n_branch), z(n_branch), r(n_branch)
    {}
};

mpoint place_pwlin::at(mlocation loc) const {
    return { interpolate(data_->x, loc.branch, loc.pos),
             interpolate(data_->y, loc.branch, loc.pos),
             interpolate(data_->z, loc.branch, loc.pos),
             interpolate(data_->r, loc.branch, loc.pos) };
}

place_pwlin::place_pwlin(const arb::morphology& m, const isometry& iso) {
    msize_t n_branch = m.num_branches();
    data_ = std::make_shared<place_pwlin_data>(n_branch);

    if (!n_branch) return;

    const auto& samples = m.samples();

    std::vector<double> sample_pos_on_branch;

    for (msize_t bid = 0; bid<n_branch; ++bid) {
        auto sample_indices = util::make_range(m.branch_indexes(bid));
        if (bid==0 && m.spherical_root()) {
            arb_assert(sample_indices.size()==1);
            arb_assert(sample_indices.front()==0);

            // Use the next distinct sample, if it exists, to determine the
            // proximal-distal axis for a spherical root.

            mpoint c = samples[0].loc;
            mpoint d = c;
            for (msize_t i = 1; i<samples.size(); ++i) {
                const auto p = samples[i].loc;
                if (p.x!=c.x || p.y!=c.y || p.z!=c.z) {
                    d = p;
                    break;
                }
            }

            mpoint u0 = c, u1 = c;

            if (c.x!=d.x || c.y!=d.y || c.z!=d.z) {
                double dx = d.x-c.x;
                double dy = d.y-c.y;
                double dz = d.z-c.z;
                double scale = c.radius/std::sqrt(dx*dx+dy*dy+dz*dz);

                u0.x -= dx*scale;
                u0.y -= dy*scale;
                u0.z -= dz*scale;

                u1.x += dx*scale;
                u1.y += dy*scale;
                u1.z += dz*scale;
            }
            else {
                // No luck, so pick distal as negative z-axis.
                u0.z += c.radius;
                u1.z -= c.radius;
            }

            u0 = iso.apply(u0);
            u1 = iso.apply(u1);

            data_->x[bid].push_back(0., 1., rat_element<1, 0>(u0.x, u1.x));
            data_->y[bid].push_back(0., 1., rat_element<1, 0>(u0.y, u1.y));
            data_->z[bid].push_back(0., 1., rat_element<1, 0>(u0.z, u1.z));
            data_->r[bid].push_back(0., 1., rat_element<1, 0>(u0.radius, u1.radius));
        }
        else {
            arb_assert(sample_indices.size()>1);

            sample_pos_on_branch.reserve(samples.size());
            sample_pos_on_branch = {0};

            for (auto i: util::count_along(sample_indices)) {
                if (!i) continue;

                sample_pos_on_branch.push_back(
                    sample_pos_on_branch.back()+
                    distance(samples[sample_indices[i-1]], samples[sample_indices[i]]));
            }

            double branch_length = sample_pos_on_branch.back();
            double length_scale = branch_length>0? 1./branch_length: 0;

            for (auto& x: sample_pos_on_branch) {
                x *= length_scale;
            }

            if (length_scale==0) {
                // Zero-length branch case?

                mpoint p = iso.apply(samples[sample_indices[0]].loc);

                data_->x[bid].push_back(0., 1., rat_element<1, 0>(p.x, p.x));
                data_->y[bid].push_back(0., 1., rat_element<1, 0>(p.y, p.y));
                data_->z[bid].push_back(0., 1., rat_element<1, 0>(p.z, p.z));
                data_->r[bid].push_back(0., 1., rat_element<1, 0>(p.radius, p.radius));
            }
            else {
                for (auto i: util::count_along(sample_indices)) {
                    if (!i) continue;

                    double p0 = i>1? sample_pos_on_branch[i-1]: 0;
                    double p1 = sample_pos_on_branch[i];
                    if (p0==p1) continue;

                    mpoint u0 = iso.apply(samples[sample_indices[i-1]].loc);
                    mpoint u1 = iso.apply(samples[sample_indices[i]].loc);

                    data_->x[bid].push_back(p0, p1, rat_element<1, 0>(u0.x, u1.x));
                    data_->y[bid].push_back(p0, p1, rat_element<1, 0>(u0.y, u1.y));
                    data_->z[bid].push_back(p0, p1, rat_element<1, 0>(u0.z, u1.z));
                    data_->r[bid].push_back(p0, p1, rat_element<1, 0>(u0.radius, u1.radius));
                }
            }
        }
    }
};

} // namespace arb
