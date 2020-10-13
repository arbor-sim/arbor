#include <cmath>
#include <memory>
#include <vector>

#include <arbor/morph/morphology.hpp>
#include <arbor/morph/place_pwlin.hpp>
#include <arbor/morph/primitives.hpp>

#include "morph/pwlin_common.hpp"
#include "util/piecewise.hpp"
#include "util/rangeutil.hpp"
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

    std::vector<double> sample_pos_on_branch;

    std::vector<double> seg_pos;
    for (msize_t bid = 0; bid<n_branch; ++bid) {
        auto& segments = m.branch_segments(bid);
        arb_assert(!segments.empty());

        seg_pos.reserve(segments.size()+1);
        seg_pos = {0};
        for (auto& seg: segments) {
            seg_pos.push_back(seg_pos.back()+distance(seg.prox, seg.dist));
        }

        double branch_length = seg_pos.back();
        double length_scale = branch_length>0? 1./branch_length: 0;
        for (auto& x: seg_pos) {
            x *= length_scale;
        }

        if (length_scale==0) {
            // Zero-length branch case?
            mpoint p = iso.apply(segments[0].prox);

            data_->x[bid].push_back(0., 1., rat_element<1, 0>(p.x, p.x));
            data_->y[bid].push_back(0., 1., rat_element<1, 0>(p.y, p.y));
            data_->z[bid].push_back(0., 1., rat_element<1, 0>(p.z, p.z));
            data_->r[bid].push_back(0., 1., rat_element<1, 0>(p.radius, p.radius));
        }
        else {
            for (auto i: util::count_along(segments)) {
                auto& seg = segments[i];
                double p0 = seg_pos[i];
                double p1 = seg_pos[i+1];
                if (p0==p1) continue;

                mpoint u0 = iso.apply(seg.prox);
                mpoint u1 = iso.apply(seg.dist);

                data_->x[bid].push_back(p0, p1, rat_element<1, 0>(u0.x, u1.x));
                data_->y[bid].push_back(p0, p1, rat_element<1, 0>(u0.y, u1.y));
                data_->z[bid].push_back(p0, p1, rat_element<1, 0>(u0.z, u1.z));
                data_->r[bid].push_back(p0, p1, rat_element<1, 0>(u0.radius, u1.radius));
            }
        }
    }
};

} // namespace arb
