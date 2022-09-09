#include <iostream>

#include <cmath>
#include <limits>
#include <memory>
#include <vector>

#include <arbor/morph/morphology.hpp>
#include <arbor/morph/place_pwlin.hpp>
#include <arbor/morph/primitives.hpp>
#include <arbor/math.hpp>

#include "util/piecewise.hpp"
#include "util/rangeutil.hpp"
#include "util/ratelem.hpp"
#include "util/span.hpp"

namespace arb {

using util::rat_element;

struct place_pwlin_data {
    // Piecewise-constant indices into segment data, by branch.
    std::vector<util::pw_elements<std::size_t>> segment_index;

    // Segments from segment tree, after isometry is applied.
    std::vector<msegment> segments;

    explicit place_pwlin_data(msize_t n_branch):
        segment_index(n_branch)
    {}
};

static mpoint interpolate_segment(const std::pair<double, double>& bounds, const msegment& seg, double pos) {
    if (bounds.first==bounds.second) {
        return seg.prox;
    }
    else {
        double u = (pos-bounds.first)/(bounds.second-bounds.first);

        util::rat_element<1, 0> x{seg.prox.x, seg.dist.x};
        util::rat_element<1, 0> y{seg.prox.y, seg.dist.y};
        util::rat_element<1, 0> z{seg.prox.z, seg.dist.z};
        util::rat_element<1, 0> r{seg.prox.radius, seg.dist.radius};

        return {x(u), y(u), z(u), r(u)};
    }
}

template <typename Elem>
static bool is_degenerate(const util::pw_elements<Elem>& pw) {
    return  pw.bounds().second==0;
}

mpoint place_pwlin::at(mlocation loc) const {
    const auto& pw_index = data_->segment_index.at(loc.branch);
    double pos = is_degenerate(pw_index)? 0: loc.pos;

    auto index = pw_index(pos);
    return interpolate_segment(index.extent, data_->segments.at(index), pos);
}

std::vector<mpoint> place_pwlin::all_at(mlocation loc) const {
    std::vector<mpoint> result;
    const auto& pw_index = data_->segment_index.at(loc.branch);
    double pos = is_degenerate(pw_index)? 0: loc.pos;

    for (auto index: util::make_range(pw_index.equal_range(pos))) {
        auto bounds = index.extent;
        auto seg = data_->segments.at(index);

        // Add both ends of zero length segment, if they differ.
        if (bounds.first==bounds.second && seg.prox!=seg.dist) {
            result.push_back(seg.prox);
            result.push_back(seg.dist);
        }
        else {
            result.push_back(interpolate_segment(bounds, seg, pos));
        }
    }
    return result;
}

template <bool exclude_trivial>
static std::vector<msegment> extent_segments_impl(const place_pwlin_data& data, const mextent& extent) {
    std::vector<msegment> result;

    for (mcable c: extent) {
        const auto& pw_index = data.segment_index.at(c.branch);
        if (is_degenerate(pw_index)) {
            c.prox_pos = c.dist_pos = 0;
        }

        auto b = pw_index.equal_range(c.prox_pos).first;
        auto e = pw_index.equal_range(c.dist_pos).second;

        for (const auto [bounds, index]: util::make_range(b, e)) {
            const msegment& seg = data.segments.at(index);

            auto partial_bounds = bounds;
            msegment partial = seg;

            if (c.prox_pos>bounds.first) {
                arb_assert(c.prox_pos<=bounds.second);
                partial.prox = interpolate_segment(bounds, seg, c.prox_pos);
                partial_bounds.first = c.prox_pos;
            }
            if (c.dist_pos<bounds.second) {
                arb_assert(c.dist_pos>=bounds.first);
                partial.dist = interpolate_segment(bounds, seg, c.dist_pos);
                partial_bounds.second = c.dist_pos;
            }

            // With exclude_trivial set, skip zero-length segments if cable is non-trivial.
            if (exclude_trivial && partial_bounds.first==partial_bounds.second && c.prox_pos!=c.dist_pos) {
                continue;
            }

            result.push_back(partial);

            // With exclude_trivial set, keep only one zero-length (partial) segment if cable is trivial.
            if (exclude_trivial && c.prox_pos==c.dist_pos) {
                break;
            }
        }
    }

    return result;
}

std::vector<msegment> place_pwlin::all_segments(const mextent& extent) const {
    return extent_segments_impl<false>(*data_, extent);
}

std::vector<msegment> place_pwlin::segments(const mextent& extent) const {
    return extent_segments_impl<true>(*data_, extent);
}

place_pwlin::place_pwlin(const arb::morphology& m, const isometry& iso) {
    msize_t n_branch = m.num_branches();
    data_ = std::make_shared<place_pwlin_data>(n_branch);

    if (!n_branch) return;

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
        if (branch_length!=0) {
            for (auto& x: seg_pos) {
                x /= branch_length;
            }
        }

        for (auto i: util::count_along(segments)) {
            msegment seg = segments[i];
            double p0 = seg_pos[i];
            double p1 = seg_pos[i+1];

            seg.prox = iso.apply(seg.prox);
            seg.dist = iso.apply(seg.dist);

            data_->segment_index[bid].push_back(p0, p1, data_->segments.size());
            data_->segments.push_back(seg);
        }
    }
};

struct p3d {
    double x=0,y=0,z=0;
    constexpr p3d() = default;
    constexpr p3d(const mpoint& p): x(p.x), y(p.y), z(p.z) {}
    constexpr p3d(double x, double y, double z): x(x), y(y), z(z) {}
    friend constexpr p3d operator-(const p3d& l, const p3d& r) {
        return {l.x-r.x, l.y-r.y, l.z-r.z};
    }
    friend constexpr p3d operator+(const p3d& l, const p3d& r) {
        return {l.x+r.x, l.y+r.y, l.z+r.z};
    }
    friend constexpr p3d operator*(double l, const p3d& r) {
        return {l*r.x, l*r.y, l*r.z};
    }
    friend constexpr double dot(const p3d& l, const p3d& r) {
        return l.x*r.x + l.y*r.y + l.z*r.z;
    }
    friend double norm(const p3d& p) {
        return std::sqrt(dot(p, p));
    }
    friend std::ostream& operator<<(std::ostream& o, const p3d& p) {
        return o << '[' << p.x << ' ' << p.y << ' ' << p.z << ']';
    }
};

std::pair<std::vector<mlocation>, double> place_pwlin::all_closest(double x, double y, double z) const {
    double mind = std::numeric_limits<double>::max();
    p3d p(x,y,z);
    std::vector<mlocation> locs;

    // loop over each branch
    for (msize_t bid: util::count_along(data_->segment_index)) {
        const auto b = data_->segment_index[bid];
        // loop over the segments in the branch
        for (auto s: b) {
            const auto& seg = data_->segments[s.value];

            // v and w are the proximal and distal ends of the segment.
            const p3d v = seg.prox;
            const p3d w = seg.dist;
            const p3d vw = w-v;
            const double wvs = dot(vw, vw);
            if (wvs==0.) { // zero length segment is a special case
                const double distance = norm(p-v);
                mlocation loc{bid, s.lower_bound()};
                if (distance<mind) {
                    mind = distance;
                    locs = {loc};
                }
                else if (distance == mind) {
                    locs.push_back(loc);
                }
            }
            else {
                // Find the relative position of the orthogonal projection onto the line segment
                // that along the axis of the segment:
                //   t=0 -> proximal end of the segment
                //   t=1 -> distal end of the segment
                // values are clamped to the range [0, 1]
                const double t = std::max(0., std::min(1., dot(vw, p-v) / wvs));
                const double distance =
                    t<=0.? norm(p-v):
                    t>=1.? norm(p-w):
                           norm(p-(v + t*vw));
                mlocation loc{bid, math::lerp(s.lower_bound(), s.upper_bound(), t)};
                if (distance<mind) {
                    locs = {loc};
                    mind = distance;
                }
                else if (distance == mind) {
                    locs.push_back(loc);
                }
            }
        }
    }
    return {locs, mind};
}

// Policy:
//  If two collated points are equidistant from the input point, take the
//  proximal location.
// Rationale:
//  if the location is on a fork point, it makes sense to take the proximal
//  location, which corresponds to the end of the parent branch.
std::pair<mlocation, double> place_pwlin::closest(double x, double y, double z) const {
    const auto& [locs, delta] = all_closest(x, y, z);
    return {locs.front(), delta};
}

} // namespace arb
