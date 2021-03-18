#include <cmath>
#include <memory>
#include <vector>

#include <arbor/morph/morphology.hpp>
#include <arbor/morph/place_pwlin.hpp>
#include <arbor/morph/primitives.hpp>

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

    auto [bounds, index] = pw_index(pos);
    return interpolate_segment(bounds, data_->segments.at(index), pos);
}

std::vector<mpoint> place_pwlin::all_at(mlocation loc) const {
    std::vector<mpoint> result;
    const auto& pw_index = data_->segment_index.at(loc.branch);
    double pos = is_degenerate(pw_index)? 0: loc.pos;

    for (auto [bounds, index]: util::make_range(pw_index.equal_range(pos))) {
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

} // namespace arb
