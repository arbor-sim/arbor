#pragma once

#include <iterator>
#include <utility>

#include <arbor/common_types.hpp>
#include <arbor/math.hpp>
#include <arbor/util/compat.hpp>

#include "util/iterutil.hpp"
#include "util/partition.hpp"
#include "util/rangeutil.hpp"
#include "util/transform.hpp"

namespace arb {

/// Divided compartments for use with (e.g.) fvm control volume setup

struct semi_compartment {
    using value_type = double;
    using value_pair = std::pair<value_type, value_type>;

    value_type length;
    value_type area;
    value_type volume;
    value_pair radii;

    semi_compartment& operator+=(const semi_compartment& x) {
        length += x.length;
        area += x.area;
        volume += x.volume;
        radii.second = x.radii.second;
        return *this;
    }

    static semi_compartment half_sphere(value_type r1) {
        using namespace math;
        return semi_compartment{
            r1,
            area_sphere(r1)/2,
            volume_sphere(r1)/2,
            {r1, r1}
        };
    }

    static semi_compartment frustrum(value_type l, value_type r1, value_type r2) {
        using namespace math;
        return semi_compartment{
            l,
            area_frustrum(l, r1, r2),
            volume_frustrum(l, r1, r2),
            {r1, r2}
        };
    }
};

struct div_compartment {
    using value_type = typename semi_compartment::value_type;
    using value_pair = typename semi_compartment::value_pair;
    using size_type = cell_local_size_type;

    size_type index = 0;
    semi_compartment left;
    semi_compartment right;

    div_compartment() = default;
    div_compartment(size_type i, semi_compartment l, semi_compartment r):
        index(i), left(std::move(l)), right(std::move(r))
    {}

    value_type length() const { return left.length+right.length; }
    value_type area() const { return left.area+right.area; }
    value_type volume() const { return left.volume+right.volume; }
    value_pair radii() const { return {left.radii.first, right.radii.second}; }
};

/// Divided compartments can be made from cables with sub-segments by 
/// sampling or integrating or by approximating cable as a single frustrum.

class div_compartment_by_ends {
public:
    using value_type = div_compartment::value_type;
    using size_type = div_compartment::size_type;

    // `lengths` must be a sequence of length at least one.
    // `radii` must be a sequence of length `size(lengths)+1`.
    template <typename Radii, typename Lengths>
    div_compartment_by_ends(size_type n, const Radii& radii, const Lengths& lengths):
        oon_(1/value_type(n)),
        length_(util::sum(lengths)),
        ra_(util::front(radii)),
        rb_(util::back(radii))
    {}

    div_compartment operator()(size_type i) const {
        value_type r1 = math::lerp(ra_, rb_, i*oon_);
        value_type rc = math::lerp(ra_, rb_, (i+0.5)*oon_);
        value_type r2 = math::lerp(ra_, rb_, (i+1)*oon_);
        value_type semilength = length_*oon_*0.5;

        return div_compartment(
            i,
            semi_compartment::frustrum(semilength, r1, rc),
            semi_compartment::frustrum(semilength, rc, r2)
        );
    }

private:
    value_type oon_;
    value_type length_;
    value_type ra_;
    value_type rb_;
};

class div_compartment_sampler {
public:
    using value_type = div_compartment::value_type;
    using size_type = div_compartment::size_type;

    // `lengths` must be a sequence of length at least one.
    // `radii` must be a sequence of length `size(lengths)+1`.
    template <typename Radii, typename Lengths>
    div_compartment_sampler(size_type n, const Radii& radii, const Lengths& lengths, bool with_soma = false) {
        // set up offset-to-subsegment lookup and interpolation
        using namespace util;

        with_soma_ = with_soma;
        segs_ = make_partition(offsets_, lengths);
        compat::compiler_barrier_if_icc_leq(20160415u);

        nseg_ = size(segs_);
        scale_ = with_soma_? (segs_.bounds().second - segs_.front().second)/(n-1): segs_.bounds().second/n;

        assign(radii_, radii);
        arb_assert(size(radii_)==size(offsets_));
    }

    div_compartment operator()(size_type i) const {
        using namespace math;

        auto r1 = radius_at(locate(scale_*i));
        auto rc = radius_at(locate(scale_*(i+0.5)));
        auto r2 = radius_at(locate(scale_*(i+1)));

        value_type semilength = 0.5*scale_;
        return div_compartment(
            i,
            semi_compartment::frustrum(semilength, r1, rc),
            semi_compartment::frustrum(semilength, rc, r2)
        );
    }

protected:
    struct sub_segment_index {
        size_type i;   // index
        value_type p;  // proportion [0,1] along sub-segment

        sub_segment_index(): i(0), p(0) {}

        sub_segment_index(size_type i_, value_type p_): i(i_), p(p_) {}
        bool operator<(sub_segment_index x) const {
            return i<x.i || (i==x.i && p<x.p);
        }
    };

    sub_segment_index locate(value_type x) const {
        arb_assert(x>=0);

        auto i = segs_.index(x);
        if (i==segs_.npos) {
            i = nseg_-1;
        }

        auto seg = segs_[i];
        if (x>=seg.second) {
            return sub_segment_index(i, 1);
        }
        return sub_segment_index(i, (x-seg.first)/(seg.second-seg.first));
    }

    value_type radius_at(sub_segment_index s) const {
        return math::lerp(radii_[s.i], radii_[s.i+1], s.p);
    }

    bool with_soma_ = false; //segment passed in includes the soma information in radii[0] and lengths[0]
    size_type nseg_ = 0;
    std::vector<value_type> offsets_;
    std::vector<value_type> radii_;
    value_type scale_ = 0;
    decltype(util::partition_view(offsets_)) segs_;
};

/// Overrides operator() with a more accurate method
class div_compartment_integrator: public div_compartment_sampler {
public:
    template <typename Radii, typename Lengths>
    div_compartment_integrator(size_type n, const Radii& radii, const Lengths& lengths, bool with_soma = false):
        div_compartment_sampler(n, radii, lengths, with_soma) {}

    div_compartment operator()(size_type i) const {
        using namespace math;
        sub_segment_index sleft, scentre, sright;
        // If segment includes the soma, divisions are specially calculated
        if (with_soma_) {
            auto soma_l = segs_.front().second;
            if (i == 0) {
                sleft   = locate(soma_l/2);
                scentre = locate(soma_l);
                sright  = locate(soma_l + scale_/4);
            } else if (i == 1) {
                sleft   = locate(soma_l + scale_/4);
                scentre = locate(soma_l + scale_/2);
                sright  = locate(soma_l + scale_);
            } else {
                sleft   = locate(soma_l + scale_ * (i-1));
                scentre = locate(soma_l + scale_ * (i-0.5));
                sright  = locate(soma_l + scale_ * i);
            }
        } else {
            sleft   = locate(scale_ * i);
            scentre = locate(scale_ * (i + 0.5));
            sright  = locate(scale_ * (i + 1));
        }

        return div_compartment(i, integrate(sleft, scentre), integrate(scentre, sright));
    }

protected:
    semi_compartment sub_segment_frustrum(sub_segment_index a, sub_segment_index b) const {
        arb_assert(a.i==b.i && a.p<=b.p);

        auto seg = segs_[a.i];
        auto l = (b.p-a.p)*(seg.second-seg.first);

        // If segment includes the soma it will be at index 0
        // The soma is represented as a sphere
        if (with_soma_ && a.i==0) {
            return semi_compartment::half_sphere(radii_.front());
        }
        return semi_compartment::frustrum(l, radius_at(a), radius_at(b));
    }

    semi_compartment integrate(sub_segment_index a, sub_segment_index b) const {
        sub_segment_index x = std::min(b, sub_segment_index(a.i, 1));

        auto s = sub_segment_frustrum(a, x);

        while (a.i<b.i) {
            ++a.i;
            a.p = 0;
            x = std::min(b, sub_segment_index(a.i, 1));
            s += sub_segment_frustrum(a, x);
        }

        return s;
    }
};

} // namespace arb


