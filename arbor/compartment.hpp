#pragma once

#include <iterator>
#include <utility>

#include <common_types.hpp>
#include <math.hpp>
#include <arbor/util/compat.hpp>
#include <util/counter.hpp>
#include <util/iterutil.hpp>
#include <util/partition.hpp>
#include <util/span.hpp>
#include <util/rangeutil.hpp>
#include <util/transform.hpp>

namespace arb {

/// Defines the simplest type of compartment
/// The compartment is a conic frustrum
struct compartment {
    using value_type = double;
    using size_type = cell_local_size_type;
    using value_pair = std::pair<value_type, value_type>;

    compartment() = delete;

    compartment(
        size_type idx,
        value_type len,
        value_type r1,
        value_type r2
    )
    :   index{idx},
        radius{r1, r2},
        length{len}
    {}


    size_type index;
    std::pair<value_type, value_type> radius;
    value_type length;
};

// (NB: auto type deduction and lambda in C++14 will simplify the following)

template <typename size_type, typename real_type>
class compartment_maker {
public:
    compartment_maker(size_type n, real_type length, real_type rL, real_type rR):
        r0_{rL},
        dr_{(rR-rL)/n},
        dx_{length/n}
    {}

    compartment operator()(size_type i) const {
        return compartment(i, dx_, r0_+i*dr_, r0_+(i+1)*dr_);
    }

private:
    real_type r0_;
    real_type dr_;
    real_type dx_;
};

template <typename size_type, typename real_type>
using compartment_iterator =
    util::transform_iterator<util::counter<size_type>, compartment_maker<size_type, real_type>>;

template <typename size_type, typename real_type>
using compartment_range =
    util::range<compartment_iterator<size_type, real_type>>;


template <typename size_type, typename real_type>
compartment_range<size_type, real_type> make_compartment_range(
    size_type num_compartments,
    real_type radius_L,
    real_type radius_R,
    real_type length)
{
    return util::transform_view(
        util::span<size_type>(0, num_compartments),
        compartment_maker<size_type, real_type>(num_compartments, length, radius_L, radius_R));
}

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
        length_(algorithms::sum(lengths)),
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
    div_compartment_sampler(size_type n, const Radii& radii, const Lengths& lengths) {
        // set up offset-to-subsegment lookup and interpolation
        using namespace util;

        segs_ = make_partition(offsets_, lengths);
        compat::compiler_barrier_if_icc_leq(20160415u);

        nseg_ = size(segs_);
        scale_ = segs_.bounds().second/n;
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
    div_compartment_integrator(size_type n, const Radii& radii, const Lengths& lengths):
        div_compartment_sampler(n, radii, lengths) {}

    div_compartment operator()(size_type i) const {
        using namespace math;

        auto sleft = locate(scale_*i);
        auto scentre = locate(scale_*(i+0.5));
        auto sright = locate(scale_*(i+1));

        return div_compartment(i, integrate(sleft, scentre), integrate(scentre, sright));
    }

protected:
    semi_compartment sub_segment_frustrum(sub_segment_index a, sub_segment_index b) const {
        arb_assert(a.i==b.i && a.p<=b.p);

        auto seg = segs_[a.i];
        auto l = (b.p-a.p)*(seg.second-seg.first);
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


