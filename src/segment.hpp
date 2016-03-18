#pragma once

#include <cmath>

#include <vector>

#include "math.hpp"
#include "point.hpp"
#include "util.hpp"

/*
    We start with a high-level description of the cell
    - list of branches of the cell
        - soma, dendrites, axons
        - spatial locations if provided
            - bare minimum spatial information required is length and radius
              at each end for each of the branches, and a soma radius
        - model properties of each branch
            - mechanisms
            - clamps
            - synapses
        - list of compartments if they have been provided

    This description is not used for solving the system
    From the description we can then build a cell solver
    - e.g. the FVM formulation
    - e.g. Green's functions

*/

namespace nestmc {

enum class segmentKind {
    soma,
    dendrite,
    axon
};

class segment {
    public:

    using value_type = double;
    using point_type = point<value_type>;

    segmentKind kind() const {
        return kind_;
    }

    virtual value_type volume() const = 0;
    virtual value_type area()  const = 0;

    virtual ~segment() = default;

    protected:

    segmentKind kind_;
};

class spherical_segment : public segment
{
    public :

    using base = segment;
    using base::kind_;
    using base::value_type;
    using base::point_type;

    spherical_segment() = delete;

    spherical_segment(value_type r)
    :   radius_{r}
    {
        kind_ = segmentKind::soma;
    }

    spherical_segment(value_type r, point_type const &c)
    :   spherical_segment(r)
    {
        center_ = c;
        kind_ = segmentKind::soma;
    }

    value_type volume() const override
    {
        return 4./3. * pi<value_type>() * radius_ * radius_ * radius_;
    }

    value_type area() const override
    {
        return 4. * pi<value_type>() * radius_ * radius_;
    }

    virtual ~spherical_segment() = default;

    private :

    // store the center and radius of the soma
    // note that the center may be undefined
    value_type radius_;
    point_type center_;
};

class frustrum_segment : public segment
{
    public :

    using segment::kind_;
    using base = segment;
    using base::value_type;
    using base::point_type;

    frustrum_segment() = delete;

    frustrum_segment(
        segmentKind k,
        value_type r1,
        value_type r2,
        value_type len
    ) {
        r1_ = r1;
        r2_ = r2;
        length_ = len;
        kind_ = k;
        assert(k==segmentKind::dendrite || k ==segmentKind::axon);
    }

    frustrum_segment(
        segmentKind k,
        value_type r1,
        value_type r2,
        point_type const& p1,
        point_type const& p2
    )
    : frustrum_segment(k, r1, r2, norm(p1-p2))
    {
        p1_ = p1;
        p2_ = p2;
    }

    value_type volume() const override
    {
        return volume_frustrum(length_, r1_, r2_);
    }
    value_type area()   const override
    {
        return area_frustrum(length_, r1_, r2_);
    }

    virtual ~frustrum_segment() = default;

    private :

    value_type length_;
    value_type r1_;
    value_type r2_;
    point_type p1_;
    point_type p2_;
};

using segment_ptr = std::unique_ptr<segment>;

template <typename T, typename... Args>
segment_ptr make_segment(Args&&... args) {
    return segment_ptr(new T(std::forward<Args>(args)...));
}

} // namespace nestmc

