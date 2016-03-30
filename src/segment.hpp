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

// forward declarations of segment specializations
class soma_segment;
class cable_segment;

// abstract base class for a cell segment
class segment {
    public:

    using value_type = double;
    using point_type = point<value_type>;

    segmentKind kind() const {
        return kind_;
    }

    bool is_soma() const
    {
        return kind_==segmentKind::soma;
    }

    bool is_dendrite() const
    {
        return kind_==segmentKind::dendrite;
    }

    bool is_axon() const
    {
        return kind_==segmentKind::axon;
    }

    virtual value_type volume() const = 0;
    virtual value_type area()   const = 0;

    virtual ~segment() = default;

    virtual cable_segment* as_cable()
    {
        return nullptr;
    }

    virtual soma_segment* as_soma()
    {
        return nullptr;
    }

    protected:

    segmentKind kind_;
};

class soma_segment : public segment
{
    public :

    using base = segment;
    using base::kind_;
    using base::value_type;
    using base::point_type;

    soma_segment() = delete;

    soma_segment(value_type r)
    :   radius_{r}
    {
        kind_ = segmentKind::soma;
    }

    soma_segment(value_type r, point_type const &c)
    :   soma_segment(r)
    {
        center_ = c;
        kind_ = segmentKind::soma;
    }

    value_type volume() const override
    {
        return math::volume_sphere(radius_);
    }

    value_type area() const override
    {
        return math::area_sphere(radius_);
    }

    value_type radius() const
    {
        return radius_;
    }

    point_type const& center() const
    {
        return center_;
    }

    soma_segment* as_soma() override
    {
        return this;
    }

    private :

    // store the center and radius of the soma
    // note that the center may be undefined
    value_type radius_;
    point_type center_;
};

class cable_segment : public segment
{
    public :

    using base = segment;
    using base::kind_;
    using base::value_type;
    using base::point_type;

    // delete the default constructor
    cable_segment() = delete;

    // constructors for a cable with no location information
    cable_segment(
        segmentKind k,
        std::vector<value_type> r,
        std::vector<value_type> lens
    ) {
        kind_ = k;
        assert(k==segmentKind::dendrite || k==segmentKind::axon);

        radii_   = std::move(r);
        lengths_ = std::move(lens);
    }

    cable_segment(
        segmentKind k,
        value_type r1,
        value_type r2,
        value_type len
    )
    : cable_segment{k, {r1, r2}, std::vector<value_type>{len}}
    { }

    // constructor that lets the user describe the cable as a
    // seriew of radii and locations
    cable_segment(
        segmentKind k,
        std::vector<value_type> r,
        std::vector<point_type> p
    ) {
        kind_ = k;
        assert(k==segmentKind::dendrite || k==segmentKind::axon);

        radii_     = std::move(r);
        locations_ = std::move(p);
        update_lengths();
    }

    // helper that lets user specify with the radius and location
    // of just the end points of the cable
    //  i.e.    describing the cable as a single frustrum
    cable_segment(
        segmentKind k,
        value_type r1,
        value_type r2,
        point_type const& p1,
        point_type const& p2
    )
    :   cable_segment(k, {r1, r2}, {p1, p2})
    { }

    value_type volume() const override
    {
        auto sum = value_type{0};
        for(auto i=0; i<num_sub_segments(); ++i) {
            sum += math::volume_frustrum(lengths_[i], radii_[i], radii_[i+1]);
        }
        return sum;

    }

    value_type area() const override
    {
        auto sum = value_type{0};
        for(auto i=0; i<num_sub_segments(); ++i) {
            sum += math::area_frustrum(lengths_[i], radii_[i], radii_[i+1]);
        }
        return sum;
    }

    bool has_locations() const {
        return locations_.size() > 0;
    }

    // the number sub-segments that define the cable segment
    int num_sub_segments() const {
        return radii_.size()-1;
    }

    std::vector<value_type> const& lengths() const
    {
        return lengths_;
    }

    cable_segment* as_cable() override
    {
        return this;
    }

    private :

    void update_lengths()
    {
        if(locations_.size()) {
            lengths_.resize(num_sub_segments());
            for(auto i=0; i<num_sub_segments(); ++i) {
                lengths_[i] = norm(locations_[i] - locations_[i+1]);
            }
        }
    }

    std::vector<value_type> lengths_;
    std::vector<value_type> radii_;
    std::vector<point_type> locations_;
};

using segment_ptr = std::unique_ptr<segment>;

template <typename T, typename... Args>
segment_ptr make_segment(Args&&... args) {
    return segment_ptr(new T(std::forward<Args>(args)...));
}

} // namespace nestmc

