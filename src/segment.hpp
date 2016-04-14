#pragma once

#include <cmath>

#include <vector>

#include "compartment.hpp"
#include "math.hpp"
#include "parameter_list.hpp"
#include "point.hpp"
#include "algorithms.hpp"
#include "util.hpp"

namespace nest {
namespace mc {

template <typename T,
          typename valid = typename std::is_floating_point<T>::type>
struct segment_properties {
    T rL = 180.0;   // resistivity [Ohm.cm]
    T cm = 0.01;    // capacitance [F/m^2] : 10 nF/mm^2 = 0.01 F/m^2
};

enum class segmentKind {
    soma,
    dendrite,
    axon,
    none
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

    virtual int num_compartments() const = 0;
    virtual void set_compartments(int) = 0;

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

    virtual bool is_placeholder() const
    {
        return false;
    }

    segment_properties<value_type> properties;

    void add_mechanism(parameter_list p)
    {
        auto it = std::find_if(
            mechanisms_.begin(), mechanisms_.end(),
            [&p](parameter_list const& l){return l.name() == p.name();}
        );
        if(it!=mechanisms_.end()) {
            throw std::out_of_range(
                "attempt to add a mechanism parameter set to a segment which has an existing mechanism with the same name"
            );
        }

        mechanisms_.push_back(std::move(p));
    }

    parameter_list& mechanism(std::string const& n)
    {
        auto it = std::find_if(
            mechanisms_.begin(), mechanisms_.end(),
            [&n](parameter_list const& l){return l.name() == n;}
        );
        if(it==mechanisms_.end()) {
            throw std::out_of_range(
                "attempt to access a parameter that is not defined in a segment"
            );
        }

        return *it;
    }

    protected:

    segmentKind kind_;
    std::vector<parameter_list> mechanisms_;
};

class placeholder_segment : public segment
{
    public:

    using base = segment;
    using base::kind_;
    using base::value_type;

    placeholder_segment()
    {
        kind_ = segmentKind::none;
    }

    value_type volume() const override
    {
        return std::numeric_limits<value_type>::quiet_NaN();
    }

    value_type area() const override
    {
        return std::numeric_limits<value_type>::quiet_NaN();
    }

    bool is_placeholder() const override
    {
        return true;
    }

    int num_compartments() const override
    {
        return 0;
    }

    virtual void set_compartments(int) override
    { }
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
        mechanisms_.push_back(membrane_parameters());
    }

    soma_segment(value_type r, point_type const &c)
    :   soma_segment(r)
    {
        center_ = c;
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

    /// soma has one and one only compartments
    int num_compartments() const override
    {
        return 1;
    }

    void set_compartments(int n) override
    { }

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

        // add default membrane parameters
        mechanisms_.push_back(membrane_parameters());
    }

    cable_segment(
        segmentKind k,
        value_type r1,
        value_type r2,
        value_type len
    )
    : cable_segment{k, std::vector<value_type>{r1, r2}, std::vector<value_type>{len}}
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

        // add default membrane parameters
        mechanisms_.push_back(membrane_parameters());
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

    value_type length() const
    {
        return algorithms::sum(lengths_);
    }

    bool has_locations() const
    {
        return locations_.size() > 0;
    }

    // the number sub-segments that define the cable segment
    int num_sub_segments() const
    {
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

    int num_compartments() const override
    {
        return num_compartments_;
    }

    void set_compartments(int n) override
    {
        if(n<1) {
            throw std::out_of_range(
                "number of compartments in a segment must be one or more"
            );
        }
        num_compartments_ = n;
    }

    value_type radius(value_type loc) const
    {
        if(loc>=1.) return radii_.back();
        if(loc<=0.) return radii_.front();

        auto len = length();
        value_type pos = loc*len;

        // This could be cached using a partial sum.
        // In fact a lot of this stuff can be cached if
        // we find ourselves having to do it over and over again.
        // The time to cache it might be when update_lengths() is called.
        auto sum = value_type(0);
        auto i=0;
        for(i=0; i<num_sub_segments(); ++i) {
            if(sum+lengths_[i]>pos) {
                break;
            }
            sum += lengths_[i];
        }

        auto rel = (len - sum)/lengths_[i];

        return rel*radii_[i] + (1.-rel)*radii_[i+1];
    }

    /// iterable range type for simple compartment representation
    compartment_range compartments() const
    {
        return {num_compartments(), radii_.front(), radii_.back(), length()};
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

    int num_compartments_ = 1;
    std::vector<value_type> lengths_;
    std::vector<value_type> radii_;
    std::vector<point_type> locations_;
};

/// Unique pointer wrapper for abstract segment base class
using segment_ptr = std::unique_ptr<segment>;

/// Helper for constructing segments in a segment_ptr unique pointer wrapper.
/// Forwards the supplied arguments to construct a segment of type SegmentType.
/// e.g. auto my_cable = make_segment<cable>(segmentKind::dendrite, ... );
template <typename SegmentType, typename... Args>
segment_ptr make_segment(Args&&... args)
{
    return segment_ptr(new SegmentType(std::forward<Args>(args)...));
}

} // namespace mc
} // namespace nest

