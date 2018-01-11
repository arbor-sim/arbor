#pragma once

#include <cmath>
#include <vector>

#include <algorithms.hpp>
#include <common_types.hpp>
#include <compartment.hpp>
#include <math.hpp>
#include <morphology.hpp>
#include <mechinfo.hpp>
#include <point.hpp>
#include <util/make_unique.hpp>

namespace arb {

// forward declarations of segment specializations
class soma_segment;
class cable_segment;

// abstract base class for a cell segment
class segment {
public:
    using value_type = double;
    using size_type = cell_local_size_type;
    using point_type = point<value_type>;

    // (Yet more motivation for a separate morphology description class!)
    virtual std::unique_ptr<segment> clone() const = 0;

    section_kind kind() const {
        return kind_;
    }

    bool is_soma() const
    {
        return kind_==section_kind::soma;
    }

    bool is_dendrite() const
    {
        return kind_==section_kind::dendrite;
    }

    bool is_axon() const
    {
        return kind_==section_kind::axon;
    }

    virtual size_type num_compartments() const = 0;
    virtual void set_compartments(size_type) = 0;

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

    virtual const cable_segment* as_cable() const
    {
        return nullptr;
    }

    virtual const soma_segment* as_soma() const
    {
        return nullptr;
    }

    virtual bool is_placeholder() const
    {
        return false;
    }

    util::optional<mechanism_spec&> mechanism(const std::string& name) {
        auto it = std::find_if(mechanisms_.begin(), mechanisms_.end(),
            [&](mechanism_spec& m) { return m.name()==name; });
        return it==mechanisms_.end()? util::nullopt: util::just(*it);
    }

    void add_mechanism(mechanism_spec mech) {
        auto m = mechanism(mech.name());
        if (m) {
            *m = std::move(mech);
        }
        else {
            mechanisms_.push_back(std::move(mech));
        }
    }

    const std::vector<mechanism_spec>& mechanisms() {
        return mechanisms_;
    }

    const std::vector<mechanism_spec>& mechanisms() const {
        return mechanisms_;
    }

    // common electrical properties
    value_type rL = 100.0;   // resistivity [Ohm.cm]
    value_type cm = 0.01;    // capacitance [F/m^2] : 10 nF/mm^2 = 0.01 F/m^2

protected:
    segment(section_kind kind): kind_(kind) {}

    section_kind kind_;
    std::vector<mechanism_spec> mechanisms_;
};

class placeholder_segment : public segment {
public:
    placeholder_segment(): segment(section_kind::none) {}

    std::unique_ptr<segment> clone() const override {
        // use default copy constructor
        return util::make_unique<placeholder_segment>(*this);
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

    size_type num_compartments() const override
    {
        return 0;
    }

    virtual void set_compartments(size_type) override {}
};

class soma_segment : public segment {
public:
    soma_segment() = delete;

    explicit soma_segment(value_type r, point_type c = point_type{}):
        segment(section_kind::soma), radius_{r}, center_(c) {}

    std::unique_ptr<segment> clone() const override {
        // use default copy constructor
        return util::make_unique<soma_segment>(*this);
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

    const soma_segment* as_soma() const override
    {
        return this;
    }

    /// soma has one and one only compartments
    size_type num_compartments() const override
    {
        return 1;
    }

    void set_compartments(size_type n) override {}

private :
    // store the center and radius of the soma
    // note that the center may be undefined
    value_type radius_;
    point_type center_;
};

class cable_segment : public segment {
public:
    using base = segment;
    using base::kind_;
    using base::value_type;
    using base::point_type;

    // delete the default constructor
    cable_segment() = delete;

    // constructors for a cable with no location information
    cable_segment(section_kind k, std::vector<value_type> r, std::vector<value_type> lens):
        segment(k), radii_(std::move(r)), lengths_(std::move(lens))
    {
        assert(kind_==section_kind::dendrite || kind_==section_kind::axon);
    }

    cable_segment(section_kind k, value_type r1, value_type r2, value_type len):
        //cable_segment{k, std::vector<value_type>{r1, r2}, std::vector<value_type>{len}}
        cable_segment{k, {r1, r2}, decltype(lengths_){len}}
    {}

    // constructor that lets the user describe the cable as a
    // seriew of radii and locations
    cable_segment(section_kind k, std::vector<value_type> r, std::vector<point_type> p):
        segment(k), radii_(std::move(r)), locations_(std::move(p))
    {
        assert(kind_==section_kind::dendrite || kind_==section_kind::axon);
        update_lengths();
    }

    // helper that lets user specify with the radius and location
    // of just the end points of the cable
    //  i.e.    describing the cable as a single frustrum
    cable_segment(
        section_kind k,
        value_type r1,
        value_type r2,
        point_type const& p1,
        point_type const& p2
    ):
        cable_segment(k, {r1, r2}, {p1, p2})
    {}

    std::unique_ptr<segment> clone() const override {
        // use default copy constructor
        return util::make_unique<cable_segment>(*this);
    }

    value_type volume() const override
    {
        auto sum = value_type{0};
        for (auto i=0u; i<num_sub_segments(); ++i) {
            sum += math::volume_frustrum(lengths_[i], radii_[i], radii_[i+1]);
        }
        return sum;
    }

    value_type area() const override
    {
        auto sum = value_type{0};
        for (auto i=0u; i<num_sub_segments(); ++i) {
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
    size_type num_sub_segments() const
    {
        return radii_.size()-1;
    }

    std::vector<value_type> const& lengths() const
    {
        return lengths_;
    }

    std::vector<value_type> const& radii() const
    {
        return radii_;
    }

    cable_segment* as_cable() override
    {
        return this;
    }

    const cable_segment* as_cable() const override
    {
        return this;
    }

    size_type num_compartments() const override
    {
        return num_compartments_;
    }

    void set_compartments(size_type n) override
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
        size_type i = 0;
        for (i = 0; i<num_sub_segments(); ++i) {
            if(sum+lengths_[i]>pos) {
                break;
            }
            sum += lengths_[i];
        }

        auto rel = (len - sum)/lengths_[i];

        return rel*radii_[i] + (1.-rel)*radii_[i+1];
    }

    /// iterable range type for simple compartment representation
    compartment_range<size_type, value_type> compartments() const
    {
        return make_compartment_range(num_compartments(), radii_.front(), radii_.back(), length());
    }

private:
    void update_lengths() {
        if (locations_.size()) {
            lengths_.resize(num_sub_segments());
            for (size_type i=0; i<num_sub_segments(); ++i) {
                lengths_[i] = norm(locations_[i] - locations_[i+1]);
            }
        }
    }

    size_type num_compartments_ = 1;
    std::vector<value_type> radii_;
    std::vector<value_type> lengths_;
    std::vector<point_type> locations_;
};

/// Unique pointer wrapper for abstract segment base class
using segment_ptr = std::unique_ptr<segment>;

/// Helper for constructing segments in a segment_ptr unique pointer wrapper.
/// Forwards the supplied arguments to construct a segment of type SegmentType.
/// e.g. auto my_cable = make_segment<cable>(section_kind::dendrite, ... );
template <typename SegmentType, typename... Args>
segment_ptr make_segment(Args&&... args) {
    return segment_ptr(new SegmentType(std::forward<Args>(args)...));
}

/// Divided compartment adaptors for cable segments

template <typename DivCompClass>
DivCompClass div_compartments(const cable_segment* cable, unsigned ncomp) {
    return DivCompClass(ncomp, cable->radii(), cable->lengths());
}

template <typename DivCompClass>
DivCompClass div_compartments(const cable_segment* cable) {
    return DivCompClass(cable->num_compartments(), cable->radii(), cable->lengths());
}

struct segment_location {
    segment_location(cell_lid_type s, double l):
        segment(s), position(l)
    {
        EXPECTS(position>=0. && position<=1.);
    }
    friend bool operator==(segment_location l, segment_location r) {
        return l.segment==r.segment && l.position==r.position;
    }
    cell_lid_type segment;
    double position;
};

} // namespace arb
