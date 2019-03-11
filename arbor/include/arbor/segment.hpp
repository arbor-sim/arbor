#pragma once

#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <arbor/assert.hpp>
#include <arbor/common_types.hpp>
#include <arbor/math.hpp>
#include <arbor/morphology.hpp>
#include <arbor/mechinfo.hpp>
#include <arbor/point.hpp>
#include <arbor/util/optional.hpp>

namespace arb {

// Mechanism information attached to a segment.

struct mechanism_desc {
    struct field_proxy {
        mechanism_desc* m;
        std::string key;

        field_proxy& operator=(double v) {
            m->set(key, v);
            return *this;
        }

        operator double() const {
            return m->get(key);
        }
    };

    // implicit
    mechanism_desc(std::string name): name_(std::move(name)) {}
    mechanism_desc(const char* name): name_(name) {}

    mechanism_desc& set(const std::string& key, double value) {
        param_[key] = value;
        return *this;
    }

    double operator[](const std::string& key) const {
        return get(key);
    }

    field_proxy operator[](const std::string& key) {
        return {this, key};
    }

    double get(const std::string& key) const {
        auto i = param_.find(key);
        if (i==param_.end()) {
            throw std::out_of_range("no field "+key+" set");
        }
        return i->second;
    }

    const std::unordered_map<std::string, double>& values() const {
        return param_;
    }

    const std::string& name() const { return name_; }

private:
    std::string name_;
    std::unordered_map<std::string, double> param_;
};

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

    // Approximate frequency-dependent length constant lower bound.
    virtual value_type length_constant(value_type freq_Hz) const {
        return 0;
    }

    util::optional<mechanism_desc&> mechanism(const std::string& name) {
        auto it = std::find_if(mechanisms_.begin(), mechanisms_.end(),
            [&](mechanism_desc& m) { return m.name()==name; });
        return it==mechanisms_.end()? util::nullopt: util::just(*it);
    }

    void add_mechanism(mechanism_desc mech) {
        auto m = mechanism(mech.name());
        if (m) {
            *m = std::move(mech);
        }
        else {
            mechanisms_.push_back(std::move(mech));
        }
    }

    const std::vector<mechanism_desc>& mechanisms() {
        return mechanisms_;
    }

    const std::vector<mechanism_desc>& mechanisms() const {
        return mechanisms_;
    }

    // common electrical properties
    value_type rL = 100.0;   // resistivity [Ohm.cm]
    value_type cm = 0.01;    // capacitance [F/m^2] : 10 nF/mm^2 = 0.01 F/m^2

protected:
    segment(section_kind kind): kind_(kind) {}

    section_kind kind_;
    std::vector<mechanism_desc> mechanisms_;
};

class placeholder_segment : public segment {
public:
    placeholder_segment(): segment(section_kind::none) {}

    std::unique_ptr<segment> clone() const override {
        // use default copy constructor
        return std::make_unique<placeholder_segment>(*this);
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
        return std::make_unique<soma_segment>(*this);
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
        arb_assert(kind_==section_kind::dendrite || kind_==section_kind::axon);
    }

    cable_segment(section_kind k, value_type r1, value_type r2, value_type len):
        cable_segment{k, {r1, r2}, decltype(lengths_){len}}
    {}

    // constructor that lets the user describe the cable as a
    // seriew of radii and locations
    cable_segment(section_kind k, std::vector<value_type> r, std::vector<point_type> p):
        segment(k), radii_(std::move(r)), locations_(std::move(p))
    {
        arb_assert(kind_==section_kind::dendrite || kind_==section_kind::axon);
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
        return std::make_unique<cable_segment>(*this);
    }

    value_type length() const
    {
        return std::accumulate(lengths_.begin(), lengths_.end(), value_type{});
    }

    value_type length_constant(value_type freq_Hz) const override {
        // Following Hine and Carnevale (2001), "NEURON: A Tool for Neuroscientists",
        // Neuroscientist 7, pp. 123-135.
        //
        // λ(f) = approx. sqrt(diameter/(pi*f*rL*cm))/2.
        //
        // Pick smallest non-zero diameter in the segment.

        value_type r_min = 0;
        for (auto r: radii_) {
            if (r>0 && (r_min==0 || r<r_min)) r_min = r;
        }
        value_type d_min = r_min*2e-6; // [m]
        value_type rc = rL*0.01*cm;  // [s/m]
        value_type lambda = std::sqrt(d_min/(math::pi<double>*freq_Hz*rc))/2.; // [m]

        return lambda*1e6; // [µm]
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

using segment_ptr = std::unique_ptr<segment>;

/// Helper for constructing segments in a segment_ptr unique pointer wrapper.
/// Forwards the supplied arguments to construct a segment of type SegmentType.
/// e.g. auto my_cable = make_segment<cable>(section_kind::dendrite, ... );
template <typename SegmentType, typename... Args>
segment_ptr make_segment(Args&&... args) {
    return segment_ptr(new SegmentType(std::forward<Args>(args)...));
}

} // namespace arb
