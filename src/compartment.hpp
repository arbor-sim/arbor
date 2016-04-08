#pragma once

#include <iterator>
#include <utility>

namespace nest {
namespace mc {

/// Defines the simplest type of compartment
/// The compartment is a conic frustrum
struct compartment {
    using value_type = double;
    using size_type = int;
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

/// The simplest type of compartment iterator :
///     - divide a segment into n compartments of equal length
///     - assume that the radius varies linearly from one end of the segment
///       to the other
class compartment_iterator :
    public std::iterator<std::forward_iterator_tag, compartment>
{

    public:

    using base = std::iterator<std::forward_iterator_tag, compartment>;
    using size_type = base::value_type::size_type;
    using real_type = base::value_type::value_type;

    compartment_iterator() = delete;

    compartment_iterator(
        size_type idx,
        real_type len,
        real_type rad,
        real_type delta_rad
    )
    :   index_(idx),
        radius_(rad),
        delta_radius_(delta_rad),
        length_(len)
    { }

    compartment_iterator(compartment_iterator const& other) = default;

    compartment_iterator& operator++()
    {
        index_++;
        radius_ += delta_radius_;
        return *this;
    }

    compartment_iterator operator++(int)
    {
        compartment_iterator ret(*this);
        operator++();
        return ret;
    }

    compartment operator*() const
    {
        return
            compartment(
                index_, length_, radius_, radius_ + delta_radius_
            );
    }

    bool operator==(compartment_iterator const& other) const
    {
        return other.index_ == index_;
    }

    bool operator!=(compartment_iterator const& other) const
    {
        return !operator==(other);
    }

    private :

    size_type index_;
    real_type radius_;
    const real_type delta_radius_;
    const real_type length_;
};

class compartment_range {
    public:

    using size_type = compartment_iterator::size_type;
    using real_type = compartment_iterator::real_type;

    compartment_range(
        size_type num_compartments,
        real_type radius_L,
        real_type radius_R,
        real_type length
    )
    :   num_compartments_(num_compartments),
        radius_L_(radius_L),
        radius_R_(radius_R),
        length_(length)
    {}

    compartment_iterator begin() const
    {
        return {0, compartment_length(), radius_L_, delta_radius()};
    }

    compartment_iterator cbegin() const
    {
        return begin();
    }

    /// With 0-based indexing compartment number "num_compartments_" is
    /// one past the end
    compartment_iterator end() const
    {
        return {num_compartments_, 0, 0, 0};
    }

    compartment_iterator cend() const
    {
        return end();
    }

    real_type delta_radius() const
    {
        return (radius_R_ - radius_L_) / num_compartments_;
    }

    real_type compartment_length() const
    {
        return length_ / num_compartments_;
    }

    private:

    size_type num_compartments_;
    real_type radius_L_;
    real_type radius_R_;
    real_type length_;
};

} // namespace mc
} // namespace nest


