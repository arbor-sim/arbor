#pragma once

#include <iterator>
#include <utility>

#include <common_types.hpp>
#include <util/counter.hpp>
#include <util/span.hpp>
#include <util/transform.hpp>

namespace nest {
namespace mc {

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

} // namespace mc
} // namespace nest


