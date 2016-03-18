#pragma once

#include <cmath>

#include <vector>

#include "point.hpp"

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

template <typename T>
T constexpr pi() {
    return 3.141592653589793238462643383279502;
}

template <typename T>
class compartment {
    using value_type = T;
    using point_type = point<value_type>;

    constexpr compartment()
    :   p1_{point_type()},
        p2_{point_type()},
        radius1_{std::numeric_limits<value_type>::quiet_NaN()},
        radius2_{std::numeric_limits<value_type>::quiet_NaN()}
    {}

    constexpr compartment(
        point_type const& p1,
        point_type const& p2,
        value_type r1,
        value_type r2
    )
    :   p1_{p1},
        p2_{p2},
        radius1_{r1},
        radius2_{r2}
    {}

    value_type length() const {
        return norm(p1_-p2_);
    }

    value_type area() const {
        return volume_frustrum(length(), radius1_, radius2_);
    }

    value_type volume() const {
        return volume_frustrum(length(), radius1_, radius2_);
    }

    constexpr point_type midpoint() const {
        return 0.5*(p1_+p2_);
    }

    constexpr point_type midradius() const {
        return 0.5*(radius1_+radius2_);
    }

    private :

    value_type area_frustrum(
        value_type L,
        value_type r1,
        value_type r2
    ) const
    {
        auto dr = r1 - r2;
        return pi<double>() * (r1+r2) * std::sqrt(L*L + dr*dr);
    }

    value_type volume_frustrum(
        value_type L,
        value_type r1,
        value_type r2
    ) const
    {
        auto meanr = (r1+r2) / 2.;
        return pi<double>() * meanr * meanr * L;
    }

    point_type p1_;
    point_type p2_;
    value_type radius1_;
    value_type radius2_;
};

class segment {

    private :

    double length_;
    double radius_start_;
    double radius_end_;
    std::vector<compartment> compartments_;
    segmentKind kind_;
};

class abstract_cell {
    abstract_cell() = default;

    private :
};

} // namespace nestmc 


