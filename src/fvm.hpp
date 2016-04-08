#pragma once

#include <algorithm>

#include "cell.hpp"
#include "segment.hpp"

#include "math.hpp"
#include "util.hpp"
#include "algorithms.hpp"

#include "../vector/include/Vector.hpp"

namespace nest {
namespace mc {
namespace fvm {

template <typename T, typename I>
class fvm_cell {
    public :

    /// the real number type
    using value_type = T;
    /// the integral index type
    using size_type  = I;

    /// the container used for indexes
    using index_type = memory::HostVector<size_type>;
    /// view into index container
    using index_view = typename index_type::view_type;

    fvm_cell(nest::mc::cell const& cell);
};

////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////

template <typename T, typename I>
fvm_cell<T, I>::fvm_cell(nest::mc::cell const& cell)
{
    auto const& parent_index = cell.parent_index();
    auto const& segment_index = cell.segment_index();
    auto num_fv = segment_index.back();

    // storage for the volume and surface area of the finite volumes
    std::vector<value_type> fv_areas(num_fv);
    std::vector<value_type> fv_volumes(num_fv);

    auto seg_idx = 0;
    for(auto const& s : cell.segments()) {
        if(auto soma = s->as_soma()) {
            // make the assumption that the soma is at 0
            if(seg_idx!=0) {
                throw std::domain_error(
                        "FVM lowering encountered soma with non-zero index"
                );
            }
            fv_volumes[0] += math::volume_sphere(soma->radius());
            fv_areas[0]   += math::area_sphere  (soma->radius());
        }
        else if(auto cable = s->as_cable()) {
            using util::left;
            using util::right;

            for(auto c : cable->compartments()) {
                auto rhs = segment_index[seg_idx] + c.index;
                auto lhs = parent_index[rhs];

                auto rad_C = math::mean(left(c.radius), right(c.radius));
                auto len = c.length/2;

                fv_volumes[lhs] += math::volume_frustrum(len, left(c.radius), rad_C);
                fv_areas[lhs]   += math::area_frustrum  (len, left(c.radius), rad_C);

                fv_volumes[rhs] += math::volume_frustrum(len, right(c.radius), rad_C);
                fv_areas[rhs]   += math::area_frustrum  (len, right(c.radius), rad_C);
            }
        }
        else {
            throw std::domain_error(
                    "FVM lowering encountered unsuported segment type"
            );
        }
        ++seg_idx;
    }

    //std::cout << "volumes " << fv_volumes << " : " << sum(fv_volumes) << "\n";
    std::cout << "areas   " << sum(fv_areas)   << ", " << cell.volume() << "\n";
    std::cout << "volumes " << sum(fv_volumes) << ", " << cell.area() << "\n";
}

} // namespace fvm
} // namespace mc
} // namespace nest
