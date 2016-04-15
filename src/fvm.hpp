#pragma once

#include <algorithm>

#include <cell.hpp>
#include <segment.hpp>
#include <math.hpp>
#include <matrix.hpp>
#include <util.hpp>
#include <algorithms.hpp>

#include <vector/include/Vector.hpp>

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

    using matrix_type = matrix<value_type, size_type>;

    /// the container used for indexes
    using index_type = memory::HostVector<size_type>;
    /// view into index container
    using index_view = typename index_type::view_type;

    /// the container used for values
    using vector_type = memory::HostVector<value_type>;
    /// view into value container
    using vector_view = typename vector_type::view_type;

    /// constructor
    fvm_cell(nest::mc::cell const& cell);

    /// build the matrix for a given time step
    void setup_matrx(value_type dt);

    matrix_type& matrix()
    {
        return matrix_;
    }

    private:

    /// the linear system for implicit time stepping of cell state
    matrix_type matrix_;

    /// cv_areas_[i] is the surface area of CV i
    vector_type cv_areas_;

    /// alpha_[i] is the following value at the CV face between
    /// CV i and its parent, required when constructing linear system
    ///     alpha_[i] = area_face  / (c_m * r_L * delta_x);
    vector_type alpha_;
};

////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////

template <typename T, typename I>
fvm_cell<T, I>::fvm_cell(nest::mc::cell const& cell)
:   matrix_(cell.parent_index())
,   cv_areas_(matrix_.size())
,   alpha_(matrix_.size())
{
    using util::left;
    using util::right;

    auto parent_index = matrix_.p();
    auto const& segment_index = cell.segment_index();

    // Use the membrane parameters for the first segment everywhere
    // in the cell to start with.
    // This has to be extended to use compartment/segment specific
    // membrane properties.
    auto membrane_params = cell.segments()[0]->mechanism("membrane");
    auto c_m = membrane_params.get("c_m").value;
    auto r_L = membrane_params.get("r_L").value;

    auto seg_idx = 0;
    for(auto const& s : cell.segments()) {
        if(auto soma = s->as_soma()) {
            // assert the assumption that the soma is at 0
            if(seg_idx!=0) {
                throw std::domain_error(
                        "FVM lowering encountered soma with non-zero index"
                );
            }
            cv_areas_[0] += math::area_sphere(soma->radius());
        }
        else if(auto cable = s->as_cable()) {
            // loop over each compartment in the cable
            // each compartment has the face between two CVs at its centre
            // the centers of the CVs are the end points of the compartment
            //
            //  __________________________________
            //  | ........ | .cvleft. |    cv    |
            //  | ........ L ........ C          R
            //  |__________|__________|__________|
            //
            //  The compartment has end points marked L and R (left and right).
            //  The left compartment is assumed to be closer to the soma
            //  (i.e. it follows the minimal degree ordering)
            //  The face is at the center, marked C.
            //  The full control volume to the left (marked with .)
            for(auto c : cable->compartments()) {
                auto i = segment_index[seg_idx] + c.index;
                auto j = parent_index[i];

                auto radius_center = math::mean(c.radius);
                auto area_face = math::area_circle( radius_center );
                alpha_[i] = area_face  / (c_m * r_L * c.length);

                auto halflen = c.length/2;

                auto al = math::area_frustrum(halflen, left(c.radius), radius_center);
                auto ar = math::area_frustrum(halflen, right(c.radius), radius_center);
                cv_areas_[j] += al;
                cv_areas_[i] += ar;
            }
        }
        else {
            throw std::domain_error("FVM lowering encountered unsuported segment type");
        }
        ++seg_idx;
    }
}

template <typename T, typename I>
void fvm_cell<T, I>::setup_matrx(T dt)
{
    using memory::all;

    // convenience accesors to matrix storage
    auto l = matrix_.l();
    auto d = matrix_.d();
    auto u = matrix_.u();
    auto p = matrix_.p();

    //  The matrix has the following layout in memory
    //  where j is the parent index of i, i.e. i<j
    //
    //      d[i] is the diagonal entry at a_ii
    //      u[i] is the upper triangle entry at a_ji
    //      l[i] is the lower triangle entry at a_ij
    //
    //       d[j] . . u[i]
    //        .  .     .
    //        .     .  .
    //       l[i] . . d[i]
    //

    //d(all) = cv_areas_ + dt*(alpha_ + alpha_(p));
    //d[0]  = cv_areas_[0];

    d(all) = cv_areas_;
    for(auto i=1; i<d.size(); ++i) {
        auto a = dt * alpha_[i];

        d[i] +=  a;
        l[i]  = -a;
        u[i]  = -a;

        // add contribution to the diagonal of parent
        d[p[i]] += a;
    }
}

} // namespace fvm
} // namespace mc
} // namespace nest
