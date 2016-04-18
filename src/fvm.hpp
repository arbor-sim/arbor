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
    void setup_matrix(value_type dt);

    /// TODO this should be const
    /// which requires const_view in the vector library
    matrix_type& jacobian();

    /// TODO this should be const
    /// return list of CV areas in :
    ///          um^2
    ///     1e-6.mm^2
    ///     1e-8.cm^2
    vector_view cv_areas();

    /// TODO this should be const
    /// return the capacitance of each CV surface
    /// note that this is the total capacitance, not per unit area
    /// which is equivalent to sigma_i * c_m
    vector_view cv_capacitance();

    std::size_t size() const
    {
        return matrix_.size();
    }

    private:

    /// the linear system for implicit time stepping of cell state
    matrix_type matrix_;

    /// cv_areas_[i] is the surface area of CV i
    vector_type cv_areas_;

    /// alpha_[i] is the following value at the CV face between
    /// CV i and its parent, required when constructing linear system
    ///     face_alpha_[i] = area_face  / (c_m * r_L * delta_x);
    vector_type face_alpha_;

    /// cv_capacitance_[i] is the capacitance of CV i per unit area (i.e. c_m)
    vector_type cv_capacitance_;

    /// the average current over the surface of each CV
    /// current_ = i_m - i_e
    /// so the total current over the surface of CV i is
    ///     current_[i] * cv_areas_
    vector_type current_;

    /// the potential in mV in each CV
    vector_type voltage_;

};

////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////

template <typename T, typename I>
fvm_cell<T, I>::fvm_cell(nest::mc::cell const& cell)
:   cv_areas_      {cell.num_compartments(), T(0)}
,   face_alpha_    {cell.num_compartments(), T(0)}
,   cv_capacitance_{cell.num_compartments(), T(0)}
,   current_       {cell.num_compartments(), T(0)}
,   voltage_       {cell.num_compartments(), T(0)}
{
    using util::left;
    using util::right;

    const auto graph = cell.model();
    matrix_ = matrix_type(graph.parent_index);
    auto parent_index = matrix_.p();
    auto const& segment_index = graph.segment_index;

    auto seg_idx = 0;
    for(auto const& s : cell.segments()) {
        if(auto soma = s->as_soma()) {
            // assert the assumption that the soma is at 0
            if(seg_idx!=0) {
                throw std::domain_error(
                        "FVM lowering encountered soma with non-zero index"
                );
            }
            auto area = math::area_sphere(soma->radius());
            cv_areas_[0] += area;
            cv_capacitance_[0] += area * soma->mechanism("membrane").get("c_m").value;
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
            auto c_m = cable->mechanism("membrane").get("c_m").value;
            auto r_L = cable->mechanism("membrane").get("r_L").value;
            for(auto c : cable->compartments()) {
                auto i = segment_index[seg_idx] + c.index;
                auto j = parent_index[i];

                auto radius_center = math::mean(c.radius);
                auto area_face = math::area_circle( radius_center );
                face_alpha_[i] = area_face  / (c_m * r_L * c.length);
                cv_capacitance_[i] = c_m;

                std::cout << "radius " << radius_center << ", c_m " << c_m << ", r_L " << r_L << ", dx " << c.length << "\n";

                auto halflen = c.length/2;

                auto al = math::area_frustrum(halflen, left(c.radius), radius_center);
                auto ar = math::area_frustrum(halflen, right(c.radius), radius_center);
                cv_areas_[j] += al;
                cv_areas_[i] += ar;
                cv_capacitance_[j] += al * c_m;
                cv_capacitance_[i] += ar * c_m;
            }
        }
        else {
            throw std::domain_error("FVM lowering encountered unsuported segment type");
        }
        ++seg_idx;
    }

    // normalize the capacitance by cv_area
    for(auto i=0u; i<size(); ++i) {
        cv_capacitance_[i] /= cv_areas_[i];
    }
}

template <typename T, typename I>
typename fvm_cell<T,I>::matrix_type&
fvm_cell<T,I>::jacobian()
{
    return matrix_;
}

template <typename T, typename I>
typename fvm_cell<T,I>::vector_view
fvm_cell<T,I>::cv_areas()
{
    return cv_areas_;
}

template <typename T, typename I>
typename fvm_cell<T,I>::vector_view
fvm_cell<T,I>::cv_capacitance()
{
    return cv_capacitance_;
}

template <typename T, typename I>
void fvm_cell<T, I>::setup_matrix(T dt)
{
    using memory::all;

    // convenience accesors to matrix storage
    auto l = matrix_.l();
    auto d = matrix_.d();
    auto u = matrix_.u();
    auto p = matrix_.p();
    auto rhs = matrix_.rhs();

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
    d(all) = cv_areas_;
    for(auto i=1u; i<d.size(); ++i) {
        auto a = dt * face_alpha_[i];

        d[i] +=  a;
        l[i]  = -a;
        u[i]  = -a;

        // add contribution to the diagonal of parent
        d[p[i]] += a;
    }

    // the RHS of the linear system is
    //      sigma_i * (V[i] - dt/cm*(im - ie))
    for(auto i=0u; i<d.size(); ++i) {
        rhs[i] = cv_areas_[i] * (voltage_[i] - dt/cv_capacitance_[i]*current_[i]);
    }
}

} // namespace fvm
} // namespace mc
} // namespace nest
