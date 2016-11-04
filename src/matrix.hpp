#pragma once

#include <type_traits>

#include <memory/memory.hpp>
#include <backends/matrix.hpp>

#include <util/debug.hpp>
#include <util/span.hpp>

namespace nest {
namespace mc {

/// Hines matrix
/// the TargetPolicy defines the backend specific data types and solver
template<class SolverPolicy>
class matrix: public SolverPolicy {
public:
    using solver_policy = SolverPolicy;

    // define basic types
    using typename solver_policy::value_type;
    using typename solver_policy::size_type;

    // define storage types
    using typename solver_policy::array;
    using typename solver_policy::iarray;

    using typename solver_policy::view;
    using typename solver_policy::iview;
    using typename solver_policy::const_view;
    using typename solver_policy::const_iview;

    using typename solver_policy::host_array;

    matrix() = default;

    /// construct matrix for one or more cells, with combined parent index
    /// and a cell index
    matrix(const std::vector<size_type>& pi, const std::vector<size_type>& ci):
        parent_index_(memory::make_const_view(pi)),
        cell_index_(memory::make_const_view(ci))
    {
        setup();
    }

    /// construct matrix for a single cell described by a parent index
    matrix(const std::vector<size_type>& pi):
        parent_index_(memory::make_const_view(pi)),
        cell_index_(2)
    {
        cell_index_[0] = 0;
        cell_index_[1] = parent_index_.size();
        setup();
    }

    /// the dimension of the matrix (i.e. the number of rows or colums)
    std::size_t size() const {
        return parent_index_.size();
    }

    /// the number of cell matrices that have been packed together
    size_type num_cells() const {
        return cell_index_.size() - 1;
    }

    /// the vector holding the diagonal of the matrix
    view d() { return d_; }
    const_view d() const { return d_; }

    /// the vector holding the upper part of the matrix
    view u() { return u_; }
    const_view u() const { return u_; }

    /// the vector holding the right hand side of the linear equation system
    view rhs() { return rhs_; }
    const_view rhs() const { return rhs_; }

    /// the vector holding the parent index
    const_iview p() const { return parent_index_; }

    /// the patrition of the parent index over the cells
    const_iview cell_index() const { return cell_index_; }

    /// Solve the linear system.
    /// Upon completion the solution is stored in the RHS storage, which can
    /// be accessed via rhs().
    void solve() {
        solver_policy::solve(d_, u_, rhs_, parent_index_, cell_index_);
    }

    /// Build the matrix and its right hand side
    /// This is forwarded to the back end
    /*
    void build_matrix(value_type dt) {
        base::build_matrix(
            d_, u_, rhs_,
            sigma_, alpha_d_, alpha_, voltage_, current_, cv_capacitance_, dt);
    }
    */

    private:

    /// Allocate memory for storing matrix and right hand side vector
    /// and build the face area contribution to the diagonal
    void setup() {
        const auto n = size();
        constexpr auto default_value = std::numeric_limits<value_type>::quiet_NaN();

        d_   = array(n, default_value);
        u_   = array(n, default_value);
        rhs_ = array(n, default_value);

        /*
        // construct the precomputed alpha_d array in host memory
        host_array alpha_d_tmp(n, 0);
        for(auto i: util::make_span(1u, n)) {
            alpha_d_tmp[i] += alpha_[i];

            // add contribution to the diagonal of parent
            alpha_d_tmp[parent_index_[i]] += alpha_[i];
        }
        // move or copy into final location (gpu->copy, host->move)
        alpha_d_ = std::move(alpha_d_tmp);
        */
    }

    /// the parent indice that describe matrix structure
    iarray parent_index_;

    /// indexes that point to the start of each cell in the index
    iarray cell_index_;

    /// storage for lower, diagonal, upper and rhs
    array d_;
    array u_;
    array rhs_;

    /// storage for components used to build the diagonals
    /*
    const_view alpha_;
    array alpha_d_;
    const_view sigma_;

    const_view cv_capacitance_;
    const_view current_;
    const_view voltage_;
    */
};

} // namespace nest
} // namespace mc
