#pragma once

#include <map>
#include <type_traits>

#include <memory/memory.hpp>

#include <util/debug.hpp>
#include <util/span.hpp>

namespace nest {
namespace mc {

/// Hines matrix
/// the TargetPolicy defines the backend specific data types and solver
template<class Backend>
class matrix {
public:
    using backend = Backend;

    // define basic types
    using value_type = typename backend::value_type;
    using size_type = typename backend::size_type;

    // define storage types
    using array = typename backend::array;
    using iarray = typename backend::iarray;

    using view = typename backend::view;
    using iview = typename backend::iview;
    using const_view = typename backend::const_view;
    using const_iview = typename backend::const_iview;

    using host_array = typename backend::host_array;

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
        backend::hines_solve(d_, u_, rhs_, parent_index_, cell_index_);
    }

    void print() const {
        auto n = size();
        std::vector<std::map<int, double>> U(n);

        for (auto i=0u; i<n; ++i) {
            U[i][i] = d_[i];
        }
        for (auto i=1u; i<n; ++i) {
            auto p = parent_index_[i];
            U[p][i] = u_[i];
        }

        for (auto i=0u; i<n; ++i) {
            printf (" [[%16.14f]]", rhs_[i]);
            for (auto uu: U[i]) {
                printf (" (%-4d %16.14f)", uu.first, uu.second);
            }
            printf("\n");
        }
    }

    private:

    /// Allocate memory for storing matrix and right hand side vector
    /// and build the face area contribution to the diagonal
    void setup() {
        const auto n = size();
        constexpr auto default_value = std::numeric_limits<value_type>::quiet_NaN();

        d_   = array(n, default_value);
        u_   = array(n, default_value);
        rhs_ = array(n, default_value);
    }

    /// the parent indice that describe matrix structure
    iarray parent_index_;

    /// indexes that point to the start of each cell in the index
    iarray cell_index_;

    /// storage for lower, diagonal, upper and rhs
    array d_;
    array u_;
    array rhs_;
};

} // namespace nest
} // namespace mc
