#pragma once

#include <type_traits>

#include <memory/memory.hpp>
#include <backends/matrix.hpp>

#include <util.hpp>
#include <util/debug.hpp>

namespace nest {
namespace mc {

/// Hines matrix
/// the TargetPolicy defines the backend specific data types and solver
template<class TargetPolicy>
class matrix: public TargetPolicy {
public:
    using base = TargetPolicy;

    // define basic types
    using typename base::value_type;
    using typename base::size_type;

    // define storage types
    using typename base::array;
    using typename base::iarray;

    using typename base::view;
    using typename base::iview;
    using typename base::const_view;
    using typename base::const_iview;

    matrix() = default;

    /// construct matrix for one or more cells, with combined parent index
    /// and a cell index
    matrix(const std::vector<size_type>& pi, const std::vector<size_type>& ci):
        parent_index_(memory::make_const_view(pi)),
        cell_index_(memory::make_const_view(pi))
    {
        setup();
        std::cout << "\n\nAAAA\n\n";
    }

    /// construct matrix for a single cell described by a parent index
    matrix(const std::vector<size_type>& pi) :
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

    /// the vector holding the lower part of the matrix
    view l() { return l_; }
    const_view l() const { return l_; }

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

    /// solve the linear system
    /// upon completion the solution is stored in the RHS storage
    /// and can be accessed via rhs()
    void solve() {
        base::solve(l_, d_, u_, rhs_, parent_index_, cell_index_);
    }

    private:

    /// allocate memory for storing matrix and right hand side vector
    void setup() {
        const auto n = size();
        constexpr auto default_value = std::numeric_limits<value_type>::quiet_NaN();

        l_   = array(n, default_value);
        d_   = array(n, default_value);
        u_   = array(n, default_value);
        rhs_ = array(n, default_value);
    }

    /// the parent indice that describe matrix structure
    iarray parent_index_;

    /// indexes that point to the start of each cell in the index
    iarray cell_index_;

    /// storage for lower, diagonal, upper and rhs
    array l_;
    array d_;
    array u_;

    /// after calling solve, the solution is stored in rhs_
    array rhs_;
};

} // namespace nest
} // namespace mc
