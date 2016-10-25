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
    template <
        typename LHS, typename RHS,
        typename = typename std::enable_if< memory::is_array<LHS>::value && memory::is_array<RHS>::value >
    >
    matrix(LHS&& pi, RHS&& ci) :
        parent_index_(std::forward<LHS>(pi)),
        cell_index_(std::forward<RHS>(ci))
    {
        setup();
    }

    /// construct matrix for a single cell described by a parent index
    template <
        typename IDX,
        typename = typename std::enable_if< memory::is_array<IDX>::value >
    >
    matrix(IDX&& pi) :
        parent_index_(memory::make_const_view(pi)),
        cell_index_(2)
    {
        cell_index_[0] = 0;
        cell_index_[1] = size();
        setup();
    }

    /// the dimension of the matrix (i.e. the number of rows or colums)
    std::size_t size() const {
        return parent_index_.size();
    }

    /// the total memory used to store the matrix
    std::size_t memory() const {
        auto s = 6 * (sizeof(value_type) * size() + sizeof(array));
        s     += sizeof(size_type) * (parent_index_.size() + cell_index_.size())
                + 2*sizeof(iarray);
        s     += sizeof(matrix);
        return s;
    }

    /// the number of cell matrices that have been packed together
    size_type num_cells() const {
        return cell_index_.size() - 1;
    }

    /// FIXME : make modparser use the correct accessors (l,d,u,rhs) instead of these
    view vec_d() { return d(); }
    const_view vec_d() const { return d(); }

    view vec_rhs() { return rhs(); }
    const_view vec_rhs() const { return rhs(); }

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
    iview p() { return parent_index_; }
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
