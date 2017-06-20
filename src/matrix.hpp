#pragma once

#include <type_traits>

#include <memory/memory.hpp>

#include <util/debug.hpp>
#include <util/span.hpp>

namespace nest {
namespace mc {

/// Hines matrix
/// Make the back end state implementation optional to allow for
/// testing different implementations in the same code.
template<class Backend, class State=typename Backend::matrix_state>
class matrix {
public:
    using backend = Backend;

    // define basic types
    using value_type = typename backend::value_type;
    using size_type = typename backend::size_type;

    // define storage types
    using array = typename backend::array;
    using iarray = typename backend::iarray;

    using const_view = typename backend::const_view;
    using const_iview = typename backend::const_iview;

    using host_array = typename backend::host_array;

    // back end specific storage for matrix state
    using state = State;

    matrix() = default;

    matrix( const std::vector<size_type>& pi,
            const std::vector<size_type>& ci,
            const std::vector<value_type>& cv_capacitance,
            const std::vector<value_type>& face_conductance):
        parent_index_(memory::make_const_view(pi)),
        cell_index_(memory::make_const_view(ci)),
        state_(pi, ci, cv_capacitance, face_conductance)
    {
        EXPECTS(cell_index_[num_cells()] == parent_index_.size());
    }

    /// the dimension of the matrix (i.e. the number of rows or colums)
    std::size_t size() const {
        return parent_index_.size();
    }

    /// the number of cell matrices that have been packed together
    size_type num_cells() const {
        return cell_index_.size() - 1;
    }

    /// the vector holding the parent index
    const_iview p() const { return parent_index_; }

    /// the partition of the parent index over the cells
    const_iview cell_index() const { return cell_index_; }

    /// Solve the linear system.
    void solve() {
        state_.solve();
    }

    /// Assemble the matrix for given dt
    void assemble(double dt, const_view voltage, const_view current) {
        state_.assemble(dt, voltage, current);
    }

    /// Get a view of the solution
    const_view solution() const {
        return state_.solution;
    }

    private:

    /// the parent indice that describe matrix structure
    iarray parent_index_;

    /// indexes that point to the start of each cell in the index
    iarray cell_index_;

public:
    // Provide via public interface to make testing much easier.
    // If you modify this directly without knowing what you are doing,
    // you get what you deserve.
    state state_;
};

} // namespace nest
} // namespace mc
