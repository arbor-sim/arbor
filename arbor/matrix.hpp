#pragma once

#include <type_traits>

#include <arbor/assert.hpp>

#include <memory/memory.hpp>
#include <util/span.hpp>

#include <arbor/arb_types.hpp>

namespace arb {

/// Hines matrix
/// Make the back end state implementation optional to allow for
/// testing different implementations in the same code.
template<class Backend, class State=typename Backend::matrix_state>
class matrix {
public:
    using backend    = Backend;
    using array      = typename backend::array;
    using iarray     = typename backend::iarray;
    using const_view = const array&;
    using state      = State;                         // backend specific storage for matrix state

    matrix() = default;

    matrix(const std::vector<arb_index_type>& pi,
           const std::vector<arb_index_type>& ci,
           const std::vector<arb_value_type>& cv_capacitance,
           const std::vector<arb_value_type>& face_conductance,
           const std::vector<arb_value_type>& cv_area,
           const std::vector<arb_index_type>& cell_to_intdom):
        num_cells_{ci.size() - 1},
        state_(pi, ci, cv_capacitance, face_conductance, cv_area, cell_to_intdom)
    {
        arb_assert(cell_index()[num_cells()] == arb_index_type(parent_index().size()));
    }

    /// the dimension of the matrix (i.e. the number of rows or colums)
    std::size_t size() const { return state_.size(); }
    /// the number of cell matrices that have been packed together
    std::size_t num_cells() const { return num_cells_; }
    /// the vector holding the parent index
    const iarray& parent_index() const { return state_.parent_index; }
    /// the partition of the parent index over the cells
    const iarray& cell_index() const { return state_.cell_cv_divs; }
    /// Solve the linear system into a given solution storage.
    void solve(array& to) { state_.solve(to); }
    /// Assemble the matrix for given dt
    void assemble(const_view& dt, const_view& U, const_view& I, const_view& g) { state_.assemble(dt, U, I, g); }

private:
   std::size_t num_cells_ = 0;

public:
    // Provide via public interface to make testing much easier. If you modify
    // this directly without knowing what you are doing, you get what you
    // deserve.
    state state_;
};

} // namespace arb
