#pragma once

#include "../vector/include/Vector.hpp"

namespace nest {
namespace mc {

/// matrix storage structure
template<typename T, typename I>
class matrix {
    public:

    // define basic types
    using value_type = T;
    using size_type  = I;

    // define storage types
    using vector_type = memory::HostVector<value_type>;
    using view_type   = typename vector_type::view_type;
    using index_type  = memory::HostVector<size_type>;
    using index_view  = typename index_type::view_type;


    /// the dimension of the matrix (i.e. the number of rows or colums)
    size_type size() const
    {
        return parent_indices_.size();
    }

    /// the total memory used to store the matrix
    std::size_t memory() const
    {
        auto s = sizeof(value_type) * data_.size();
        s     += sizeof(size_type)  * parent_indices_.size();
        return s + sizeof(matrix);
    }

    /// the number of cell matrices that have been packed together
    size_type num_cells() const
    {
        return cell_index_.size();
    }

    private:

    /// calculate the number of values per sub-array plus padding
    size_type reservation() const
    {
        constexpr auto alignment = vector_type::coordinator_type::alignment();
        auto const padding = memory::impl::get_padding<value_type>(alignment, size());
        return size() + padding;
    }

    /// subdivide the memory in data_ into the sub-arrays that store the matrix
    /// diagonals and rhs/solution vectors
    void setup()
    {
        const auto n = size();
        const auto n_alloc = reservation();

        // point sub-vectors into storage
        a_   = data_(        0,   n);
        d_   = data_(  n_alloc,   n_alloc + n);
        b_   = data_(2*n_alloc, 2*n_alloc + n);
        rhs_ = data_(3*n_alloc, 3*n_alloc + n);
        v_   = data_(4*n_alloc, 4*n_alloc + n);
    }

    // the parent indices are stored as an index array
    index_type parent_indices_;
    index_type cell_index_;

    size_type num_cells_;

    // all the data fields point into a single block of storage
    vector_type data_;

    // the individual data files that point into block storage
    view_type a_;
    view_type d_;
    view_type b_;
    view_type rhs_;
    view_type v_;
};

} // namespace nest
} // namespace mc
