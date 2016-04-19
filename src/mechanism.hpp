#pragma once

#pragma once

#include <memory>
#include <string>

#include "matrix.hpp"
#include "parameter_list.hpp"
#include "util.hpp"

namespace nest {
namespace mc {

enum class mechanismKind {point, density};

template <typename T, typename I>
class mechanism {
    using value_type  = T;
    using size_type   = I;

    // define storage types
    using vector_type = memory::HostVector<value_type>;
    using view_type   = typename vector_type::view_type;
    using index_type  = memory::HostVector<size_type>;
    using index_view  = typename index_type::view_type;
    //using indexed_view= IndexedView<value_type, size_type, target()>;

    using matrix_type = matrix<value_type, size_type>;

    mechanism(matrix_type *matrix, index_view node_indices)
    :   matrix_(matrix)
    ,   node_indices_(node_indices)
    {}

    std::size_t size() const
    {
        return node_indices_.size();
    }

    virtual void set_params(value_type t_, value_type dt_) = 0;
    virtual std::string name() const = 0;
    virtual std::size_t memory() const = 0;
    virtual void init()     = 0;
    virtual void state()    = 0;
    virtual void current()  = 0;

    virtual mechanismKind kind() const = 0;

    matrix_type* matrix_;
    index_type node_indices_;
};

template <typename T, typename I>
using mechanism_ptr = std::unique_ptr<mechanism<T,I>>;

template <typename M>
mechanism_ptr<typename M::value_type, typename M::size_type>
make_mechanism(
    typename M::matrix_type* matrix,
    typename M::index_view   node_indices
) {
    return util::make_unique<M>(matrix, node_indices);
}

} // namespace nest
} // namespace mc
