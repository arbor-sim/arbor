#pragma once

#pragma once

#include <memory>
#include <string>

#include "indexed_view.hpp"
#include "parameter_list.hpp"
#include "util.hpp"
#include "ion.hpp"

namespace nest {
namespace mc {
namespace mechanisms {

enum class mechanismKind {point, density};

template <typename T, typename I>
class mechanism {

public:

    using value_type  = T;
    using size_type   = I;

    // define storage types
    using vector_type = memory::HostVector<value_type>;
    using view_type   = typename vector_type::view_type;
    using index_type  = memory::HostVector<size_type>;
    using index_view  = typename index_type::view_type;
    using indexed_view_type = indexed_view<value_type, size_type>;

    using ion_type    = ion<value_type, size_type>;

    mechanism(view_type vec_v, view_type vec_i, index_view node_index)
    :   vec_v_(vec_v)
    ,   vec_i_(vec_i)
    ,   node_index_(node_index)
    {}

    std::size_t size() const
    {
        return node_index_.size();
    }

    index_view node_index() const
    {
        return node_index_;
    }

    value_type voltage(size_type i) const
    {
        return vec_v_[node_index_[i]];
    }

    value_type current(size_type i) const
    {
        return vec_i_[node_index_[i]];
    }

    virtual void set_params(value_type t_, value_type dt_) = 0;
    virtual std::string name() const = 0;
    virtual std::size_t memory() const = 0;
    virtual void nrn_init()     = 0;
    virtual void nrn_state()    = 0;
    virtual void nrn_current()  = 0;
    virtual bool uses_ion(ionKind) const = 0;
    virtual void set_ion(ionKind k, ion_type& i) = 0;

    virtual mechanismKind kind() const = 0;

    view_type vec_v_;
    view_type vec_i_;
    index_type node_index_;
};

template <typename T, typename I>
using mechanism_ptr = std::unique_ptr<mechanism<T,I>>;

template <typename M>
mechanism_ptr<typename M::value_type, typename M::size_type>
make_mechanism(
    typename M::view_type  vec_v,
    typename M::view_type  vec_i,
    typename M::index_view node_indices
) {
    return util::make_unique<M>(vec_v, vec_i, node_indices);
}

} // namespace mechanisms
} // namespace nest
} // namespace mc
