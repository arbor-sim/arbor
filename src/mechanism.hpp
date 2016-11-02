#pragma once

#include <algorithm>
#include <memory>
#include <string>
#include <util/meta.hpp>

#include <indexed_view.hpp>
#include <ion.hpp>
#include <parameter_list.hpp>
#include <util/make_unique.hpp>

namespace nest {
namespace mc {
namespace mechanisms {

enum class mechanismKind {point, density};

/// The mechanism type is templated on a memory policy type.
/// The only difference between the abstract definition of a mechanism on host
/// or gpu is the information is stored, and how it is accessed.
template <typename MemoryTraits>
class mechanism : MemoryTraits {
public:
    using memory_traits = MemoryTraits;

    using typename memory_traits::value_type;
    using typename memory_traits::size_type;

    // define storage types
    using typename memory_traits::array;
    using typename memory_traits::iarray;

    using typename memory_traits::view;
    using typename memory_traits::iview;

    using typename memory_traits::const_view;
    using typename memory_traits::const_iview;

    using indexed_view_type = indexed_view<memory_traits>;

    using ion_type = ion<memory_traits>;

    mechanism(view vec_v, view vec_i, iarray&& node_index):
        vec_v_(vec_v), vec_i_(vec_i), node_index_(std::move(node_index))
    {}

    std::size_t size() const {
        return node_index_.size();
    }

    const_iview node_index() const {
        return node_index_;
    }

    value_type voltage(size_type i) const {
        return vec_v_[node_index_[i]];
    }

    value_type current(size_type i) const {
        return vec_i_[node_index_[i]];
    }

    virtual void set_params(value_type t_, value_type dt_) = 0;
    virtual std::string name() const = 0;
    virtual std::size_t memory() const = 0;
    virtual void nrn_init()     = 0;
    virtual void nrn_state()    = 0;
    virtual void nrn_current()  = 0;
    virtual void net_receive(int, value_type) {};
    virtual bool uses_ion(ionKind) const = 0;
    virtual void set_ion(ionKind k, ion_type& i, const std::vector<size_type>& index) = 0;

    void set_areas(view area) {
        vec_area_ = area;
    }

    virtual mechanismKind kind() const = 0;

    view vec_v_;
    view vec_i_;
    iarray node_index_;
    view vec_area_;
};

template <class MemoryTraits>
using mechanism_ptr = std::unique_ptr<mechanism<MemoryTraits>>;

template <typename M>
auto make_mechanism(
    typename M::view  vec_v,
    typename M::view  vec_i,
    typename M::iarray&& node_indices)
-> decltype(util::make_unique<M>(vec_v, vec_i, std::move(node_indices)))
{
    return util::make_unique<M>(vec_v, vec_i, std::move(node_indices));
}

} // namespace mechanisms
} // namespace mc
} // namespace nest
