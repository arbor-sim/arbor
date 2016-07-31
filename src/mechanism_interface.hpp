#pragma once

// just for compatibility with current version of modparser...

#include "mechanism.hpp"
#include "parameter_list.hpp"

namespace nest {
namespace mc {
namespace mechanisms {

template <typename T, typename I>
struct mechanism_helper {
    using value_type = T;
    using size_type = I;
    using index_type = memory::HostVector<I>;
    using index_view = typename index_type::view_type;
    using mechanism_ptr_type = mechanism_ptr<T, I>;
    using view_type = typename mechanism<T,I>::view_type;

    virtual std::string name() const = 0;
    virtual mechanism_ptr<T,I> new_mechanism(view_type, view_type, index_view) const = 0;
    virtual void set_parameters(mechanism_ptr_type&, parameter_list const&) const = 0;
};

} // namespace mechanisms
} // namespace mc
} // namespace nest
