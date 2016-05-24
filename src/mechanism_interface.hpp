#pragma once

#include <map>
#include <string>

#include "mechanism.hpp"
#include "parameter_list.hpp"

namespace nest {
namespace mc {
namespace mechanisms {

using value_type = double;
using index_type = int;

/// helper type for building mechanisms
/// the use of abstract base classes everywhere is a bit ugly
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

template <typename T, typename I>
using mechanism_helper_ptr =
    std::unique_ptr<mechanism_helper<T,I>>;

template <typename M>
mechanism_helper_ptr<typename M::value_type, typename M::size_type>
make_mechanism_helper()
{
    return util::make_unique<M>();
}

// for now use a global variable for the map of mechanism helpers
extern std::map<
    std::string,
    mechanism_helper_ptr<value_type, index_type>
> mechanism_helpers;

void setup_mechanism_helpers();

mechanism_helper_ptr<value_type, index_type>&
get_mechanism_helper(const std::string& name);

} // namespace mechanisms
} // namespace mc
} // namespace nest
