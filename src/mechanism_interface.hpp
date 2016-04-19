#pragma once

#include <map>
#include <string>

#include "matrix.hpp"
#include "mechanism.hpp"
#include "parameter_list.hpp"

namespace nest {
namespace mc {

///
template <typename T, typename I>
struct mechanism_helper {
    using index_type = memory::HostView<I>;
    using index_view = typename index_type::view_type;
    using mechanism_type = mechanism<T, I>;
    using matrix_type = typename mechanism_type::matrix_type;

    virtual std::string name() const = 0;

    virtual mechanism_type new_mechanism(matrix_type*, index_view) const = 0;

    virtual void set_parameters(mechanism_type&, parameter_list const&) const = 0;
};

extern std::map<std::string, mechanism_helper<double, int>> mechanism_helpers;
void setup_mechanism_helpers();

} // namespace nest
} // namespace mc
