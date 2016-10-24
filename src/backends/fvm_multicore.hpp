#pragma once

#include <mechanism_catalogue.hpp>
#include "matrix_multicore.hpp"

namespace nest {
namespace mc {
namespace multicore {

struct fvm_policy : memory_traits {
    using memory_traits = nest::mc::multicore::memory_traits;

    /// define matrix type
    using matrix_policy = nest::mc::multicore::matrix_policy;

    /// mechanism pointer type
    using mechanism_type =
        nest::mc::mechanisms::mechanism_ptr<memory_traits>;

    /// mechanism factory
    using mechanism_catalogue =
        nest::mc::mechanisms::catalogue<memory_traits>;

    /// helper function that converts containers into target specific view/rvalue
    template <typename U>
    auto on_target(U&& u) -> decltype(memory::on_host(std::forward<U>(u))) {
        return memory::on_host(std::forward<U>(u));
    }
};

} // namespace multicore
} // namespace mc
} // namespace nest

