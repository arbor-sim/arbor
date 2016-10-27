#pragma once

#include "catalogue_multicore.hpp"
#include "matrix_multicore.hpp"

namespace nest {
namespace mc {
namespace multicore {

struct fvm_policy : public memory_traits {
    /// define matrix type
    using matrix_policy = nest::mc::multicore::matrix_policy;

    /// mechanism factory
    using mechanism_catalogue = nest::mc::multicore::catalogue;

    /// back end specific storage for mechanisms
    using mechanism_type = mechanism_catalogue::mechanism_ptr_type;

    /// back end specific storage for shared ion specie state
    using ion_type = mechanism_catalogue::ion_type;
};

} // namespace multicore
} // namespace mc
} // namespace nest

