#pragma once

#include <ion.hpp>
#include <mechanism_catalogue.hpp>
#include "matrix_multicore.hpp"

namespace nest {
namespace mc {
namespace multicore {

struct fvm_policy : public memory_traits {
    /// define matrix type
    using matrix_policy = nest::mc::multicore::matrix_policy;

    // FIXME
    // the back end specific part of the catalogue shoud be in... the back end!
    // i.e. we shouldn't include the ion and mechanism_catalogue headers above
    //

    /// mechanism factory
    using mechanism_catalogue =
        nest::mc::mechanisms::catalogue<memory_traits>;

    // FIXME
    // the ion_type and mechanism_type should be provided by the catalogue
    //

    /// back end specific storage for ion channel information
    using ion_type =
        nest::mc::mechanisms::ion<memory_traits>;

    /// mechanism pointer type
    using mechanism_type =
        nest::mc::mechanisms::mechanism_ptr<memory_traits>;

};

} // namespace multicore
} // namespace mc
} // namespace nest

