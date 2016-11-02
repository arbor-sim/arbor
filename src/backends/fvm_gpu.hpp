#pragma once

#include "catalogue_gpu.hpp"
#include "matrix_gpu.hpp"

namespace nest {
namespace mc {
namespace gpu {

struct fvm_policy : public memory_traits {
    /// define matrix solver
    using matrix_solver = nest::mc::gpu::matrix_solver;

    /// define matrix builder
    using matrix_builder = nest::mc::gpu::fvm_matrix_builder;

    /// mechanism factory
    using mechanism_catalogue = nest::mc::gpu::catalogue;

    /// back end specific storage for mechanisms
    using mechanism_type = mechanism_catalogue::mechanism_ptr_type;

    /// back end specific storage for shared ion specie state
    using ion_type = mechanism_catalogue::ion_type;

    static std::string name() {
        return "gpu";
    }
};

} // namespace multicore
} // namespace mc
} // namespace nest

