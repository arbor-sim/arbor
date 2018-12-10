#include <mpi.h>

#include <sup/gpu.hpp>

namespace sup {

// Currently a placeholder.
// Take the default gpu for serial simulations.
template <>
int find_private_gpu(MPI_Comm comm) {
    return default_gpu();
}

} // namespace sup
