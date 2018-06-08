#pragma once

#include <communication/distributed_context.hpp>

namespace arb {

distributed_context mpi_context();

template <typename MPICommType>
distributed_context mpi_context(MPICommType);

} // namespace arb

