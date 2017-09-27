#pragma once

namespace arb {
namespace config {

// has_memory_measurement
//     Support for measuring total allocated memory.
//     * true:  calls to util::allocated_memory() will return valid results
//     * false: calls to util::allocated_memory() will return -1
//
// has_power_measurement
//     Support for measuring energy consumption.
//     Currently only on Cray XC30/40/50 systems.
//     * true:  calls to util::energy() will return valid results
//     * false: calls to util::energy() will return -1
//
// has_cuda
//     Has been compiled with CUDA back end support

#ifdef __linux__
constexpr bool has_memory_measurement = true;
#else
constexpr bool has_memory_measurement = false;
#endif

#ifdef ARB_HAVE_CRAY
constexpr bool has_power_measurement = true;
#else
constexpr bool has_power_measurement = false;
#endif

#ifdef ARB_HAVE_GPU
constexpr bool has_cuda = true;
#else
constexpr bool has_cuda = false;
#endif

} // namespace config
} // namespace arb
