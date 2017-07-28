#pragma once

#include <cstdint>
#include <vector>

#include <util/optional.hpp>

namespace nest {
namespace mc {
namespace hw {

// The list of cores for which the calling thread has affinity.
// If calling from the main thread at application start up, before
// attempting to change thread affinity, may produce unreliable
// results.
//  - beware thread pinning or custom job scheduler affinity
//    flags that assign threads to specific cores.
//
// Returns an empty vector if unable to determine the number of
// available cores.
std::vector<int> get_affinity();

// Attempts to find the number of cores available to the application
// This is likely to give inaccurate results if the caller has already
// been playing with thread affinity.
//
// Returns 0 if unable to determine the number of cores.
util::optional<std::size_t> num_cores();

} // namespace util
} // namespace mc
} // namespace nest
