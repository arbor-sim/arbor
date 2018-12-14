#pragma once

#include <cstdint>
#include <vector>

namespace sup {

// The list of logical processors for which the calling thread has affinity.
// If calling from the main thread at application start up, before
// attempting to change thread affinity, may produce unreliable
// results.
//  - beware thread pinning or custom job scheduler affinity
//    flags that assign threads to specific cores.
//
// Returns an empty vector if unable to determine the number of
// available cores.
std::vector<int> get_affinity();

} // namespace sup
