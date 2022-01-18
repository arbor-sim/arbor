#pragma once

#include <vector>

namespace arbenv {

// Attempt to determine number of available threads that can be run concurrently.
// Will return at least 1.

unsigned long thread_concurrency();

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

} // namespace arbenv
