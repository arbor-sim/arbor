#pragma once

#include <string>

namespace nest {
namespace mc {
namespace util {

// Get the name of the host on which this process is running.
// Returns "unknown" if unable to determine hostname.
std::string hostname();

} // namespace util
} // namespace mc
} // namespace nest
