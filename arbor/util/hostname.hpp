#pragma once

#include <optional>
#include <string>

namespace arb {
namespace util {

// Get the name of the host on which this process is running.
std::optional<std::string> hostname();

} // namespace util
} // namespace arb
