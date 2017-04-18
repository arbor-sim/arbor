#pragma once

#include <string>

#include <util/optional.hpp>

namespace nest {
namespace mc {
namespace util {

// Get the name of the host on which this process is running.
util::optional<std::string> hostname();

} // namespace util
} // namespace mc
} // namespace nest
