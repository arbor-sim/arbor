#pragma once

// glob (3) wrapper
// TODO: emulate for not-entirely-POSIX platforms.


#include <sup/path.hpp>

namespace sup {

std::vector<path> glob(const std::string& pattern);

} // namespace sup

