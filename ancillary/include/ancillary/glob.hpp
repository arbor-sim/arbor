#pragma once

// glob (3) wrapper
// TODO: emulate for not-entirely-POSIX platforms.


#include <ancillary/path.hpp>

namespace anc {

std::vector<path> glob(const std::string& pattern);

} // namespace anc

