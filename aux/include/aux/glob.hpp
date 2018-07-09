#pragma once

// glob (3) wrapper
// TODO: emulate for not-entirely-POSIX platforms.


#include <aux/path.hpp>

namespace aux {

std::vector<path> glob(const std::string& pattern);

} // namespace aux

