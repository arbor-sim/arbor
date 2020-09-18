#pragma once

#include <string>

namespace arb::literals {

inline
std::string operator "" _lab(const char* s, std::size_t) {
    return std::string("\"") + s + "\"";
}

} // namespace arb
