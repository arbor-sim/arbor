#pragma once

#include <string>

#include <arbor/arbexcept.hpp>

namespace arb {
namespace util {

namespace impl{
void* dl_get_symbol(const std::string& filename, const std::string& symbol);
} // namespace impl

struct dl_error: arbor_exception {
    dl_error(const std::string& msg): arbor_exception{msg} {}
};

// Find and return a symbol from a dynamic library with filename.
// Throws dl_error on error.
template<typename T>
T dl_get_symbol(const std::string& filename, const std::string& symbol) {
    return reinterpret_cast<T>(impl::dl_get_symbol(filename, symbol));
}

} // namespace util
} // namespace arbor
