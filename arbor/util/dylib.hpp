#pragma once

#include <string>

#include <arbor/arbexcept.hpp>

namespace arb {
namespace util {

struct dl_error: arbor_exception {
    dl_error(const std::string& msg): arbor_exception{msg} {}
};

// Opaque handle to a dynamic library
struct dl_handle {
    void* inner = nullptr;
};

// Open a file containing shared object library; throws file_not_found/dl_error on failure
dl_handle dl_open(const std::string&);

// Free handle
void dl_close(dl_handle&);

// Retrieve symbol; throws dl_error on failure
void* dl_get_symbol(const dl_handle&, const std::string& symbol);

} // namespace util
} // namespace arbor
