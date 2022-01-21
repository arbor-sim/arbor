#include <fstream>
#include <string>

#include <dlfcn.h>

#include <arbor/arbexcept.hpp>

#include "util/dylib.hpp"
#include "util/strprintf.hpp"

namespace arb {
namespace util {

dl_handle dl_open(const std::string& fn) {
    try {
        std::ifstream fd{fn.c_str()};
        if(!fd.good()) throw file_not_found_error{fn};
    } catch(...) {
        throw file_not_found_error{fn};
    }
    // Call once to clear errors not caused by us
    dlerror();
    auto result = dlopen(fn.c_str(), RTLD_LAZY);
    // dlopen fails by returning NULL
    if (nullptr == result) {
        auto error = dlerror();
        throw dl_error{util::pprintf("[POSIX] dl_open failed with: {}", error)};
    }
    return {result};
}

void* dl_get_symbol(const dl_handle& handle, const std::string& symbol) {
    //
    if (!handle.inner) {
        throw dl_error{"[POSIX] dl_get_symbol was passed an invalid handle."};
    }
    // Call once to clear errors not caused by us
    dlerror();
    // Get symbol from shared object, may return NULL if that is what symbol refers to
    auto result = dlsym(handle.inner, symbol.c_str());
    // dlsym mayb return NULL even if succeeding
    if (auto error = dlerror()) {
        throw dl_error{util::pprintf("[POSIX] dl_get_symbol failed with: {}", error)};
    }
    return result;
}

void dl_close(dl_handle& handle) {
    if (auto dl = handle.inner; dl != nullptr) {
        dlclose(dl);
    }
    handle.inner = nullptr;
}
} // namespace util
} // namespace arb
