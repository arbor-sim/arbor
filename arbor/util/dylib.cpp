#include <fstream>
#include <string>
#include <filesystem>

#include <arbor/arbexcept.hpp>

#include "util/dylib.hpp"
#include "util/strprintf.hpp"

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#else
#  include <dlfcn.h>
#endif

namespace arb {
namespace util {

void* dl_open(const std::filesystem::path& fn) {
    try {
        std::ifstream fd{fn.c_str()};
        if(!fd.good()) throw file_not_found_error{fn.string()};
    } catch(...) {
        throw file_not_found_error{fn.string()};
    }
#ifdef _WIN32
    auto result = reinterpret_cast<void*>(LoadLibraryW(fn.c_str()));
    if (nullptr == result) {
        throw dl_error{util::pprintf("[WIN32] dl_open failed with error: {}", GetLastError())};
    }
#else
    // Call once to clear errors not caused by us
    dlerror();
    auto result = dlopen(fn.c_str(), RTLD_LAZY);
    // dlopen fails by returning NULL
    if (nullptr == result) {
        auto error = dlerror();
        throw dl_error{util::pprintf("[POSIX] dl_open failed with: {}", error)};
    }
#endif
    return result;
}

namespace impl{
void* dl_get_symbol(const std::filesystem::path& fn, const std::string& symbol) {
    auto handle = dl_open(fn);
#ifdef _WIN32
    auto result = reinterpret_cast<void*>(GetProcAddress(reinterpret_cast<HMODULE>(handle), symbol.c_str()));
    if (!result) {
        throw dl_error{util::pprintf("[WIN32] dl_get_symbol failed with error: {}", GetLastError())};
    }
#else
    // Call once to clear errors not caused by us
    dlerror();
    // Get symbol from shared object, may return NULL if that is what symbol refers to
    auto result = dlsym(handle, symbol.c_str());
    // dlsym mayb return NULL even if succeeding
    if (auto error = dlerror()) {
        throw dl_error{util::pprintf("[POSIX] dl_get_symbol failed with: {}", error)};
    }
#endif
    return result;
}
} // namespace impl

} // namespace util
} // namespace arb
