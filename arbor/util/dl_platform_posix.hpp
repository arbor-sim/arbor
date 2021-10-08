#pragma once

#include <fstream>
#include <string>

#include <dlfcn.h>

#include <arbor/arbexcept.hpp>

#include "util/strprintf.hpp"

namespace arb {

struct dl_error: arbor_exception {
    dl_error(const std::string& msg): arbor_exception{msg} {}
};

struct dl_handle {
    void* dl = nullptr;
};

inline
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

template<typename T> inline
T dl_get_symbol(const dl_handle& handle, const std::string& symbol) {
    // Call once to clear errors not caused by us
    dlerror();
    auto result = dlsym(handle.dl, symbol.c_str());
    // dlsym mayb return NULL even if succeeding
    auto error = dlerror();
    if (error) {
        throw dl_error{util::pprintf("[POSIX] dl_get_symbol failed with: {}", error)};
    }
    return reinterpret_cast<T>(result);
}

inline
void dl_close(dl_handle& handle) {
    dlclose(handle.dl);
    handle.dl = nullptr;
}

}
