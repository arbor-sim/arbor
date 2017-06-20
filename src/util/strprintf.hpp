#pragma once

/* Thin wrapper around std::snprintf for sprintf-like formatting
 * to std::string. */

#include <cstdio>
#include <memory>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace nest {
namespace mc {
namespace util {

namespace impl {
    template <typename X>
    X sprintf_arg_translate(const X& x) { return x; }

    inline const char* sprintf_arg_translate(const std::string& x) { return x.c_str(); }

    template <typename T, typename Deleter>
    T* sprintf_arg_translate(const std::unique_ptr<T, Deleter>& x) { return x.get(); }

    template <typename T>
    T* sprintf_arg_translate(const std::shared_ptr<T>& x) { return x.get(); }
}

template <typename... Args>
std::string strprintf(const char* fmt, Args&&... args) {
    thread_local static std::vector<char> buffer(1024);

    for (;;) {
        int n = std::snprintf(buffer.data(), buffer.size(), fmt, impl::sprintf_arg_translate(std::forward<Args>(args))...);
        if (n<0) {
            throw std::system_error(errno, std::generic_category());
        }
        else if ((unsigned)n<buffer.size()) {
            return std::string(buffer.data(), n);
        }
        buffer.resize(2*n);
    }
}

template <typename... Args>
std::string strprintf(const std::string& fmt, Args&&... args) {
    return strprintf(fmt.c_str(), std::forward<Args>(args)...);
}

} // namespace util
} // namespace mc
} // namespace nest
