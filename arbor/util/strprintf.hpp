#pragma once

// printf-like routines that return std::string.
//
// TODO: Consolidate with a single routine that provides a consistent interface
// along the lines of the PO645R2 text formatting proposal.

#include <cstdio>
#include <memory>
#include <string>
#include <sstream>
#include <system_error>
#include <type_traits>
#include <utility>
#include <vector>

#include "util/meta.hpp"

namespace arb {
namespace util {

// Use ADL to_string or std::to_string, falling back to ostream formatting:

namespace impl_to_string {
    using std::to_string;

    template <typename T, typename = void>
    struct select {
        static std::string str(const T& value) {
            std::ostringstream o;
            o << value;
            return o.str();
        }
    };

    template <typename T>
    struct select<T, std::void_t<decltype(to_string(std::declval<T>()))>> {
        static std::string str(const T& v) {
            return to_string(v);
        }
    };
}

template <typename T>
std::string to_string(const T& value) {
    return impl_to_string::select<T>::str(value);
}

// Use snprintf to format a string, with special handling for standard
// smart pointer types and strings.

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

// Substitute instances of '{}' in the format string with the following parameters,
// using default std::ostream formatting.

namespace impl {
    inline void pprintf_(std::ostringstream& o, const char* s) {
        o << s;
    }

    template <typename T, typename... Tail>
    void pprintf_(std::ostringstream& o, const char* s, T&& value, Tail&&... tail) {
        const char* t = s;
        while (*t && !(*t=='{' && t[1]=='}')) {
            ++t;
        }
        o.write(s, t-s);
        if (*t) {
            o << std::forward<T>(value);
            pprintf_(o, t+2, std::forward<Tail>(tail)...);
        }
    }
}

template <typename... Args>
std::string pprintf(const char *s, Args&&... args) {
    std::ostringstream o;
    impl::pprintf_(o, s, std::forward<Args>(args)...);
    return o.str();
}

} // namespace util
} // namespace arb
