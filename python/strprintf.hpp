#pragma once

// printf-like routines that return std::string.

#include <optional>
#include <string>
#include <sstream>
#include <type_traits>
#include <utility>

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <fmt/core.h>

namespace pyarb {
namespace util {

namespace impl {
    // Wrapper for formatted output of optional values.
    template <typename T>
    struct opt_wrap {
        const T& ref;
        opt_wrap(const T& ref): ref(ref) {}
        friend std::ostream& operator<<(std::ostream& out, const opt_wrap& wrap) {
            return out << wrap.ref;
        }
    };

    template <typename T>
    struct opt_wrap<std::optional<T>> {
        const std::optional<T>& ref;
        opt_wrap(const std::optional<T>& ref): ref(ref) {}
        friend std::ostream& operator<<(std::ostream& out, const opt_wrap& wrap) {
            if (wrap.ref) {
                return out << *wrap.ref;
            }
            else {
                return out << "None";
            }
        }
    };
}

// Use ADL to_string or std::to_string, falling back to ostream formatting:

namespace impl_to_string {
    using std::to_string;

    template <typename T, typename = void>
    struct select {
        static std::string str(const T& value) {
            std::ostringstream o;
            o << impl::opt_wrap(value);
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

template <typename T>
std::string to_string(const std::vector<T>& vs) {
    std::string res = "[";
    for (const auto& v: vs) res += to_string(v) + ", ";
    res += "]";
    return res;
}

template <typename K, typename V>
std::string to_string(const std::unordered_map<K, V>& vs) {
    std::string res = "{";
    for (const auto& [k, v]: vs) res += to_string(v) + ": " + to_string(v) + ", ";
    res += "}";
    return res;
}

} // namespace util
} // namespace pyarb
