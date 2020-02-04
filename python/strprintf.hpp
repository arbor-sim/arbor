#pragma once

// printf-like routines that return std::string.

#include <cstdio>
#include <memory>
#include <string>
#include <sstream>
#include <system_error>
#include <unordered_map>
#include <utility>
#include <vector>

#include <arbor/util/optional.hpp>

namespace pyarb {
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

    // Can be eplaced with std::void_t in c++17.
    template <typename ...Args>
    using void_t = void;

    template <typename T>
    struct select<T, void_t<decltype(to_string(std::declval<T>()))>> {
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
        while (*t && !(t[0]=='{' && t[1]=='}')) {
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

namespace impl {

    template <typename Seq>
    struct sepval {
        const Seq& seq_;
        const char* sep_;

        sepval(const Seq& seq, const char* sep): seq_(seq), sep_(sep) {}

        friend std::ostream& operator<<(std::ostream& o, const sepval& s) {
            bool first = true;
            for (auto& x: s.seq_) {
                if (!first) o << s.sep_;
                first = false;
                o << x;
            }
            return o;
        }
    };

    template <typename Seq, typename F>
    struct sepval_transform {
        const Seq& seq_;
        const char* sep_;
        const F f_;

        sepval_transform(const Seq& seq, const char* sep, F&& f): seq_(seq), sep_(sep), f_(std::forward(f)) {}

        friend std::ostream& operator<<(std::ostream& o, const sepval_transform& s) {
            bool first = true;
            for (auto& x: s.seq_) {
                if (!first) o << s.sep_;
                first = false;
                o << s.f(x);
            }
            return o;
        }
    };

    template <typename Seq>
    struct sepval_lim {
        const Seq& seq_;
        const char* sep_;
        unsigned count_;

        sepval_lim(const Seq& seq, const char* sep, unsigned count): seq_(seq), sep_(sep), count_(count) {}

        friend std::ostream& operator<<(std::ostream& o, const sepval_lim& s) {
            bool first = true;
            unsigned n = s.count_;
            for (auto& x: s.seq_) {
                if (!first) {
                    o << s.sep_;
                }
                first = false;
                if (!n) {
                    return o << "...";
                }
                --n;
                o << x;
            }
            return o;
        }
    };
}

template <typename Seq>
impl::sepval<Seq> sepval(const char* sep, const Seq& seq) {
    return impl::sepval<Seq>(seq, sep);
}

template <typename Seq>
impl::sepval_lim<Seq> sepval(const char* sep, const Seq& seq, unsigned n) {
    return impl::sepval_lim<Seq>(seq, sep, n);
}

template <typename Seq>
impl::sepval<Seq> csv(const Seq& seq) {
    return impl::sepval<Seq>(seq, ", ");
}

template <typename Seq>
impl::sepval_lim<Seq> csv(const Seq& seq, unsigned n) {
    return impl::sepval_lim<Seq>(seq, ", ", n);
}

// Print dictionary: this could be done easily with range adaptors in C++17
template <typename Key, typename T>
std::string dictionary_csv(const std::unordered_map<Key, T>& dict) {
    constexpr bool string_key = std::is_same<std::string, std::decay_t<Key>>::value;
    std::string s = "{";
    bool first = true;
    for (auto& p: dict) {
        if (!first) s += ", ";
        first = false;
        s += pprintf(string_key? "'{}': {}": "{}: {}", p.first, p.second);
    }
    s += "}";
    return s;
}

} // namespace util

template <typename T>
std::ostream& operator<<(std::ostream& o, const arb::util::optional<T>& x) {
    return o << (x? util::to_string(*x): "None");
}

} // namespace pyarb

