#pragma once

#include <exception>
#include <memory>
#include <sstream>
#include <vector>
#include <initializer_list>

namespace impl {
    template <typename C, typename V>
    struct has_count_method {
        template <typename T, typename U>
        static decltype(std::declval<T>().count(std::declval<U>()), std::true_type{}) test(int);
        template <typename T, typename U>
        static std::false_type test(...);

        using type = decltype(test<C, V>(0));
    };

    template <typename X, typename C>
    bool is_in(const X& x, const C& c, std::false_type) {
        for (const auto& y: c) {
            if (y==x) return true;
        }
        return false;
    }

    template <typename X, typename C>
    bool is_in(const X& x, const C& c, std::true_type) {
        return !!c.count(x);
    }
}

template <typename X, typename C>
bool is_in(const X& x, const C& c) {
    return impl::is_in(x, c, typename impl::has_count_method<C,X>::type{});
}

template <typename X>
bool is_in(const X& x, const std::initializer_list<X>& c) {
    return impl::is_in(x, c, std::false_type{});
}

struct enum_hash {
    template <typename E, typename V = typename std::underlying_type<E>::type>
    std::size_t operator()(E e) const noexcept {
        return std::hash<V>{}(static_cast<V>(e));
    }
};

inline std::string pprintf(const char *s) {
    std::string errstring;
    while(*s) {
        if(*s == '%' && s[1]!='%') {
            // instead of throwing an exception, replace with ??
            //throw std::runtime_error("pprintf: the number of arguments did not match the format ");
            errstring += "<?>";
        }
        else {
            errstring += *s;
        }
        ++s;
    }
    return errstring;
}

// variadic printf for easy error messages
template <typename T, typename ... Args>
std::string pprintf(const char *s, T value, Args... args) {
    std::string errstring;
    while(*s) {
        if(*s == '%' && s[1]!='%') {
            std::stringstream str;
            str << value;
            errstring += str.str();
            errstring += pprintf(++s, args...);
            return errstring;
        }
        else {
            errstring += *s;
            ++s;
        }
    }
    return errstring;
}

template <typename T>
std::string to_string(T val) {
    std::stringstream str;
    str << val;
    return str.str();
}

//'\e[1;31m' # Red
//'\e[1;32m' # Green
//'\e[1;33m' # Yellow
//'\e[1;34m' # Blue
//'\e[1;35m' # Purple
//'\e[1;36m' # Cyan
//'\e[1;37m' # White
enum class stringColor {white, red, green, blue, yellow, purple, cyan};

#define COLOR_PRINTING
#ifdef COLOR_PRINTING
inline std::string colorize(std::string const& s, stringColor c) {
    switch(c) {
        case stringColor::white :
            return "\033[1;37m"  + s + "\033[0m";
        case stringColor::red   :
            return "\033[1;31m" + s + "\033[0m";
        case stringColor::green :
            return "\033[1;32m" + s + "\033[0m";
        case stringColor::blue  :
            return "\033[1;34m" + s + "\033[0m";
        case stringColor::yellow:
            return "\033[1;33m" + s + "\033[0m";
        case stringColor::purple:
            return "\033[1;35m" + s + "\033[0m";
        case stringColor::cyan  :
            return "\033[1;36m" + s + "\033[0m";
    }
    return s;
}
#else
inline std::string colorize(std::string const& s, stringColor c) {
    return s;
}
#endif

// helpers for inline printing
inline std::string red(std::string const& s) {
    return colorize(s, stringColor::red);
}
inline std::string green(std::string const& s) {
    return colorize(s, stringColor::green);
}
inline std::string yellow(std::string const& s) {
    return colorize(s, stringColor::yellow);
}
inline std::string blue(std::string const& s) {
    return colorize(s, stringColor::blue);
}
inline std::string purple(std::string const& s) {
    return colorize(s, stringColor::purple);
}
inline std::string cyan(std::string const& s) {
    return colorize(s, stringColor::cyan);
}
inline std::string white(std::string const& s) {
    return colorize(s, stringColor::white);
}

template <typename T>
std::ostream& operator<< (std::ostream& os, std::vector<T> const& V) {
    os << "[";
    for(auto it = V.begin(); it!=V.end(); ++it) { // ugly loop, pretty printing
        os << *it << (it==V.end()-1 ? "" : " ");
    }
    return os << "]";
}

template <typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args) ...));
}

