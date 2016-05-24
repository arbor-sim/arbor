#pragma once

#include "vector/include/Vector.hpp"

#ifdef DEBUG
#define EXPECTS(expression) assert(expression)
#else
#define EXPECTS(expression)
#endif

/*
using memory::util::red;
using memory::util::yellow;
using memory::util::green;
using memory::util::white;
using memory::util::blue;
using memory::util::cyan;
*/

#include <memory>
#include <ostream>
#include <utility>
#include <vector>

template <typename T>
std::ostream&
operator << (std::ostream &o, std::vector<T>const& v)
{
    o << "[";
    for(auto const& i: v) {
        o << i << ",";
    }
    o << "]";
    return o;
}

template <typename T>
std::ostream& print(std::ostream &o, std::vector<T>const& v)
{
    o << "[";
    for(auto const& i: v) {
        o << i << ",";
    }
    o << "]";
    return o;
}

namespace nest {
namespace mc {
namespace util {

    // just because we aren't using C++14, doesn't mean we shouldn't go
    // without make_unique
    template <typename T, typename... Args>
    std::unique_ptr<T> make_unique(Args&&... args) {
        return std::unique_ptr<T>(new T(std::forward<Args>(args) ...));
    }

    /// helper for taking the first value in a std::pair
    template <typename L, typename R>
    L const& left(std::pair<L, R> const& p)
    {
        return p.first;
    }

    /// helper for taking the second value in a std::pair
    template <typename L, typename R>
    R const& right(std::pair<L, R> const& p)
    {
        return p.second;
    }

    template <typename T>
    struct is_std_vector : std::false_type {};

    template <typename T, typename C>
    struct is_std_vector<std::vector<T,C>> : std::true_type {};

    template <typename T>
    struct is_container :
        std::conditional<
             is_std_vector<typename std::decay<T>::type>::value ||
             memory::is_array<T>::value,
             std::true_type,
             std::false_type
        >::type
    {};

    // printf with variadic templates for simplified string creation
    // mostly used to simplify error string creation
    [[gnu::unused]] static
    std::string pprintf(const char *s) {
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

} // namespace util
} // namespace mc
} // namespace nest

