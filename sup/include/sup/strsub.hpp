#pragma once

// Substitute instances of a given character (defaults to '%') in a template C
// string with the remaining arguments, and write the result to an ostream or
// return the result as a string.
//
// The special character itself can be escaped by duplicating it, e.g.
//
//     strsub("%%%-%%%", 30, 70)
//
// returns the string
//
//     "%30-%70"

#include <iostream>
#include <sstream>
#include <string>
#include <utility>

namespace sup {

// Stream-writing strsub(...):

inline std::ostream& strsub(std::ostream& o, char c, const char* templ) {
    return o << templ;
}

template <typename T, typename... Tail>
std::ostream& strsub(std::ostream& o, char c, const char* templ, T value, Tail&&... tail) {
    const char* t = templ;
    for (;;) {
        while (*t && !(*t==c)) ++t;

        if (t>templ) o.write(templ, t-templ);

        if (!*t) return o;

        if (t[1]!=c) break;

        o.put(c);
        templ = t += 2;
    }

    o << std::forward<T>(value);
    return strsub(o, c, t+1, std::forward<Tail>(tail)...);
}

template <typename... Args>
std::ostream& strsub(std::ostream& o, const char* templ, Args&&... args) {
    return strsub(o, '%', templ, std::forward<Args>(args)...);
}

// String-returning strsub(...) wrappers:

template <typename... Args>
std::string strsub(char c, const char* templ, Args&&... args) {
    std::ostringstream o;
    return strsub(o, c, templ, std::forward<Args>(args)...), o.str();
}

template <typename... Args>
std::string strsub(const char* templ, Args&&... args) {
    return strsub('%', templ, std::forward<Args>(args)...);
}

} // namespace sup
