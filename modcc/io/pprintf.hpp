#pragma once

#include <sstream>
#include <string>

//'\e[1;31m' # Red
//'\e[1;32m' # Green
//'\e[1;33m' # Yellow
//'\e[1;34m' # Blue
//'\e[1;35m' # Purple
//'\e[1;36m' # Cyan
//'\e[1;37m' # White
enum class stringColor {white, red, green, blue, yellow, purple, cyan};

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

// variadic printf for easy error messages

inline std::string pprintf(const char* s) {
    return s;
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

