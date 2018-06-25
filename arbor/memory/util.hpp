#pragma once

#include <iostream>
#include <sstream>
#include <vector>

#define LOG_ERROR(msg) util::log_error(__FILE__, __LINE__, msg)

namespace arb {
namespace memory {
namespace util {

//'\e[1;31m' # Red
//'\e[1;32m' # Green
//'\e[1;33m' # Yellow
//'\e[1;34m' # Blue
//'\e[1;35m' # Purple
//'\e[1;36m' # Cyan
//'\e[1;37m' # White
enum stringColor {kWhite, kRed, kGreen, kBlue, kYellow, kPurple, kCyan};

#define COLOR_PRINTING
#ifdef COLOR_PRINTING
inline std::string colorize(std::string const& s, stringColor c) {
    switch(c) {
        case kWhite :
            return "\033[1;37m"  + s + "\033[0m";
        case kRed   :
            return "\033[1;31m" + s + "\033[0m";
        case kGreen :
            return "\033[1;32m" + s + "\033[0m";
        case kBlue  :
            return "\033[1;34m" + s + "\033[0m";
        case kYellow:
            return "\033[1;33m" + s + "\033[0m";
        case kPurple:
            return "\033[1;35m" + s + "\033[0m";
        case kCyan  :
            return "\033[1;36m" + s + "\033[0m";
    }
    // avoid warnings from GCC
    return s;
}
#else
inline std::string colorize(std::string const& s, stringColor c) {
    return s;
}
#endif

// helpers for inline printing
inline std::string red(std::string const& s) {
    return colorize(s, kRed);
}
inline std::string green(std::string const& s) {
    return colorize(s, kGreen);
}
inline std::string yellow(std::string const& s) {
    return colorize(s, kYellow);
}
inline std::string blue(std::string const& s) {
    return colorize(s, kBlue);
}
inline std::string purple(std::string const& s) {
    return colorize(s, kPurple);
}
inline std::string cyan(std::string const& s) {
    return colorize(s, kCyan);
}
inline std::string white(std::string const& s) {
    return colorize(s, kWhite);
}

template <typename T>
std::string print_pointer(const T* ptr) {
    std::stringstream s;
    s << ptr;
    return yellow(s.str());
}

void log_error(const char* file, int line, const std::string& msg);

} // namespace util
} // namespace memory
} // namespace arb
