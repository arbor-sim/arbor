#pragma once

#include <sstream>
#include <string>

#include <fmt/format.h>
#include <fmt/color.h>
#include <fmt/compile.h>

enum class stringColor {white, red, green, blue, yellow, purple, cyan};

// helpers for inline printing
inline std::string cyan(std::string const& s) { return fmt::format("{}",  fmt::styled(s, fmt::fg(fmt::color::cyan))); }
inline std::string red(std::string const& s) { return fmt::format("{}",  fmt::styled(s, fmt::fg(fmt::color::red))); }
inline std::string green(std::string const& s) { return fmt::format("{}",  fmt::styled(s, fmt::fg(fmt::color::green))); }
inline std::string blue(std::string const& s) { return fmt::format("{}",  fmt::styled(s, fmt::fg(fmt::color::blue))); }
inline std::string yellow(std::string const& s) { return fmt::format("{}",  fmt::styled(s, fmt::fg(fmt::color::yellow))); }
inline std::string purple(std::string const& s) { return fmt::format("{}",  fmt::styled(s, fmt::fg(fmt::color::purple))); }
inline std::string white(std::string const& s) { return fmt::format("{}",  fmt::styled(s, fmt::fg(fmt::color::white))); }

inline auto colorize(std::string const& s, stringColor c) {
    switch(c) {
        case stringColor::white : return white(s);
        case stringColor::red   : return red(s);
        case stringColor::green : return green(s);
        case stringColor::blue  : return blue(s);
        case stringColor::yellow: return yellow(s);
        case stringColor::purple: return purple(s);
        case stringColor::cyan  : return cyan(s);
    }
    return s;
}
