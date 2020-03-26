#pragma once

#include <cstring>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <type_traits>
#include <vector>

#include <tinyopt/common.h>
#include <tinyopt/maybe.h>

namespace to {

template <typename V = std::string, typename P = default_parser<V>, typename = std::enable_if_t<!std::is_same<V, void>::value>>
maybe<V> parse(char**& argp, char shortopt, const char* longopt = nullptr, const P& parser = P{}) {
    const char* arg = argp[0];

    if (!arg || arg[0]!='-') {
        return nothing;
    }

    const char* text;

    if (arg[1]=='-' && longopt) {
        const char* rest = arg+2;
        const char* eq = std::strchr(rest, '=');

        if (!std::strcmp(rest, longopt)) {
            if (!argp[1]) throw missing_argument(arg);
            text = argp[1];
            argp += 2;
        }
        else if (eq && !std::strncmp(rest, longopt,  eq-rest)) {
            text = eq+1;
            argp += 1;
        }
        else {
            return nothing;
        }
    }
    else if (shortopt && arg[1]==shortopt && arg[2]==0) {
        if (!argp[1]) throw missing_argument(arg);
        text = argp[1];
        argp += 2;
    }
    else {
        return nothing;
    }

    auto v = parser(text);

    if (!v) throw option_parse_error(arg);
    return v;
}

inline maybe<void> parse(char**& argp, char shortopt, const char* longopt = nullptr) {
    if (!*argp || *argp[0]!='-') {
        return nothing;
    }
    else if (argp[0][1]=='-' && longopt && !std::strcmp(argp[0]+2, longopt)) {
        ++argp;
        return true;
    }
    else if (shortopt && argp[0][1]==shortopt && argp[0][2]==0) {
        ++argp;
        return true;
    }
    else {
        return nothing;
    }
}

} // namespace to
