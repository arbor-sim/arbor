#pragma once

#include <cstring>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "../gtest.h"

#include <communication/global_policy.hpp>

#include "util/optional.hpp"

namespace nest {
namespace mc {
namespace to {

struct parse_opt_error: public std::runtime_error {
    parse_opt_error(const std::string& s): std::runtime_error(s) {}
    parse_opt_error(const char *arg, const std::string& s):
        std::runtime_error(s+": "+arg) {}
};

void usage(const char* argv0, const std::string& usage_str) {
    const char* basename = std::strrchr(argv0, '/');
    basename = basename? basename+1: argv0;

    std::cout << "Usage: " << basename << " " << usage_str << "\n";
}

void usage(const char* argv0, const std::string& usage_str, const std::string& parse_err) {
    const char* basename = std::strrchr(argv0, '/');
    basename = basename? basename+1: argv0;

    std::cerr << basename << ": " << parse_err << "\n";
    std::cerr << "Usage: " << basename << " " << usage_str << "\n";
}

template <typename V = std::string>
util::optional<V> parse_opt(char **& argp, char shortopt, const char* longopt=nullptr) {
    const char* arg = argp[0];

    if (!arg || arg[0]!='-') {
        return util::nothing;
    }

    std::stringstream buf;

    if (arg[1]=='-' && longopt) {
        const char* rest = arg+2;
        const char* eq = std::strchr(rest, '=');

        if (!std::strcmp(rest, longopt)) {
            buf.str(argp[1]? argp[1]: throw parse_opt_error(arg, "missing argument"));
            argp += 2;
        }
        else if (eq && !std::strncmp(rest, longopt,  eq-rest)) {
            buf.str(eq+1);
            argp += 1;
        }
        else {
            return util::nothing;
        }
    }
    else if (shortopt && arg[1]==shortopt && arg[2]==0) {
        buf.str(argp[1]? argp[1]: throw parse_opt_error(arg, "missing argument"));
        argp += 2;
    }
    else {
        return util::nothing;
    }

    V v;
    if (!(buf >> v)) {
        throw parse_opt_error(arg, "failed to parse option argument");
    }
    return v;
}

template <>
util::optional<void> parse_opt(char **& argp, char shortopt, const char* longopt) {
    if (!*argp || *argp[0]!='-') {
        return util::nothing;
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
        return util::nothing;
    }
}


} // namespace to;
} // namespace mc
} // namespace nest
