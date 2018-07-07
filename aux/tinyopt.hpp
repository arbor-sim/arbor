#pragma once

#include <cstring>
#include <iostream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <arbor/util/optional.hpp>

namespace to {

using arb::util::optional;
using arb::util::nullopt;
using arb::util::just;

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

template <typename V>
struct default_parser {
    optional<V> operator()(const std::string& text) const {
        V v;
        std::istringstream stream(text);
        stream >> v;
        return stream? just(v): nullopt;
    }
};

template <typename V>
class keyword_parser {
    std::vector<std::pair<std::string, V>> map_;

public:
    template <typename KeywordPairs>
    keyword_parser(const KeywordPairs& pairs): map_(std::begin(pairs), std::end(pairs)) {}

    optional<V> operator()(const std::string& text) const {
        for (const auto& p: map_) {
            if (text==p.first) return p.second;
        }
        return nullopt;
    }
};

template <typename KeywordPairs>
auto keywords(const KeywordPairs& pairs) -> keyword_parser<decltype(std::begin(pairs)->second)> {
    return keyword_parser<decltype(std::begin(pairs)->second)>(pairs);
}

template <typename V = std::string, typename P = default_parser<V>, typename = std::enable_if_t<!std::is_same<V, void>::value>>
optional<V> parse_opt(char **& argp, char shortopt, const char* longopt=nullptr, const P& parse = P{}) {
    const char* arg = argp[0];

    if (!arg || arg[0]!='-') {
        return nullopt;
    }

    std::string text;

    if (arg[1]=='-' && longopt) {
        const char* rest = arg+2;
        const char* eq = std::strchr(rest, '=');

        if (!std::strcmp(rest, longopt)) {
            if (!argp[1]) throw parse_opt_error(arg, "missing argument");
            text = argp[1];
            argp += 2;
        }
        else if (eq && !std::strncmp(rest, longopt,  eq-rest)) {
            text = eq+1;
            argp += 1;
        }
        else {
            return nullopt;
        }
    }
    else if (shortopt && arg[1]==shortopt && arg[2]==0) {
        if (!argp[1]) throw parse_opt_error(arg, "missing argument");
        text = argp[1];
        argp += 2;
    }
    else {
        return nullopt;
    }

    auto v = parse(text);

    if (!v) throw parse_opt_error(arg, "failed to parse option argument");
    return v;
}

optional<void> parse_opt(char **& argp, char shortopt, const char* longopt) {
    if (!*argp || *argp[0]!='-') {
        return nullopt;
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
        return nullopt;
    }
}

} // namespace to
