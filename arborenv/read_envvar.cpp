#include <cerrno>
#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <string>

#include <arborenv/arbenvexcept.hpp>
#include "read_envvar.hpp"

using namespace std::literals;

namespace arbenv {

static std::optional<long long> read_env_integer_(const char* env_var, bool throw_on_err) {
    const char* str = std::getenv(env_var);
    if (!str || !*str) return std::nullopt;

    char* end = 0;
    errno = 0;
    long long v = std::strtoll(str, &end, 10);
    bool out_of_range = errno==ERANGE;
    errno = 0;

    if (out_of_range && throw_on_err) {
        throw invalid_env_value(env_var, str);
    }

    while (*end && std::isspace(*end)) ++end;
    if (*end) {
        if (throw_on_err) {
            throw invalid_env_value(env_var, str);
        }
        else {
            return std::nullopt;
        }
    }

    return v;
}

std::optional<long long> read_env_integer(const char* env_var) {
    return read_env_integer_(env_var, false);
}

std::optional<long long> read_env_integer(const char* env_var, throw_on_invalid_t) {
    return read_env_integer_(env_var, true);
}

} // namespace arbenv
