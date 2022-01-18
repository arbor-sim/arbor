#include <optional>

namespace arbenv {

constexpr struct throw_on_invalid_t {} throw_on_invalid;

// Return signed integer value represented by supplied environment variable.
// If the environment variable is unset or empty or does not represent an integer, return std::nullopt.
// If the value does not fit within the range of long long, return LLONG_MIN or LLONG_MAX based on its sign.

std::optional<long long> read_env_integer(const char* env_var);

// Return signed integer value represented by supplied environment variable.
// If the environment variable is unset or empty, return std::nullopt.
// If the environment variable does not represent an integer, throw arbenv::invalid_env_value.
// If the value does not fit within the range of long long, throw arbenv::invalid_env_value.

std::optional<long long> read_env_integer(const char* env_var, throw_on_invalid_t);

} // namespace arbenv
