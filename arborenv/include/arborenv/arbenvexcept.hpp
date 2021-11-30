#pragma once

#include <stdexcept>
#include <string>

// Arborenv-specific exception hierarchy.

namespace arbenv {

// Common base-class for arborenv run-time errors.

struct arborenv_exception: std::runtime_error {
    arborenv_exception(const std::string& what_arg):
        std::runtime_error(what_arg)
    {}
};

// Environment variable parsing errors.

struct invalid_env_value: arborenv_exception {
    invalid_env_value(const std::string& variable, const std::string& value);
    std::string env_variable;
    std::string env_value;
};

// GPU enumeration, selection.

struct no_such_gpu: arborenv_exception {
    no_such_gpu(int gpu_id);
    int gpu_id;
};

struct gpu_uuid_error: arborenv_exception {
    gpu_uuid_error(std::string what);
};

} // namespace arbenv
