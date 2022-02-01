#pragma once

#include <stdexcept>
#include <string>

#include <arborenv/export.hpp>

// Arborenv-specific exception hierarchy.

namespace arbenv {

// Common base-class for arborenv run-time errors.

struct ARB_ARBORENV_API arborenv_exception: std::runtime_error {
    arborenv_exception(const std::string& what_arg):
        std::runtime_error(what_arg)
    {}
};

// Environment variable parsing errors.

struct ARB_ARBORENV_API invalid_env_value: arborenv_exception {
    invalid_env_value(const std::string& variable, const std::string& value);
    std::string env_variable;
    std::string env_value;
};

// GPU enumeration, selection.

struct ARB_ARBORENV_API no_such_gpu: arborenv_exception {
    no_such_gpu(int gpu_id);
    int gpu_id;
};

struct ARB_ARBORENV_API gpu_uuid_error: arborenv_exception {
    gpu_uuid_error(std::string what);
};

} // namespace arbenv
