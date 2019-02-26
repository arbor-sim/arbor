#pragma once

#include <string>

#include <arbor/arbexcept.hpp>

namespace pyarb {

using arb::arbor_exception;

// Python wrapper errors

struct python_error: arbor_exception {
    explicit python_error(const std::string& message);
};

} // namespace pyarb
