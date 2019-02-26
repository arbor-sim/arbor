#pragma once

// Flags controlling printer behaviour and code generation.
// (Not all flags need be recognized by all printers.)

#include <string>

#include "simd.hpp"
struct printer_options {
    // C++ namespace for generated code.
    std::string cpp_namespace;

    // Explicit vectorization (C printer only)? Default is none.
    simd_spec simd;

    // Instrument kernels? True => use ::arb::profile regions.
    // Currently only supported for C printer.

    bool profile = false;
};
