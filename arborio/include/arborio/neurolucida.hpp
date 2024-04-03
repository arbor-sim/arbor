#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <filesystem>

#include <arbor/arbexcept.hpp>

#include <arborio/loaded_morphology.hpp>
#include <arborio/export.hpp>

namespace arborio {

// Common base-class for arborio run-time errors.
struct ARB_SYMBOL_VISIBLE asc_exception: public arb::arbor_exception {
    asc_exception(const std::string& what_arg):
        arb::arbor_exception(what_arg)
    {}
};

// Generic error parsing asc data.
struct ARB_SYMBOL_VISIBLE asc_parse_error: asc_exception {
    asc_parse_error(const std::string& error_msg, unsigned line, unsigned column);
    std::string message;
    unsigned line;
    unsigned column;
};

// An unsupported ASC description feature was encountered.
struct ARB_SYMBOL_VISIBLE asc_unsupported: asc_exception {
    asc_unsupported(const std::string& error_msg);
    std::string message;
};

// Perform the parsing of the input as a string.
ARB_ARBORIO_API loaded_morphology parse_asc_string(const char* input);

// Load asc morphology from file with name filename.
ARB_ARBORIO_API loaded_morphology load_asc(const std::filesystem::path& filename);

} // namespace arborio
