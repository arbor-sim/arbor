#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>

#include <arbor/arbexcept.hpp>
#include <arbor/morph/label_dict.hpp>
#include <arbor/morph/morphology.hpp>
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

struct asc_morphology {
    // Raw segment tree from ASC, identical to morphology.
    arb::segment_tree segment_tree;

    // Morphology constructed from asc description.
    arb::morphology morphology;

    // Regions and locsets defined in the asc description.
    arb::label_dict labels;
};

// Perform the parsing of the input as a string.
ARB_ARBORIO_API asc_morphology parse_asc_string(const char* input);
ARB_ARBORIO_API arb::segment_tree parse_asc_string_raw(const char* input);

// Load asc morphology from file with name filename.
ARB_ARBORIO_API asc_morphology load_asc(std::string filename);
ARB_ARBORIO_API arb::segment_tree load_asc_raw(std::string filename);

} // namespace arborio
