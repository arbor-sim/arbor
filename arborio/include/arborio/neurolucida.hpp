#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>

#include <arbor/morph/label_dict.hpp>
#include <arbor/morph/morphology.hpp>

namespace arborio {

// Common base-class for neuroml run-time errors.
struct asc_exception: std::runtime_error {
    asc_exception(const std::string& what_arg):
        std::runtime_error(what_arg)
    {}
};

// Can't parse asc if we don't have a document.
struct asc_no_document: asc_exception {
    asc_no_document();
};

// Generic error parsing asc data.
struct asc_parse_error: asc_exception {
    asc_parse_error(const std::string& error_msg, unsigned line, unsigned column);
    std::string message;
    unsigned line;
    unsigned column;
};

// An unsupported ASC description feature was encountered.
struct asc_unsupported: asc_exception {
    asc_unsupported(const std::string& error_msg);
    std::string message;
};

struct asc_morphology {
    // Morphology constructed from asc description.
    arb::morphology morphology;

    // Regions and locsets defined in the asc description.
    arb::label_dict labels;
};

// Load asc morphology from file with name filename.
asc_morphology load_asc(std::string filename);

} // namespace arborio
