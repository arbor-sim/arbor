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

/*
// Generic error parsing asc data.
struct asc_parse_error: asc_exception {
    asc_parse_error(const std::string& error_msg, unsigned line = 0);
    std::string error_msg;
    unsigned line;
};

// NeuroML morphology error: improper segment data, e.g. bad id specification,
// segment parent does not exist, fractionAlong is out of bounds, missing
// required <proximal> data.
struct bad_segment: neuroml_exception {
    bad_segment(unsigned long long segment_id, unsigned line = 0);
    unsigned long long segment_id;
    unsigned line;
};

// NeuroML morphology error: improper segmentGroup data, e.g. malformed
// element data, missing referenced segments or groups, etc.
struct bad_segment_group: neuroml_exception {
    bad_segment_group(const std::string& group_id, unsigned line = 0);
    std::string group_id;
    unsigned line;
};

// A segment or segmentGroup ultimately refers to itself via `parent`
// or `include` elements respectively.
struct cyclic_dependency: neuroml_exception {
    cyclic_dependency(const std::string& id, unsigned line = 0);
    std::string id;
    unsigned line;
};
*/

struct asc_morphology {
    // Morphology constructed from asc description.
    arb::morphology morphology;

    // Regions and locsets defined in the asc description.
    arb::label_dict labels;
};

asc_morphology load_asc(std::string filename);

} // namespace arborio
