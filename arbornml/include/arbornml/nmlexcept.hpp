#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>

namespace arbnml {

// Common base-class for arbnml run-time errors.

struct neuroml_exception: std::runtime_error {
    neuroml_exception(const std::string& what_arg):
        std::runtime_error(what_arg)
    {}
};

// Generic XML error (as reported by libxml2).

struct xml_error: neuroml_exception {
    xml_error(const std::string& xml_error_msg, unsigned line = 0);
    std::string xml_error_msg;
    unsigned line;
};

// Can't parse NeuroML if we don't have a document.

struct no_document: neuroml_exception {
    no_document();
};

// Generic error parsing NeuroML data.

struct parse_error: neuroml_exception {
    parse_error(const std::string& error_msg, unsigned line = 0);
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

} // namespace arbnml
