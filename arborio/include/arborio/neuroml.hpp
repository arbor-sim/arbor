#pragma once

#include <cstddef>
#include <stdexcept>
#include <optional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <arbor/morph/label_dict.hpp>
#include <arbor/morph/morphology.hpp>
#include <arborio/export.hpp>

namespace arborio {

// `non_negative` represents the corresponding constraint in the schema, which
// can mean any arbitrarily large non-negative integer value.
//
// A faithful representation would use an arbitrary-size 'big' integer or a
// string, but for ease of implementation (and a bit more speed) we restrict it
// to whatever we can fit in an unsigned long long.

using non_negative = unsigned long long;

// Common base-class for neuroml run-time errors.
struct ARB_SYMBOL_VISIBLE neuroml_exception: std::runtime_error {
    neuroml_exception(const std::string& what_arg):
        std::runtime_error(what_arg)
    {}
};

// Can't parse NeuroML if we don't have a document.
struct ARB_SYMBOL_VISIBLE nml_no_document: neuroml_exception {
    nml_no_document();
};

// Generic error parsing NeuroML data.
struct ARB_SYMBOL_VISIBLE nml_parse_error: neuroml_exception {
    nml_parse_error(const std::string& error_msg);
    std::string error_msg;
};

// NeuroML morphology error: improper segment data, e.g. bad id specification,
// segment parent does not exist, fractionAlong is out of bounds, missing
// required <proximal> data.
struct ARB_SYMBOL_VISIBLE nml_bad_segment: neuroml_exception {
    nml_bad_segment(unsigned long long segment_id);
    unsigned long long segment_id;
};

// NeuroML morphology error: improper segmentGroup data, e.g. malformed
// element data, missing referenced segments or groups, etc.
struct ARB_SYMBOL_VISIBLE nml_bad_segment_group: neuroml_exception {
    nml_bad_segment_group(const std::string& group_id);
    std::string group_id;
};

// A segment or segmentGroup ultimately refers to itself via `parent`
// or `include` elements respectively.
struct ARB_SYMBOL_VISIBLE nml_cyclic_dependency: neuroml_exception {
    nml_cyclic_dependency(const std::string& id);
    std::string id;
};

// Collect and parse morphology elements from XML.
// No validation is performed against the NeuroML v2 schema.

// Note: segment id values are interpreted as unsigned long long values;
// parsing larger segment ids will throw an exception.

struct nml_morphology_data {
    // Cell id, or empty if morphology was taken from a top-level <morphology> element.
    std::optional<std::string> cell_id;

    // Morphology id.
    std::string id;

    // Morphology constructed from a single NeuroML <morphology> element.
    arb::morphology morphology;

    // One region expression for each segment id.
    arb::label_dict segments;

    // One region expression for each name applied to one or more segments.
    arb::label_dict named_segments;

    // One region expression for each segmentGroup id.
    arb::label_dict groups;

    // Map from segmentGroup ids to their corresponding segment ids.
    std::unordered_map<std::string, std::vector<unsigned long long>> group_segments;
};

// Represent NeuroML data determined by provided string.

struct ARB_ARBORIO_API neuroml_impl;

// TODO: C++20, replace with enum class and deploy using enum as appropriate.
namespace neuroml_options {
    enum values {
        none = 0,
        allow_spherical_root = 1
    };
}

struct ARB_ARBORIO_API neuroml {
    // Correct interpretation of zero-length segments is currently a bit unclear
    // in NeuroML 2.0, hence these options. For further options, use flags in powers of two
    // so that we can bitwise combine and test them.

    neuroml();
    explicit neuroml(std::string nml_document);

    neuroml(neuroml&&);
    neuroml(const neuroml&) = delete;

    neuroml& operator=(neuroml&&);
    neuroml& operator=(const neuroml&) = delete;

    // Query top-level cells and (standalone) morphologies.

    std::vector<std::string> cell_ids() const;
    std::vector<std::string> morphology_ids() const;

    // Parse and retrieve top-level morphology or morphology associated with a cell.
    // Return nullopt if not found.

    std::optional<nml_morphology_data> morphology(const std::string& morph_id, enum neuroml_options::values = neuroml_options::none) const;
    std::optional<nml_morphology_data> cell_morphology(const std::string& cell_id, enum neuroml_options::values = neuroml_options::none) const;

    ~neuroml();

private:
    std::unique_ptr<neuroml_impl> impl_;
};

} // namespace arborio
