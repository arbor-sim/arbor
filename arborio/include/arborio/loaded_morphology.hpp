#pragma once

#include <any>

#include <arbor/morph/label_dict.hpp>
#include <arbor/morph/morphology.hpp>
#include <arbor/morph/segment_tree.hpp>

namespace arborio {

struct ARB_SYMBOL_VISIBLE swc_metadata {};

struct ARB_SYMBOL_VISIBLE asc_color {
    uint8_t r = 0;
    uint8_t g = 0;
    uint8_t b = 0;
};

struct ARB_SYMBOL_VISIBLE asc_spine {
    std::string name;
    arb::mpoint location;
};

enum ARB_SYMBOL_VISIBLE asc_marker { dot, circle, cross, none };

struct ARB_SYMBOL_VISIBLE asc_marker_set {
    asc_color color;
    asc_marker marker = asc_marker::none;
    std::string name;
    std::vector<arb::mpoint> locations;
};

struct ARB_SYMBOL_VISIBLE asc_metadata {
    std::vector<asc_marker_set> markers;
    std::vector<asc_spine> spines;
};

// Bundle some detailed metadata for neuroml ingestion.
struct ARB_SYMBOL_VISIBLE nml_metadata {
    // Cell id, or empty if morphology was taken from a top-level <morphology> element.
    std::optional<std::string> cell_id;

    // Morphology id.
    std::string id;

    // One region expression for each segment id.
    arb::label_dict segments;

    // One region expression for each name applied to one or more segments.
    arb::label_dict named_segments;

    // One region expression for each segmentGroup id.
    arb::label_dict groups;

    // Map from segmentGroup ids to their corresponding segment ids.
    std::unordered_map<std::string, std::vector<unsigned long long>> group_segments;
};

// Interface for ingesting morphology data
struct ARB_SYMBOL_VISIBLE loaded_morphology {
    // Raw segment tree, identical to morphology.
    arb::segment_tree segment_tree;

    // Morphology constructed from description.
    arb::morphology morphology;

    // Regions and locsets defined in the description.
    arb::label_dict labels;

    // Loader specific metadata
    std::variant<swc_metadata, asc_metadata, nml_metadata> metadata;
};

}
