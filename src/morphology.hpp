#pragma once

// Representation of 3-d embedded cable morphology, independent of other
// cell information.

#include <vector>

namespace nest {
namespace mc {

struct section_point {
    double x, y, z, r;  // [µm], r is radius.
};

enum class section_kind {
    soma,
    dendrite,
    axon,
    none
};

struct section_geometry {
    unsigned id = 0; // ids should be contigously numbered from 1 in the morphology.
    unsigned parent_id = 0;
    bool terminal = false;
    std::vector<section_point> points;
    double length = 0; // µm
    section_kind kind = section_kind::none;

    // Re-discretize the section into ceil(length/dx) segments.
    void segment(double dx);

    // (Re-)compute length.
    void compute_length();
};

struct morphology {
    // origin + spherical radius; convention: r==0 => no soma
    section_point soma = { 0, 0, 0, 0};

    std::vector<section_geometry> sections;

    bool has_soma() const {
        return soma.r!=0;
    }

    bool empty() const {
        return sections.empty() && !has_soma();
    }

    operator bool() const { return !empty(); }

    // Check invariants:
    // 1. sections[i].id = i+1  (id 0 corresponds to soma)
    // 2. sections[i].parent_id < sections[i].id
    // 3. sections[i].terminal iff !exists j s.t. sections[j].parent_id = sections[i].id

    bool check_invariants() const;

    // Re-discretize all sections.
    void segment(double dx) {
        for (auto& s: sections) s.segment(dx);
    }

};

} // namespace mc
} // namespace nest
