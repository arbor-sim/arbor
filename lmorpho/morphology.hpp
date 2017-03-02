#pragma once

// Representation of 3-d embedded cable morphology.

#include <vector>

struct section_point {
    double x, y, z, r;  // [µm], r is radius.
};

struct section_geometry {
    unsigned id;
    unsigned parent_id;
    bool terminal;
    std::vector<section_point> points;
    double length; // µm

    void segment(double dx);
};

struct morphology {
    section_point soma; // origin + spherical radius
    std::vector<section_geometry> sections;

    void segment(double dx) {
        for (auto& s: sections) s.segment(dx);
    }
};

