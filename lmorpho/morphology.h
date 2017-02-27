#pragma once

// Representation of 3-d embedded cable morphology.

#include <vector>

struct segment_point {
    double x, y, z, r;  // r is radius.
};


struct segment_geometry {
    unsigned id;
    unsigned parent_id;
    std::vector<segment_point> points;
};

struct morphology {
    segment_point soma; // origin + spherical radius
    std::vector<segment_geometry> segments;
};

