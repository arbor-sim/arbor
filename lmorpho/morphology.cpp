#include <cmath>
#include <vector>

#include <math.hpp>

#include "morphology.hpp"

using nest::mc::math::lerp;

static section_point lerp(const section_point& a, const section_point& b, double u) {
    return { lerp(a.x, b.x, u), lerp(a.y, b.y, u), lerp(a.z, b.z, u), lerp(a.r, b.r, u) };
}

static double distance(const section_point& a, const section_point& b) {
    double dx = b.x-a.x;
    double dy = b.y-a.y;
    double dz = b.z-a.z;

    return std::sqrt(dx*dx+dy*dy+dz*dz);
}

void section_geometry::segment(double dx) {
    unsigned npoint = points.size();
    if (dx<=0 || npoint<2) return;

    unsigned nseg = static_cast<unsigned>(std::ceil(length/dx));

    std::vector<section_point> sampled;
    sampled.push_back(points.front());
    double sampled_length = 0;

    double left = 0;
    double right = left+distance(points[1], points[0]);
    double x = length/nseg;

    for (unsigned i = 1; i<npoint;) {
        if (right>x) {
            double u = (x-left)/(right-left);
            sampled.push_back(lerp(points[i-1], points[i], u));
            unsigned k = sampled.size();
            sampled_length += distance(sampled[k-2], sampled[k]);
            x = k*length/nseg;
        }
        else {
            ++i;
            left = right;
            right = left+distance(points[i-1], points[i]);
        }
    }
    if (sampled.size()<=nseg) {
        sampled.push_back(points.back());
    }

    points = std::move(sampled);
    length = sampled_length;
}
