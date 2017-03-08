#include <cmath>
#include <vector>

#include <math.hpp>

#include <morphology.hpp>

namespace nest {
namespace mc {

using ::nest::mc::math::lerp;

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

    // Re-discretize into nseg segments (nseg+1 points).
    unsigned nseg = static_cast<unsigned>(std::ceil(length/dx));

    std::vector<section_point> sampled;
    sampled.push_back(points.front());
    double sampled_length = 0;

    // [left, right) is the path-length interval for successive
    // linear segments in the section.
    double left = 0;
    double right = left+distance(points[1], points[0]);

    // x is the next sample point (in path-length).
    double x = length/nseg;

    // Scan segments for sample points.
    unsigned i = 1;
    for (;;) {
        if (right>x) {
            double u = (x-left)/(right-left);
            sampled.push_back(lerp(points[i-1], points[i], u));
            unsigned k = sampled.size();
            sampled_length += distance(sampled[k-2], sampled[k-1]);
            x = k*length/nseg;
        }
        else {
            ++i;
            if (i>=npoint) break;

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

void section_geometry::compute_length() {
    length = 0;
    std::size_t npoint = points.size();

    for (std::size_t i =1; i<npoint; ++i) {
        length += distance(points[i], points[i-1]);
    }
}

bool morphology::check_invariants() const {
    std::size_t nsection = sections.size();
    std::vector<int> terminal(true, nsection);

    for (std::size_t i=0; i<nsection; ++i) {
        auto id = sections[i].id;
        auto parent_id = sections[i].parent_id;

        if (id!=i+1) return false;
        if (parent_id>=id) return false;
        if (parent_id>0) {
            auto parent_index = parent_id-1;
            terminal[parent_index] = false;
        }
    }

    for (std::size_t i=0; i<nsection; ++i) {
        if (terminal[i] != sections[i].terminal) return false;
    }

    return true;
}


} // namespace mc
} // namespace nest
