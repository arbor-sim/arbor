#include <iostream>

#include "morphology.h"
#include "lsystem.h"

struct xyz {
    double x, y, z;
    xyz() {}
    template <typename S>
    explicit xyz(const S& s): x(s.x), y(s.y), z(s.z) {}

    friend std::ostream& operator<<(std::ostream& o, const xyz& p) {
        return o << p.x << ',' << p.y << ',' << p.z;
    }
};


// serialize morphology as SWC (soon)
std::ostream& operator<<(std::ostream& O, const morphology& m) {
    O << "soma radius: " << m.soma.r << "\n";
    O << "#segments: " << m.segments.size() << "\n";
    segment_point p = m.segments.back().points.back();
    O << "last segment tip: " << xyz(p) << "; radius: " << p.r << "\n";
    return O;
}

int main() {
    lsys_param P; // default
    std::minstd_rand g;

    for (int i=0; i<5; ++i) {
        auto morph = generate_morphology(P, g);
        std::cout << "#" << i << ":\n" << morph << "\n";
    }
}

