#pragma once

// Per-branch piecewise rational polynomial definitions and interpolation.

#include <utility>
#include <vector>

#include "util/piecewise.hpp"
#include "util/ratelem.hpp"

namespace arb {

template <unsigned p, unsigned q>
using pw_ratpoly = util::pw_elements<util::rat_element<p, q>>;

template <unsigned p, unsigned q>
using branch_pw_ratpoly = std::vector<pw_ratpoly<p, q>>;

template <unsigned p, unsigned q>
double interpolate(const pw_ratpoly<p, q>& f, double pos) {
    unsigned index = f.index_of(pos);

    const auto& element = f.element(index);
    std::pair<double, double> bounds = f.interval(index);

    if (bounds.first==bounds.second) return element[0];
    else {
        double x = (pos-bounds.first)/(bounds.second-bounds.first);
        return element(x);
    }
}

template <unsigned p, unsigned q>
double interpolate(const branch_pw_ratpoly<p, q>& f, unsigned bid, double pos) {
    return interpolate(f.at(bid), pos);
}

} // namespace arb
