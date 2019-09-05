#pragma once

#include <vector>

#include <arbor/morph/morphology.hpp>
#include <arbor/morph/primitives.hpp>

namespace arb {

class em_morphology {
    morphology morph_;

    mlocation_list terminals_;
    mlocation_list forks_;
    mlocation_list sample_locs_;

    // distance from sample to root
    std::vector<double> dist2root_;
    std::vector<double> branch_lengths_;

public:
    em_morphology(const morphology& m);

    const morphology& morph() const;

    mlocation_list terminals() const;
    mlocation root() const;

    mlocation sample2loc(msize_t sid) const;

    mlocation canonicalize(mlocation) const;

    mlocation_list cover(mlocation) const;
};

} // namespace arb
