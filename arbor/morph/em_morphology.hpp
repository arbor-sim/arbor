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
    em_morphology();
    em_morphology(const morphology& m);

    const morphology& morph() const;

    mlocation_list terminals() const;
    mlocation root() const;

    mlocation sample2loc(msize_t sid) const;

    void assert_valid_location(mlocation) const;
    mlocation canonicalize(mlocation) const;

    // Find all locations on the morphology that share the same canonoical
    // representation of loc.
    // If include_loc is false, the input location is excluded from the result.
    mlocation_list cover(mlocation, bool include_loc=true) const;

    mlocation_list minset(const mlocation_list&) const;

    double branch_length(msize_t bid) const { return branch_lengths_.at(bid); }
};

} // namespace arb
