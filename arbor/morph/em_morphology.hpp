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

    // Convenience methods for morphology access
    // that are forwarded directly to the morphology object:

    auto empty() const { return morph_.empty(); }
    auto spherical_root() const { return morph_.spherical_root(); }
    auto num_branches() const { return morph_.num_branches(); }
    auto num_samples() const { return morph_.num_samples(); }
    auto branch_parent(msize_t b) const { return morph_.branch_parent(b); }
    auto branch_children(msize_t b) const { return morph_.branch_children(b); }

    // Access to computed and cached data:

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
