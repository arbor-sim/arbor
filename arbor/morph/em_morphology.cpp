#include <mutex>
#include <vector>

#include <arbor/morph/error.hpp>
#include <arbor/morph/morphology.hpp>
#include <arbor/morph/primitives.hpp>

#include "morph/em_morphology.hpp"
#include "util/span.hpp"
#include "util/strprintf.hpp"

namespace arb {

em_morphology::em_morphology(const morphology& m):
    morph_(m)
{
    using util::count_along;
    using util::make_span;

    auto& parents = morph_.sample_parents();
    auto& samples = morph_.samples();

    const auto ns = morph_.num_samples();
    const auto nb = morph_.num_branches();

    // Cache distance of each sample from the root.
    dist2root_.resize(ns);
    dist2root_[0] = 0.;
    for (auto i: make_span(1, ns)) {
        const auto p = parents[i];
        dist2root_[i] = dist2root_[p] + distance(samples[p], samples[i]);
    }
    // Cache the legth of each branch.
    branch_lengths_.reserve(nb);
    for (auto i: make_span(nb)) {
        auto idx = util::make_range(morph_.branch_indexes(i));
        branch_lengths_.push_back(dist2root_[idx.back()]- dist2root_[idx.front()]);
    }

    // Cache the sample locations.
    // Iterate backwards over branches distal to root, so that the parent branch at
    // fork points will label its distal sample.
    sample_locs_.resize(ns);
    for (int b=nb-1; b>=0; --b) {
        auto idx = util::make_range(morph_.branch_indexes(b));
        double len = branch_lengths_[b];
        // Handle 0 length branch.
        len = len==0.? 1.: len;
        double start = dist2root_[idx.front()];
        for (auto i: idx) {
            sample_locs_[i] = {msize_t(b), (dist2root_[i]-start)/len};
        }
        // For ensure that all non-spherical branches have their last sample 
        if (idx.size()>1u) {
            sample_locs_[idx.back()] = mlocation{msize_t(b), 1};
        }
    }
    sample_locs_[0] = mlocation{0, 0.};

    // Cache the location of terminal and fork points.
    auto& props = morph_.sample_props();
    for (auto i: count_along(props)) {
        auto p = props[i];
        if (is_terminal(p)) {
            terminals_.push_back(sample2loc(i));
        }
        if (is_fork(p)) {
            forks_.push_back(sample2loc(i));
        }
    }
}

const morphology& em_morphology::morph() const {
    return morph_;
}

mlocation em_morphology::root() const {
    return {0,0};
}

mlocation em_morphology::sample2loc(msize_t sid) const {
    if (sid>=morph_.num_samples()) {
        throw morphology_error(util::pprintf("Sample {} does not exist in morpology", sid));
    }
    return sample_locs_[sid];
}

mlocation_list em_morphology::terminals() const {
    return terminals_;
}

mlocation em_morphology::canonicalize(mlocation loc) const {
    if (!test_invariants(loc)) {
        throw morphology_error(util::pprintf("Invalid location {}", loc));
    }
    if (loc.branch>=morph_.num_branches()) {
        throw morphology_error(util::pprintf("Location {} does not exist in morpology", loc));
    }

    // Test if location is at the start of a branch.
    if (loc.pos==0.) {
        auto p = morph_.branch_parent(loc.branch);
        return p==mnpos? root(): mlocation{p, 1};
    }
    return loc;
}

} // namespace arb
