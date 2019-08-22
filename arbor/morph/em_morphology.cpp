#include <mutex>
#include <vector>

#include <arbor/morph/morphology.hpp>
#include <arbor/morph/primitives.hpp>

#include "morph/em_morphology.hpp"
#include "util/span.hpp"

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

    // Gargh! What shit.
    // We can't consistently label the root node: for spherical root, it should
    // be {npos, 0}? Gargh!
    // We have to rely on the fact that consumers will use our interface to
    // request ranges etc, to handle the root special case.

    // Cache the sample locations.
    // Iterate backwards over branches distal to root, so that the parent branch at
    // fork points will label its distal sample.
    sample_locs_.resize(ns);
    for (msize_t b=nb-1; b>0; --b) {
        auto idx = util::make_range(morph_.branch_indexes(b));
        double len = branch_lengths_[b];
        // Handle 0 length branch.
        len = len==0.? 1.: len;
        double start = dist2root_[idx.front()];
        for (auto i: idx) {
            sample_locs_[i] = {b, (dist2root_[i]-start)/len};
        }
        // For ensure that all non-spherical branches have their last sample 
        if (idx.size()>1u) {
            sample_locs_[idx.back()] = mlocation{b, 1.};
        }
    }
    sample_locs_[0] = mlocation{0, 0.};

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

mlocation em_morphology::root() const {
    return sample_locs_[0];
}

mlocation em_morphology::sample2loc(msize_t sid) const {
    return sample_locs_[sid];
}

mlocation_list em_morphology::terminals() const {
    return terminals_;
}

} // namespace arb
