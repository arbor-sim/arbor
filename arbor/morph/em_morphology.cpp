#include <mutex>
#include <stack>
#include <vector>

#include <arbor/morph/error.hpp>
#include <arbor/morph/morphology.hpp>
#include <arbor/morph/primitives.hpp>

#include "morph/em_morphology.hpp"
#include "util/rangeutil.hpp"
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

    if (!ns) return;

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
    if (morph_.spherical_root()) {
        branch_lengths_[0] = samples[0].loc.radius*2;
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
        // Ensure that all non-spherical branches have their last sample 
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

em_morphology::em_morphology():
    em_morphology(morphology())
{}

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

mlocation_list em_morphology::cover(mlocation loc, bool include_loc) const {
    mlocation_list L{};
    if (include_loc) L.push_back(loc);

    // If the location is not at the end of a branch, it is its own cover.
    if (loc.pos>0. && loc.pos<1.) return L;

    // First location is {0,0} on a spherical root: nothing more to do.
    if (loc==mlocation{0,0} && morph_.spherical_root()) {
        return L;
    }

    if (loc.pos==1) {
        // The location is at the end of a branch:
        //      add the location at the start of each child branch.
        for (auto b: morph_.branch_children(loc.branch)) {
            L.push_back({b, 0});
        }
    }
    else if (loc.pos==0) {
        // The location is at the start of a branch:
        //      add the location at the end of the parent branch, and locations
        //      at the start of each child of the parent branch.
        auto p = morph_.branch_parent(loc.branch);
        if (p!=mnpos) L.push_back({p, 1});
        for (auto b: morph_.branch_children(p)) {
            if (b!=loc.branch) L.push_back({b, 0});
        }
    }

    util::sort(L);

    return L;
}

void em_morphology::assert_valid_location(mlocation loc) const {
    if (!test_invariants(loc)) {
        throw morphology_error(util::pprintf("Invalid location {}", loc));
    }
    if (loc.branch>=morph_.num_branches()) {
        throw morphology_error(util::pprintf("Location {} does not exist in morpology", loc));
    }
}

mlocation em_morphology::canonicalize(mlocation loc) const {
    assert_valid_location(loc);

    // Test if location is at the start of a branch.
    if (loc.pos==0.) {
        auto p = morph_.branch_parent(loc.branch);
        return p==mnpos? root(): mlocation{p, 1};
    }
    return loc;
}

mlocation_list em_morphology::minset(const mlocation_list& in) const {
    mlocation_list L;

    std::stack<msize_t> stack;

    // All root branches must be searched.
    for (auto c: morph_.branch_children(mnpos)) {
        stack.push(c);
    }

    // Depth-first traversal of the branch tree.
    while (!stack.empty()) {
        auto branch = stack.top();
        stack.pop();

        // Search for a location on branch.
        auto it = std::lower_bound(in.begin(), in.end(), mlocation{branch, 0});

        // If found, insert to the minset and skip the rest of this sub-tree.
        if (it!=in.end() && it->branch==branch) {
            L.push_back(*it);
            continue;
        }

        // No location on this branch, so continue searching in this sub-tree.
        for (auto c: morph_.branch_children(branch)) {
            stack.push(c);
        }
    }

    util::sort(L);

    return L;
}

} // namespace arb

