#pragma once

#include <cstdlib>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <arbor/export.hpp>
#include <arbor/morph/primitives.hpp>
#include <arbor/morph/morphology.hpp>

namespace arb {

struct mprovider;

class locset;
class locset_tag {};

class ARB_SYMBOL_VISIBLE locset {
public:
    template <typename Impl,
              typename = std::enable_if_t<std::is_base_of<locset_tag, std::decay_t<Impl>>::value>>
    explicit locset(Impl&& impl):
        impl_(new wrap<Impl>(std::forward<Impl>(impl))) {}

    template <typename Impl,
              typename = std::enable_if_t<std::is_base_of<locset_tag, std::decay_t<Impl>>::value>>
    explicit locset(const Impl& impl):
        impl_(new wrap<Impl>(impl)) {}

    locset(locset&& other) = default;

    locset(const locset& other):
        impl_(other.impl_->clone()) {}

    locset& operator=(locset&& other) = default;

    locset& operator=(const locset& other) {
        impl_ = other.impl_->clone();
        return *this;
    }

    // The default constructor creates an empty "nil" set.
    locset();

    // Implicity convert mlocation and mlocation_lists to locsets.
    locset(mlocation other);
    locset(mlocation_list other);

    template <typename Impl,
              typename = std::enable_if_t<std::is_base_of<locset_tag, std::decay_t<Impl>>::value>>
    locset& operator=(Impl&& other) {
        impl_ = new wrap<Impl>(std::forward<Impl>(other));
        return *this;
    }

    friend mlocation_list thingify(const locset& p, const mprovider& m) {
        return p.impl_->thingify(m);
    }

    friend std::ostream& operator<<(std::ostream& o, const locset& p) {
        return p.impl_->print(o);
    }

    // The sum of two location sets.
    friend locset sum(locset, locset);

    template <typename ...Args>
    friend locset sum(locset l, locset r, Args... args) {
        return sum(sum(std::move(l), std::move(r)), std::move(args)...);
    }

    // The union of two location sets.
    friend locset join(locset, locset);

    template <typename ...Args>
    friend locset join(locset l, locset r, Args... args) {
        return join(join(std::move(l), std::move(r)), std::move(args)...);
    }

private:
    struct interface {
        virtual ~interface() {}
        virtual std::unique_ptr<interface> clone() = 0;
        virtual std::ostream& print(std::ostream&) = 0;
        virtual mlocation_list thingify(const mprovider&) = 0;
    };

    std::unique_ptr<interface> impl_;

    template <typename Impl>
    struct wrap: interface {
        explicit wrap(const Impl& impl): wrapped(impl) {}
        explicit wrap(Impl&& impl): wrapped(std::move(impl)) {}

        virtual std::unique_ptr<interface> clone() override {
            return std::unique_ptr<interface>(new wrap<Impl>(wrapped));
        }

        virtual mlocation_list thingify(const mprovider& m) override {
            return thingify_(wrapped, m);
        }

        virtual std::ostream& print(std::ostream& o) override {
            return o << wrapped;
        }

        Impl wrapped;
    };
};

class region;

namespace ls {

// Explicit location on morphology.
ARB_ARBOR_API locset location(msize_t branch, double pos);

// Set of terminal nodes on a morphology.
ARB_ARBOR_API locset terminal();

// The root node of a morphology.
ARB_ARBOR_API locset root();

// Named locset.
ARB_ARBOR_API locset named(std::string);

// The null (empty) set.
ARB_ARBOR_API locset nil();

// Most distal points of a region.
ARB_ARBOR_API locset most_distal(region reg);

// Most proximal points of a region.
ARB_ARBOR_API locset most_proximal(region reg);

// Translate locations in locset distance μm in the distal direction
ARB_ARBOR_API locset distal_translate(locset ls, double distance);

// Translate locations in locset distance μm in the proximal direction
ARB_ARBOR_API locset proximal_translate(locset ls, double distance);

// Boundary points of a region.
ARB_ARBOR_API locset boundary(region reg);

// Completed boundary points of a region.
// (Boundary of completed components.)
ARB_ARBOR_API locset cboundary(region reg);

// Returns all locations in a locset that are also in the region.
ARB_ARBOR_API locset restrict(locset ls, region reg);

// Returns locations that mark the segments.
ARB_ARBOR_API locset segment_boundaries();

// A range `left` to `right` of randomly selected locations with a
// uniform distribution from region `reg` generated using `seed`
ARB_ARBOR_API locset uniform(region reg, unsigned left, unsigned right, uint64_t seed);

// Proportional location on every branch.
ARB_ARBOR_API locset on_branches(double pos);

// Proportional locations on each component:
// For each component C of the region, find locations L
// s.t. dist(h, L) = r * max {dist(h, t) | t is a distal point in C}.
ARB_ARBOR_API locset on_components(double relpos, region reg);

// Set of locations in the locset with duplicates removed, i.e. the support of the input multiset
ARB_ARBOR_API locset support(locset);

} // namespace ls

// Union of two locsets.
ARB_ARBOR_API locset join(locset, locset);

// Multiset sum of two locsets.
ARB_ARBOR_API locset sum(locset, locset);

} // namespace arb
