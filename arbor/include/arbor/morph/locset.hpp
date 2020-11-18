#pragma once

#include <cstdlib>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <arbor/morph/primitives.hpp>
#include <arbor/morph/morphology.hpp>

namespace arb {

struct mprovider;

class locset;
class locset_tag {};

class locset {
public:
    template <typename Impl,
              typename = std::enable_if_t<std::is_base_of<locset_tag, std::decay_t<Impl>>::value>>
    explicit locset(Impl&& impl):
        impl_(new wrap<Impl>(std::forward<Impl>(impl))) {}

    template <typename Impl>
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

    // Implicitly convert string to named locset expression.
    locset(const std::string& label);
    locset(const char* label);

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
locset location(msize_t branch, double pos);

// Set of terminal nodes on a morphology.
locset terminal();

// The root node of a morphology.
locset root();

// Named locset.
locset named(std::string);

// The null (empty) set.
locset nil();

// Most distal points of a region.
locset most_distal(region reg);

// Most proximal points of a region.
locset most_proximal(region reg);

// Boundary points of a region.
locset boundary(region reg);

// Completed boundary points of a region.
// (Boundary of completed components.)
locset cboundary(region reg);

// Returns all locations in a locset that are also in the region.
locset restrict(locset ls, region reg);

// Returns locations that mark the segments.
locset segment_boundaries();

// A range `left` to `right` of randomly selected locations with a
// uniform distribution from region `reg` generated using `seed`
locset uniform(region reg, unsigned left, unsigned right, uint64_t seed);

// Proportional location on every branch.
locset on_branches(double pos);

// Proportional locations on each component:
// For each component C of the region, find locations L
// s.t. dist(h, L) = r * max {dist(h, t) | t is a distal point in C}.
locset on_components(double relpos, region reg);

// Support of a locset (x s.t. x in locset).
locset support(locset);

} // namespace ls

// Union of two locsets.
locset join(locset, locset);

// Multiset sum of two locsets.
locset sum(locset, locset);

} // namespace arb
