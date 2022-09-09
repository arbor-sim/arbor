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
struct region_tag {};

class ARB_SYMBOL_VISIBLE region {
public:
    template <typename Impl,
              typename = std::enable_if_t<std::is_base_of<region_tag, std::decay_t<Impl>>::value>>
    explicit region(Impl&& impl):
        impl_(new wrap<Impl>(std::forward<Impl>(impl))) {}

   template <typename Impl,
              typename = std::enable_if_t<std::is_base_of<region_tag, std::decay_t<Impl>>::value>>
    explicit region(const Impl& impl):
        impl_(new wrap<Impl>(impl)) {}

    region(region&& other) = default;

    // The default constructor creates an empty "nil" region.
    region();

    region(const region& other):
        impl_(other.impl_->clone()) {}

    region& operator=(const region& other) {
        impl_ = other.impl_->clone();
        return *this;
    }

    template <typename Impl,
              typename = std::enable_if_t<std::is_base_of<region_tag, std::decay_t<Impl>>::value>>
    region& operator=(Impl&& other) {
        impl_ = new wrap<Impl>(std::forward<Impl>(other));
        return *this;
    }

    template <typename Impl,
              typename = std::enable_if_t<std::is_base_of<region_tag, std::decay_t<Impl>>::value>>
    region& operator=(const Impl& other) {
        impl_ = new wrap<Impl>(other);
        return *this;
    }

    // Implicit conversion from mcable, mcable_list, or mextent.
    region(mcable);
    region(mextent);
    region(mcable_list);

    friend mextent thingify(const region& r, const mprovider& m) {
        return r.impl_->thingify(m);
    }

    friend std::ostream& operator<<(std::ostream& o, const region& p) {
        return p.impl_->print(o);
    }

    // The union of regions.
    friend region join(region, region);

    template <typename ...Args>
    friend region join(region l, region r, Args... args) {
        return join(join(std::move(l), std::move(r)), std::move(args)...);
    }

    // The intersection of regions.
    friend region intersect(region, region);

    template <typename ...Args>
    friend region intersect(region l, region r, Args... args) {
        return intersect(intersect(std::move(l), std::move(r)), std::move(args)...);
    }

private:
    struct interface {
        virtual ~interface() {}
        virtual std::unique_ptr<interface> clone() = 0;
        virtual std::ostream& print(std::ostream&) = 0;
        virtual mextent thingify(const mprovider&) = 0;
    };

    std::unique_ptr<interface> impl_;

    template <typename Impl>
    struct wrap: interface {
        explicit wrap(const Impl& impl): wrapped(impl) {}
        explicit wrap(Impl&& impl): wrapped(std::move(impl)) {}

        virtual std::unique_ptr<interface> clone() override {
            return std::make_unique<wrap<Impl>>(wrapped);
        }

        virtual mextent thingify(const mprovider& m) override {
            return thingify_(wrapped, m);
        }

        virtual std::ostream& print(std::ostream& o) override {
            return o << wrapped;
        }

        Impl wrapped;
    };
};

class locset;

namespace reg {

// An empty region.
ARB_ARBOR_API region nil();

// An explicit cable section.
ARB_ARBOR_API region cable(msize_t, double, double);

// An explicit branch.
ARB_ARBOR_API region branch(msize_t);

// Region with all segments with segment tag id.
ARB_ARBOR_API region tagged(int id);

// Region corresponding to a single segment.
ARB_ARBOR_API region segment(int id);

// Region up to `distance` distal from points in `start`.
ARB_ARBOR_API region distal_interval(locset start, double distance);

// Region up to `distance` proximal from points in `start`.
ARB_ARBOR_API region proximal_interval(locset end, double distance);

// Region with all segments with radius less than/less than or equal to r
ARB_ARBOR_API region radius_lt(region reg, double r);
ARB_ARBOR_API region radius_le(region reg, double r);

// Region with all segments with radius greater than/greater than or equal to r
ARB_ARBOR_API region radius_gt(region reg, double r);
ARB_ARBOR_API region radius_ge(region reg, double r);

// Region with all segments with projection less than/less than or equal to r
ARB_ARBOR_API region z_dist_from_root_lt(double r);
ARB_ARBOR_API region z_dist_from_root_le(double r);

// Region with all segments with projection greater than/greater than or equal to r
ARB_ARBOR_API region z_dist_from_root_gt(double r);
ARB_ARBOR_API region z_dist_from_root_ge(double r);

// Region with all segments in a cell.
ARB_ARBOR_API region all();

// Region including all covers of included fork points.
// (Pre-image of projection onto the topological tree.)
ARB_ARBOR_API region complete(region);

// Region associated with a name.
ARB_ARBOR_API region named(std::string);

} // namespace reg

// Union of two regions.
ARB_ARBOR_API region join(region, region);

// Intersection of two regions.
ARB_ARBOR_API region intersect(region, region);

// Closed complement of a region.
ARB_ARBOR_API region complement(region);

// (Closure of) set difference of two regions.
ARB_ARBOR_API region difference(region a, region b);

} // namespace arb
