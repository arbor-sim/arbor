#pragma once

#include <memory>
#include <utility>

#include <arbor/export.hpp>
#include <arbor/morph/region.hpp>
#include <arbor/morph/locset.hpp>

// FVM discretization policies/hints.
//
// CV polices, given a cable cell, provide a locset comprising CV boundary
// points to be used by the discretization. The discretization need not adopt
// the boundary points 100% faithfully; for example, it may elide empty CVs or
// perform other transformations for numeric fidelity or performance reasons.
//
// The cv_policy class is a value-like wrapper for actual policies that derive
// from `cv_policy_base`. At present, there are only a handful of policies
// implemented, described below. The intent is to provide more sophisticated
// policies in the near future, specifically one based on the 'd-lambda' rule.
//
//   cv_policy_explicit:
//       Simply use the provided locset.
//
//   cv_policy_single:
//       One CV for whole region.
//
//   cv_policy_fixed_per_branch:
//       Use the same number of CVs for each branch.
//
//   cv_policy_max_extent:
//       Use as many CVs as required to ensure that no CV has
//       a length longer than a given value.
//
// The policies above can be restricted to apply only to a given region of a
// cell morphology. If a region is supplied, the CV policy is applied to the
// completion of each connected component of the morphology within the region,
// as if the subtree described by each component were itself a full cell
// morphology. The boundary points of these completed components are always
// included as boundary points provided by the policy.
//
// Except for the single and explicit policies, CV policies may also take
// various flags (implemented as bitwise orable enums) to modify their
// behaviour. In general, future CV policies may choose to ignore flag values,
// but should respect them if their semantics are relevant.
//
//   cv_policy_flag::interior_forks:
//       Position CVs so as to include fork points, as opposed to positioning
//       them so that fork points are at the boundaries of CVs.
//
// CV policy objects can be combined. For two CV polices A & B,
//
//   A + B:
//       Use the CV boundary points from both A and B.
//
//   A | B:
//       Use the CV boundary points from A except for on the region where
//       B is defined, where the boundary points from B are used.

namespace arb {

struct cable_cell;

struct cv_policy_base {
    virtual locset cv_boundary_points(const cable_cell& cell) const = 0;
    virtual region domain() const = 0;
    virtual std::unique_ptr<cv_policy_base> clone() const = 0;
    virtual ~cv_policy_base() {}
    virtual std::ostream& print(std::ostream&) = 0;
};

struct ARB_SYMBOL_VISIBLE cv_policy {
    // construct from anything except other policies
    template <typename Impl, typename = std::enable_if_t<!std::is_same_v<std::remove_cvref_t<Impl>, cv_policy>>>
    explicit cv_policy(const Impl& impl): impl_(std::make_unique<wrap<Impl>>(impl)) {}
    template <typename Impl, typename = std::enable_if_t<!std::is_same_v<std::remove_cvref_t<Impl>, cv_policy>>>
    explicit cv_policy(Impl&& impl): impl_(std::make_unique<wrap<Impl>>(std::move(impl))) {}
    // move
    cv_policy(cv_policy&&) = default;
    cv_policy& operator=(cv_policy&&) = default;
    // copy
    cv_policy(const cv_policy& other): impl_(other.impl_->clone()) {}
    cv_policy& operator=(const cv_policy& other) {
        impl_ = other.impl_->clone();
        return *this;
    }

    // interface
    locset cv_boundary_points(const cable_cell& cell) const { return impl_->cv_boundary_points(cell); }
    region domain() const { return impl_->domain(); }
    std::ostream& format(std::ostream& os) const { return impl_->format(os); }

    friend ARB_ARBOR_API std::ostream& operator<<(std::ostream& os, const cv_policy& cvp) { return cvp.format(os); }

private:
    struct iface {
        virtual locset cv_boundary_points(const cable_cell& cell) const = 0;
        virtual region domain() const = 0;
        virtual std::unique_ptr<iface> clone() const = 0;
        virtual ~iface() {}
        virtual std::ostream& format(std::ostream&) const = 0;
    };

    using iface_ptr = std::unique_ptr<iface>;

    template <typename Impl>
    struct wrap: iface {
        explicit wrap(const Impl& impl): inner_(impl) {}
        explicit wrap(Impl&& impl): inner_(std::move(impl)) {}

        locset cv_boundary_points(const cable_cell& cell) const override { return inner_.cv_boundary_points(cell); }
        region domain() const override { return inner_.domain(); };
        iface_ptr clone() const override { return std::make_unique<wrap<Impl>>(inner_); }
        std::ostream& format(std::ostream& os) const override { return inner_.format(os); };

        Impl inner_;
    };

    iface_ptr impl_;
};

// Common flags for CV policies; bitwise composable.
enum class cv_policy_flag: unsigned {
  none = 0,
  interior_forks = 1<<0
};

ARB_ARBOR_API cv_policy operator+(const cv_policy&, const cv_policy&);
ARB_ARBOR_API cv_policy operator|(const cv_policy&, const cv_policy&);

ARB_ARBOR_API cv_policy cv_policy_explicit(locset, region = reg::all());

ARB_ARBOR_API cv_policy cv_policy_max_extent(double, region, cv_policy_flag = cv_policy_flag::none);
ARB_ARBOR_API cv_policy cv_policy_max_extent(double, cv_policy_flag = cv_policy_flag::none);

ARB_ARBOR_API cv_policy cv_policy_fixed_per_branch(unsigned, region, cv_policy_flag = cv_policy_flag::none);
ARB_ARBOR_API cv_policy cv_policy_fixed_per_branch(unsigned, cv_policy_flag = cv_policy_flag::none);

ARB_ARBOR_API cv_policy cv_policy_single(region domain = reg::all());

ARB_ARBOR_API cv_policy cv_policy_every_segment(region domain = reg::all());

inline cv_policy default_cv_policy() { return cv_policy_fixed_per_branch(1); }

} // namespace arb
