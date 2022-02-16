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

class cable_cell;

struct cv_policy_base {
    virtual locset cv_boundary_points(const cable_cell& cell) const = 0;
    virtual region domain() const = 0;
    virtual std::unique_ptr<cv_policy_base> clone() const = 0;
    virtual ~cv_policy_base() {}
    virtual std::ostream& print(std::ostream&) = 0;
};

using cv_policy_base_ptr = std::unique_ptr<cv_policy_base>;

struct ARB_SYMBOL_VISIBLE cv_policy {
    cv_policy(const cv_policy_base& ref) { // implicit
        policy_ptr = ref.clone();
    }

    cv_policy(const cv_policy& other):
        policy_ptr(other.policy_ptr->clone()) {}

    cv_policy& operator=(const cv_policy& other) {
        policy_ptr = other.policy_ptr->clone();
        return *this;
    }

    cv_policy(cv_policy&&) = default;
    cv_policy& operator=(cv_policy&&) = default;

    locset cv_boundary_points(const cable_cell& cell) const {
        return policy_ptr->cv_boundary_points(cell);
    }

    region domain() const {
        return policy_ptr->domain();
    }

    friend std::ostream& operator<<(std::ostream& o, const cv_policy& p) {
        return p.policy_ptr->print(o);
    }

private:
    cv_policy_base_ptr policy_ptr;
};

ARB_ARBOR_API cv_policy operator+(const cv_policy&, const cv_policy&);
ARB_ARBOR_API cv_policy operator|(const cv_policy&, const cv_policy&);


// Common flags for CV policies; bitwise composable.
namespace cv_policy_flag {
    using value = unsigned;
    enum : unsigned {
        none = 0,
        interior_forks = 1<<0
    };
}

struct ARB_ARBOR_API cv_policy_explicit: cv_policy_base {
    explicit cv_policy_explicit(locset locs, region domain = reg::all()):
        locs_(std::move(locs)), domain_(std::move(domain)) {}

    cv_policy_base_ptr clone() const override;
    locset cv_boundary_points(const cable_cell&) const override;
    region domain() const override;
    std::ostream& print(std::ostream& os) override {
        os << "(explicit " << locs_ << ' ' << domain_ << ')';
        return os;
    }

private:
    locset locs_;
    region domain_;
};

struct ARB_ARBOR_API cv_policy_single: cv_policy_base {
    explicit cv_policy_single(region domain = reg::all()):
        domain_(domain) {}

    cv_policy_base_ptr clone() const override;
    locset cv_boundary_points(const cable_cell&) const override;
    region domain() const override;
    std::ostream& print(std::ostream& os) override {
        os << "(single " << domain_ << ')';
        return os;
    }

private:
    region domain_;
};

struct ARB_ARBOR_API cv_policy_max_extent: cv_policy_base {
    cv_policy_max_extent(double max_extent, region domain, cv_policy_flag::value flags = cv_policy_flag::none):
         max_extent_(max_extent), domain_(std::move(domain)), flags_(flags) {}

    explicit cv_policy_max_extent(double max_extent, cv_policy_flag::value flags = cv_policy_flag::none):
         max_extent_(max_extent), domain_(reg::all()), flags_(flags) {}

    cv_policy_base_ptr clone() const override;
    locset cv_boundary_points(const cable_cell&) const override;
    region domain() const override;
    std::ostream& print(std::ostream& os) override {
        os << "(max-extent " << max_extent_ << ' ' << domain_ << ' ' << flags_ << ')';
        return os;
    }

private:
    double max_extent_;
    region domain_;
    cv_policy_flag::value flags_;
};

struct ARB_ARBOR_API cv_policy_fixed_per_branch: cv_policy_base {
    cv_policy_fixed_per_branch(unsigned cv_per_branch, region domain, cv_policy_flag::value flags = cv_policy_flag::none):
         cv_per_branch_(cv_per_branch), domain_(std::move(domain)), flags_(flags) {}

    explicit cv_policy_fixed_per_branch(unsigned cv_per_branch, cv_policy_flag::value flags = cv_policy_flag::none):
         cv_per_branch_(cv_per_branch), domain_(reg::all()), flags_(flags) {}

    cv_policy_base_ptr clone() const override;
    locset cv_boundary_points(const cable_cell&) const override;
    region domain() const override;
    std::ostream& print(std::ostream& os) override {
        os << "(fixed-per-branch " << cv_per_branch_ << ' ' << domain_ << ' ' << flags_ << ')';
        return os;
    }

private:
    unsigned cv_per_branch_;
    region domain_;
    cv_policy_flag::value flags_;
};

struct ARB_ARBOR_API cv_policy_every_segment: cv_policy_base {
    explicit cv_policy_every_segment(region domain = reg::all()):
         domain_(std::move(domain)) {}

    cv_policy_base_ptr clone() const override;
    locset cv_boundary_points(const cable_cell&) const override;
    region domain() const override;
    std::ostream& print(std::ostream& os) override {
        os << "(every-segment " << domain_ << ')';
        return os;
    }

private:
    region domain_;
};

inline cv_policy default_cv_policy() {
    return cv_policy_fixed_per_branch(1);
}

} // namespace arb
