#pragma once

// Embedding of cell morphology as 1-d tree with piecewise linear radius.

#include <vector>

#include <arbor/export.hpp>
#include <arbor/morph/morphology.hpp>
#include <arbor/morph/primitives.hpp>

namespace arb {

struct embed_pwlin_data;

namespace util {
template <typename X> struct pw_elements;
}

// Piecewise-constant functions are represented as scalar (double)
// values defined over contiguous intervals.
using pw_constant_fn = util::pw_elements<double>;

struct ARB_ARBOR_API embed_pwlin {
    explicit embed_pwlin(const arb::morphology& m);

    // Segment queries.
    msize_t num_segments() const {
        return segment_cables_.size();
    }

    mcable segment(msize_t seg_id) const {
        return segment_cables_.at(seg_id);
    }

    const mlocation_list& segment_ends() const {
        return all_segment_ends_;
    }

    // Interpolated radius at location.
    double radius(mlocation) const;
    mcable_list radius_cmp(msize_t bid, double rad_lim, comp_op op) const;

    double directed_projection(mlocation) const;
    mcable_list projection_cmp(msize_t bid, double proj_lim, comp_op op) const;

    // Computed length of mcable.
    double integrate_length(const mcable& c) const;
    double integrate_length(mlocation proxmal, mlocation distal) const;

    double integrate_length(const mcable& c, const pw_constant_fn&) const;
    double integrate_length(msize_t bid, const pw_constant_fn&) const;

    // Membrane surface area of given mcable.
    double integrate_area(const mcable& c) const;
    double integrate_area(mlocation proxmal, mlocation distal) const;

    double integrate_area(const mcable& c, const pw_constant_fn&) const;
    double integrate_area(msize_t bid, const pw_constant_fn&) const;

    // Integrated inverse cross-sectional area of given mcable.
    double integrate_ixa(const mcable& c) const;

    double integrate_ixa(const mcable& c, const pw_constant_fn&) const;
    double integrate_ixa(msize_t bid, const pw_constant_fn&) const;

    // Length of whole branch.
    double branch_length(msize_t bid) const {
        return integrate_length(mcable{bid, 0, 1});
    }

private:
    mlocation_list all_segment_ends_;
    std::vector<mcable> segment_cables_;
    std::shared_ptr<embed_pwlin_data> data_;
};

} // namespace arb


