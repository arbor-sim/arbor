#pragma once

// Embedding of cell morphology as 1-d tree with piecewise linear radius.

#include <vector>

#include <arbor/morph/morphology.hpp>
#include <arbor/morph/primitives.hpp>

namespace arb {

struct embed_pwlin1d_data;

namespace util {
template <typename X> struct pw_elements;
}

using pw_constant_fn = util::pw_elements<double>;

struct embed_pwlin1d {
    explicit embed_pwlin1d(const arb::morphology& m);

    mlocation sample_location(msize_t sid) const {
        return sample_locations_.at(sid);
    }

    // Interpolated radius at location.
    double radius(mlocation) const;

    // Computed length of mcable.
    double integrate_length(mcable c) const;
    double integrate_length(msize_t bid, const pw_constant_fn&) const;

    // Membrane surface area of given mcable.
    double integrate_area(mcable c) const;
    double integrate_area(msize_t bid, const pw_constant_fn&) const;

    // Integrated cross-sectional area of given mcable.
    double integrate_ixa(mcable c) const;
    double integrate_ixa(msize_t bid, const pw_constant_fn&) const;

    // Length of whole branch.
    double branch_length(msize_t bid) const {
        return integrate_length(mcable{bid, 0, 1});
    }

private:
    std::vector<mlocation> sample_locations_;
    std::shared_ptr<embed_pwlin1d_data> data_;
};

} // namespace arb


