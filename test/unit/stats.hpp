#pragma once

#include <limits>

#include <util/meta.hpp>

/* For randomness, distribution testing we need some
 * statistical tests... */

namespace testing {

// Simple summary statistics (min, max, mean, variance)
// computed with on-line algorithm.
//
// Note that variance is population variance (i.e. with
// denominator n, not n-1).

struct summary_stats {
    int n = 0;
    double max = -std::numeric_limits<double>::infinity();
    double min = std::numeric_limits<double>::infinity();
    double mean = 0;
    double variance = 0;
};

template <typename Seq>
summary_stats summarize(const Seq& xs) {
    summary_stats s;

    for (auto x: xs) {
        ++s.n;
        s.min = x<s.min? x: s.min;
        s.max = x>s.max? x: s.max;
        double d1 = x-s.mean;
        s.mean += d1/s.n;
        double d2 = x-s.mean;
        s.variance += d1*d2;
    }

    if (s.n) {
        s.variance /= s.n;
    }
    return s;
}

// One-sample Kolmogorov-Smirnov test: test probability sample
// comes from a continuous, 1-dimensional distribution.

namespace ks {

// Compute K-S test statistic:
//
// Input: n quantiles of the sample data F(x_1) ... F(x_n) in
// increasing order, where F is the cdf of the specified distribution.
//
// Output: K-S test statistic Dn of the x_i.

template <typename Seq>
double dn_statistic(const Seq& qs) {
    double n = static_cast<double>(std::size(qs));
    double d = 0;
    int j = 0;
    for (auto q: qs) {
        auto nq = n*q;
        double d1 = nq-j;
        ++j;
        double d2 = j-nq;
        d = std::max(d, std::max(d1, d2));
    }
    return d/n;
}

// One sample K-S test.
// 
// Input: K-S test statistic d, number of observations n.
// Output: P(Dn < d) where Dn ~ one-sample K–S statistic for n observations.
//
// Note: the implementation does not cover the full parameter space;
// if N>140 and n·u² in [0.3, 18], the algorithm used may give a poor
// result. The Pelz-Good approximation should be used here, but it has
// not yet been implemented.

double dn_cdf(double d, int n);

} // namespace ks

// Functions related to the Poisson distribution.

namespace poisson {

// Approximate cdf of Poisson distribution using Wilson-Hilferty transform.
//
// Input: integer sample n.
// Output: P(X < n) where X ~ Poisson(mu).

double poisson_cdf_approx(int n, double mu);

} // namespace poisson

} // namespace testing

