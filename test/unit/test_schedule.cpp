#include <algorithm>
#include <random>
#include <stdexcept>
#include <vector>

#include <arbor/common_types.hpp>
#include <arbor/schedule.hpp>

#include "util/partition.hpp"
#include "util/rangeutil.hpp"

#include "common.hpp"
#include "stats.hpp"

using namespace arb;
using namespace testing;

using time_range = util::range<const time_type*>;

// Pull events from n non-contiguous subintervals of [t0, t1)
// and check for monotonicity and boundedness.

void run_invariant_checks(schedule S, time_type t0, time_type t1, unsigned n, int seed=0) {
    if (!n) return;

    std::minstd_rand R(seed);
    std::uniform_real_distribution<time_type> U(t0, t1);

    std::vector<time_type> divisions = {t0, t1};
    std::generate_n(std::back_inserter(divisions), 2*(n-1), [&] { return U(R); });
    util::sort(divisions);

    bool skip = false;
    for (auto ival: util::partition_view(divisions)) {
        if (!skip) {
            time_range ts = S.events(ival.first, ival.second);

            EXPECT_TRUE(std::is_sorted(ts.begin(), ts.end()));
            if (!ts.empty()) {
                EXPECT_LE(t0, ts.front());
                EXPECT_GT(t1, ts.back());
            }
        }
        skip = !skip;
    }
}

// Take events from n contiguous intervals comprising [t0, t1), reset, and
// then compare with events taken from a different set of contiguous
// intervals comprising [t0, t1).

void run_reset_check(schedule S, time_type t0, time_type t1, unsigned n, int seed=0) {
    if (!n) return;

    std::minstd_rand R(seed);
    std::uniform_real_distribution<time_type> U(t0, t1);

    std::vector<time_type> first_div = {t0, t1};
    std::generate_n(std::back_inserter(first_div), n-1, [&] { return U(R); });
    util::sort(first_div);

    std::vector<time_type> second_div = {t0, t1};
    std::generate_n(std::back_inserter(second_div), n-1, [&] { return U(R); });
    util::sort(second_div);

    std::vector<time_type> first;
    for (auto ival: util::partition_view(first_div)) {
        time_range ts = S.events(ival.first, ival.second);
        util::append(first, ts);
    }

    S.reset();
    std::vector<time_type> second;
    for (auto ival: util::partition_view(second_div)) {
        time_range ts = S.events(ival.first, ival.second);
        util::append(second, ts);
    }

    EXPECT_EQ(first, second);
}

static std::vector<time_type> as_vector(std::pair<const time_type*, const time_type*> ts) {
    return std::vector<time_type>(ts.first, ts.second);
}

TEST(schedule, regular) {
    // Use exact fp representations for strict equality testing.
    std::vector<time_type> expected = {0, 0.25, 0.5, 0.75, 1.0};

    schedule S = regular_schedule(0.25);
    EXPECT_EQ(expected, as_vector(S.events(0, 1.25)));

    S.reset();
    EXPECT_EQ(expected, as_vector(S.events(0, 1.25)));

    S.reset();
    expected = {0.25, 0.5, 0.75, 1.0};
    EXPECT_EQ(expected, as_vector(S.events(0.1, 1.01)));
}

TEST(schedule, regular_invariants) {
    SCOPED_TRACE("regular_invariants");
    run_invariant_checks(regular_schedule(0.3), 3, 12, 7);
}

TEST(schedule, regular_reset) {
    SCOPED_TRACE("regular_reset");
    run_reset_check(regular_schedule(0.3), 3, 12, 7);
}

TEST(schedule, regular_rounding) {
    // Test for consistent behaviour in the face of rounding at large time values.
    // Example: with t1, dt below, and int n = floor(t0/dt),
    // then n*dt is not the smallest multiple of dt greater than or equal to t0.
    // In fact, (n-4)*dt is still greater than t0.

    time_type t1 = 1802667.f;
    time_type dt = 0.024999f;

    time_type t0 = t1-10*dt;
    time_type t2 = t1+10*dt;

    schedule S = regular_schedule(t0, dt);
    auto int_l = as_vector(S.events(t0, t1));
    auto int_r = as_vector(S.events(t1, t2));

    S.reset();
    auto int_a = as_vector(S.events(t0, t2));

    EXPECT_GE(int_l.front(), t0);
    EXPECT_LT(int_l.back(), t1);

    EXPECT_GE(int_r.front(), t1);
    EXPECT_LT(int_r.back(), t2);

    EXPECT_GE(int_a.front(), t0);
    EXPECT_LT(int_a.back(), t2);

    std::vector<time_type> int_merged = int_l;
    util::append(int_merged, int_r);

    EXPECT_EQ(int_merged, int_a);
    EXPECT_TRUE(util::is_sorted(int_a));
}

TEST(schedule, explicit_schedule) {
    time_type times[] = {0.1, 0.3, 1.0, 1.25, 1.7, 2.2};
    std::vector<time_type> expected = {0.1, 0.3, 1.0};

    schedule S = explicit_schedule(times);
    EXPECT_EQ(expected, as_vector(S.events(0, 1.25)));

    S.reset();
    EXPECT_EQ(expected, as_vector(S.events(0, 1.25)));

    S.reset();
    expected = {0.3, 1.0, 1.25, 1.7};
    EXPECT_EQ(expected, as_vector(S.events(0.3, 1.71)));
}

TEST(schedule, explicit_invariants) {
    SCOPED_TRACE("explicit_invariants");

    time_type times[] = {0.1, 0.3, 0.4, 0.42, 2.1, 2.3, 6.01, 9, 9.1, 9.8, 10, 11.2, 13};
    run_invariant_checks(explicit_schedule(times), 0.4, 10.2, 5);
}

TEST(schedule, explicit_reset) {
    SCOPED_TRACE("explicit_reset");

    time_type times[] = {0.1, 0.3, 0.4, 0.42, 2.1, 2.3, 6.01, 9, 9.1, 9.8, 10, 11.2, 13};
    run_reset_check(explicit_schedule(times), 0.4, 10.2, 5);
}

// A Uniform Random Bit Generator[*] adaptor that deliberately
// skews the generated numbers by raising their quantile to
// the given power.
//
// [*] Not actually uniform.

template <typename RNG>
struct skew_adaptor {
    using result_type = typename RNG::result_type;
    static constexpr result_type min() { return RNG::min(); }
    static constexpr result_type max() { return RNG::max(); }

    explicit skew_adaptor(double power): power_(power) {}
    result_type operator()() {
        constexpr double scale = (double)(max()-min());
        constexpr double ooscale = 1./scale;

        double x = ooscale*(G_()-min());
        x = std::pow(x, power_);
        return min()+scale*x;
    }

private:
    RNG G_;
    double power_;
};

template <typename RNG>
double poisson_schedule_dispersion(int nbin, double rate_kHz, RNG& G) {
    schedule S = poisson_schedule(rate_kHz, G);

    std::vector<int> bin(nbin);
    for (auto t: time_range(S.events(0, nbin))) {
        int j = (int)t;
        if (j<0 || j>=nbin) {
            throw std::logic_error("poisson schedule result out of bounds");
        }
        ++bin[j];
    }

    summary_stats stats = summarize(bin);
    return stats.mean/stats.variance;
}

// NOTE: schedule.poisson_uniformity tests can be expected to
// fail approximately 1% of the time, if the underlying
// random sequence were allowed to vary freely.

TEST(schedule, poisson_uniformity) {
    // Run Poisson dispersion test for N=1001 with two-sided
    // χ²-test critical value α=0.01.
    //
    // Test based on: N·dispersion ~ χ²(N-1) (approximately)
    // 
    // F(chi2_lb; N-1) = α/2
    // F(chi2_ub; N-1) = 1-α/2
    //
    // Numbers taken from scipy:
    //    scipy.stats.chi2.isf(0.01/2, 1000)
    //    scipy.stats.chi2.isf(1-0.01/2, 1000)

    constexpr int N = 1001;
    //constexpr double alpha = 0.01;
    constexpr double chi2_lb = 888.56352318146696;
    constexpr double chi2_ub = 1118.9480663231843;

    std::mt19937_64 G;
    double dispersion = poisson_schedule_dispersion(N, .813, G);
    double test_value = N*dispersion;
    EXPECT_GT(test_value, chi2_lb);
    EXPECT_LT(test_value, chi2_ub);

    // Run one sample K-S test for uniformity, with critical
    // value for the finite K-S statistic Dn of α=0.01.

    schedule S = poisson_schedule(100., G);
    auto events = as_vector(S.events(0,1));
    int n = (int)events.size();
    double dn = ks::dn_statistic(events);

    EXPECT_LT(ks::dn_cdf(dn, n), 0.99);

    // Check that these tests fail for a non-Poisson
    // source.

    skew_adaptor<std::mt19937_64> W(1.5);
    dispersion = poisson_schedule_dispersion(N, .813, W);
    test_value = N*dispersion;

    EXPECT_FALSE(test_value>=chi2_lb && test_value<=chi2_ub);

    S = poisson_schedule(100., W);
    events = as_vector(S.events(0,1));
    n = (int)events.size();
    dn = ks::dn_statistic(events);

    // This test is currently failing, because we can't
    // use a sufficiently high `n` in the `dn_cdf` function
    // to get enough discrimination from the K-S test at
    // 1%. TODO: Fix this by implementing n>140 case in
    // `dn_cdf`.

    // EXPECT_GT(ks::dn_cdf(dn, n), 0.99);
}

TEST(schedule, poisson_rate) {
    // Test Poisson events over an interval against
    // corresponding Poisson distribution.

    constexpr double alpha = 0.01;
    constexpr double lambda = 123.4;

    std::mt19937_64 G;
    schedule S = poisson_schedule(lambda, G);
    int n = (int)time_range(S.events(0, 1)).size();
    double cdf = poisson::poisson_cdf_approx(n, lambda);

    EXPECT_GT(cdf, alpha/2);
    EXPECT_LT(cdf, 1-alpha/2);

    // Check that the test fails for a non-Poisson
    // source.

    skew_adaptor<std::mt19937_64> W(1.5);
    S = poisson_schedule(lambda, W);
    n = (int)time_range(S.events(0, 1)).size();
    cdf = poisson::poisson_cdf_approx(n, lambda);

    EXPECT_FALSE(cdf>=alpha/2 && cdf<=1-alpha/2);
}

TEST(schedule, poisson_invariants) {
    SCOPED_TRACE("poisson_invariants");
    std::mt19937_64 G;
    G.discard(100);
    run_invariant_checks(poisson_schedule(0.81, G), 5.1, 15.3, 7);
}

TEST(schedule, poisson_reset) {
    SCOPED_TRACE("poisson_reset");
    std::mt19937_64 G;
    G.discard(200);
    run_reset_check(poisson_schedule(.11, G), 1, 10, 7);
}

TEST(schedule, poisson_offset) {
    // Expect Poisson schedule with an offset to give exactly the
    // same sequence, after the offset, as a regular zero-based Poisson.

    const double offset = 3.3;

    std::mt19937_64 G1;
    G1.discard(300);

    std::vector<time_type> expected;
    for (auto t: as_vector(poisson_schedule(.234, G1).events(0., 100.))) {
        t += offset;
        if (t<100.) {
            expected.push_back(t);
        }
    }

    std::mt19937_64 G2;
    G2.discard(300);

    EXPECT_TRUE(seq_almost_eq<time_type>(expected,
        as_vector(poisson_schedule(offset, .234, G2).events(0., 100.))));
}

TEST(schedule, poisson_offset_reset) {
    SCOPED_TRACE("poisson_reset");
    std::mt19937_64 G;
    G.discard(400);
    run_reset_check(poisson_schedule(3.3, 9.1, G), 1, 10, 7);
}

TEST(schedule, poisson_tstop) {
    SCOPED_TRACE("poisson_tstop");
    std::mt19937_64 G;
    G.discard(500);

    const double tstop = 50;

    auto const times = as_vector(poisson_schedule(0, .234, G, tstop).events(0., 100.));
    auto const max = std::max_element(begin(times), end(times));

    EXPECT_TRUE(max != end(times));
    EXPECT_TRUE(*max <= tstop);
}

