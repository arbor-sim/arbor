#include <string>
#include <string>


#include "common.hpp"
#include "stats.hpp"

/* Unit tests for the test statistics routines */

using namespace testing;

TEST(stats, dn_statistic) {
    // Dn should equal the maximum vertical distance between the
    // empirical distribution and the straight line from (0,0) to
    // (1,1).

    // Expected: statistic=0.75 (empirical cdf should be 1 at 0.25).
    double x1[] = {0.25};
    EXPECT_DOUBLE_EQ(0.75, ks::dn_statistic(x1));

    // Expected: statistic=0.55 (empirical cdf should be 0.75 at 0.2).
    double x2[] = {0.1, 0.15, 0.2, 0.8};
    EXPECT_DOUBLE_EQ(0.55, ks::dn_statistic(x2));

    // Expected: statistic=1 (empirical cdf should be 0 at 1¯).
    double x3[] = {1.0};
    EXPECT_DOUBLE_EQ(1.0, ks::dn_statistic(x3));
}

TEST(stats, dn_cdf) {
    using std::pow;
    using std::tgamma;

    // Compute cdf F(x; n) of Dn for some known values and accuracies.
    // (See e.g. Simard and L'Ecuyer 2011.).

    // Zero and one tails:
    // 0 if x ≤ 1/2n; 1 if x ≥ 1.

    EXPECT_EQ(0.0, ks::dn_cdf(0.0, 1));
    EXPECT_EQ(0.0, ks::dn_cdf(0.0, 10));
    EXPECT_EQ(0.0, ks::dn_cdf(0.01, 10));
    EXPECT_EQ(0.0, ks::dn_cdf(0.04999, 10));
    EXPECT_EQ(0.0, ks::dn_cdf(0.00004999, 10000));
    EXPECT_EQ(1.0, ks::dn_cdf(1, 1));
    EXPECT_EQ(1.0, ks::dn_cdf(1234.45, 1));
    EXPECT_EQ(1.0, ks::dn_cdf(1, 10000));
    EXPECT_EQ(1.0, ks::dn_cdf(1234.45, 10000));

    // When x in [1/2n, 1/n), F(x; n) = n!(2x-1/n)^n.
    int n = 3;
    double x = 0.3;
    double expected = tgamma(n+1)*pow(2*x-1./n, n);
    EXPECT_NEAR(expected, ks::dn_cdf(x, n), expected*1e-15);

    // When x in [1-1/n, 1), F(x; n) = 1-2(1-x)^n.
    n = 5;
    x = 0.81;
    expected = 1-2*pow(1-x, n);
    EXPECT_NEAR(expected, ks::dn_cdf(x, n), expected*1e-15);

    // When n·x^2 > 18.37, F(x; n) should be within double epsilon of 1.
    n = 75;
    x = 0.5;
    EXPECT_EQ(1., ks::dn_cdf(x, n));

    // Various spots in the middle (avoiding n>140 until we have
    // a more complete implementation).

    n = 100;
    x = 0.2;
    expected = 1-0.000555192732802810;
    EXPECT_NEAR(expected, ks::dn_cdf(x, n), 1e-15); // note: absolute error bound

    n = 140;
    x = 0.0464158883361278;
    expected = 0.0902623294750042;
    EXPECT_NEAR(expected, ks::dn_cdf(x, n), expected*1e-14); // note: larger rel tol here
}

TEST(stats, running) {
    // Exercise simple summary statistics:
    summary_stats S;

    double x1[] = {};
    S = summarize(x1);

    EXPECT_EQ(0, S.n);
    EXPECT_EQ(0, S.mean);

    double x2[] = {3.1, 3.1, 3.1};
    S = summarize(x2);

    EXPECT_EQ(3, S.n);
    EXPECT_EQ(3.1, S.mean);
    EXPECT_EQ(0, S.variance);
    EXPECT_EQ(3.1, S.min);
    EXPECT_EQ(3.1, S.max);

    constexpr double inf = std::numeric_limits<double>::infinity();
    double x3[] = {0.3, inf, 1.};
    S = summarize(x3);

    EXPECT_EQ(3, S.n);
    EXPECT_EQ(0.3, S.min);
    EXPECT_EQ(inf, S.max);

    double x4[] = {-1, -2, 1, 0, 2};
    S = summarize(x4);

    EXPECT_EQ(5, S.n);
    EXPECT_DOUBLE_EQ(0, S.mean);
    EXPECT_DOUBLE_EQ(2, S.variance);
    EXPECT_EQ(-2, S.min);
    EXPECT_EQ(2, S.max);
}

TEST(stats, poisson_cdf) {
    // Wilson-Hilferty transform approximation of the Poisson CDF
    // is expected to be within circa 1e-3 (absolute) of true for
    // parameter μ ≥ 6.

    // Exact values in table computed with scipy.stats.
    struct point {
        int n;
        double mu;
        double cdf;
    };

    point points[] = {
        {1,   6.00,   0.017351265236665},
        {3,   6.00,   0.151203882776648},
        {6,   6.00,   0.606302782412592},
        {10,  6.00,   0.957379076417462},
        {4,   11.30,  0.012323331570747},
        {7,   11.30,  0.124852978325848},
        {11,  11.30,  0.543501467974833},
        {18,  11.30,  0.977530236363240},
        {13,  23.00,  0.017428210696087},
        {18,  23.00,  0.174768719569196},
        {23,  23.00,  0.555149935616771},
        {32,  23.00,  0.971056607478885},
        {31,  44.70,  0.019726646735143},
        {38,  44.70,  0.177614024770857},
        {44,  44.70,  0.498035947942948},
        {58,  44.70,  0.976892082270190},
        {171, 200.00, 0.019982398243966},
        {185, 200.00, 0.152411852483996},
        {200, 200.00, 0.518794309678716},
        {228, 200.00, 0.976235501016406},
    };

    for (auto p: points) {
        SCOPED_TRACE("n="+std::to_string(p.n)+"; mu="+std::to_string(p.mu));
        EXPECT_NEAR(p.cdf, poisson::poisson_cdf_approx(p.n, p.mu), 1.e-3); 
    }
}
