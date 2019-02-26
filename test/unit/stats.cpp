#include <cmath>
#include <numeric>
#include <vector>

#include "stats.hpp"

namespace testing {
namespace ks {

/* Compute the exact cdf F(d) of the one-sample, two-sided Kolmogorov–Smirnov
 * statisitc Dn for n observations, using the Dubin 1973 algorithm with the
 * faster implementation of Carvalho 2015.
 *
 * Bounds on numerical applicability, and the lower bound of 1-10^-15 (giving a
 * result of 1 in double precision) when n*d^2 ≥ 18 is from Simard and L'Ecuyer
 * 2011.
 *
 * This function should implement the Pelz-Good asymptopic approximation, but
 * doesn't. Instead it will throw an error if the parameters fall into a bad
 * region.
 *
 * References:
 *
 * Durbin, J. 1973. "Distribution Theory for Tests Based on the Sample
 *     Distribution Function". Society for Industrial Mathematics.
 *     ISBN: 9780898710076
 *     DOI: 10.1137/1.9781611970568.ch1
 * 
 * Carvalho, L. 2015. "An improved evaluation of Kolmogorov's distribution."
 *     Journal of Statistical Software, Code Samples 65 (3), pp. 1-8.
 *     DOI: 10.18637/jss.v065.c03
 *
 * Simard, R. and P. L'Ecuyer, 2011. "Computing the two-sided Kolmogorov-Smirnov distributiobnn."
 *     Journal of Statistical Softwarem, Articles 39 (11), pp. 1-18.
 *     DOI: 10.18637/jss.v039.i11
 */

double dn_cdf(double d, int  n) {
    // Tail cases:

    double nd = n*d;

    if (d>=1 || nd*d>=18.37) {
        return 1;
    }

    if (nd<=0.5) {
        return 0;
    }
    else if (nd<=1) {
        double x = 2*d-1./n;
        double s = x;
        for (int i = 2; i<=n; ++i) {
            s *= x*i;
        }
        return s;
    }
    else if (nd>=n-1) {
       return 1-2*std::pow(1-d, n);
    }

    // If n>=10^5, or n>140 and n·d²>0.3, throw an
    // exception rather than risk bad numeric behaviour.
    // Fix this if required by implementing Pelz-Good.

    if (n>=1e5 || (n>140 && nd*d>0.3)) {
        throw std::range_error("approximation for parameters not implemented");
    }

    int k = std::ceil(nd);
    int m = 2*k-1;
    double h = k-nd;

    // Representation of matrix H [Dubin eq 2.4.3, Carvalho eq 6].
    std::vector<double> data(3*m-2);
    double* v = data.data(); // length m
    double* q = v+m;  // length m
    double* w = q+m;  // length m-2

    // Initialize H representation.
    double r=1;
    double p=h;
    for (int i=0; i<m-1; ++i) {
        if (i<m-2) {
            w[i] = r;
        }
        v[i] = (1-p)*r;
        p *= h;
        r /= i+2;
    }
    double pp = h<=0.5? 0: std::pow(2*h-1, m);
    v[m-1] = (1-2*p+pp)*r;
    q[k-1] = 1;

    // Compute n!/n^n H(k,k).
    auto dot = [](double* a, double* b, int n) {
        return std::inner_product(a, a+n, b, 0.);
    };

    constexpr double scale = 1e140;
    constexpr double ooscale = 1/scale;
    int scale_pow = 0;

    double s = 1;
    double oon = 1.0/n;
    for (int i = 1; i<=n; ++i) {
        s = i*oon;

        double qprev = q[0];
        q[0] = s*dot(v, q, m);
        for (int j = 1; j<m-1; ++j) {
            double a = qprev;
            qprev = q[j];
            q[j] = s*(dot(w, q+j, m-j-1) + v[m-j-1]*q[m-1] + a);
        }
        q[m-1] = s*(v[0]*q[m-1] + qprev);

        if (q[k-1]>scale) {
            for (int i = 0; i<m; ++i) q[i] *= ooscale;
            ++scale_pow;
        }
        else if (q[k-1]<ooscale) {
            for (int i = 0; i<m; ++i) q[i] *= scale;
            --scale_pow;
        }
    }

    return scale_pow? q[k-1]*std::pow(scale, scale_pow): q[k-1];
}

} // namespace ks


namespace poisson {

/* Approximate cdf of Poisson(μ) by using Wilson-Hilferty transform
 * and then the normal distribution cdf.
 *
 * See e.g.:
 *
 * Terrell, G. 2003. "The Wilson–Hilferty transformation is locally saddlepoint,"
 *     Biometrika. 90 (2), pp. 445-453.
 *     DOI: 10.1093/biomet/90.2.445
 */

double poisson_cdf_approx(int n, double mu) {
    double x = std::pow(0.5+n, 2./3.);
    double norm_mu = std::pow(mu, 2./3.)*(1-1./(9*mu));
    double norm_sd = 2./3.*std::pow(mu, 1./6.)*(1+1./(24*mu));

    return 0.5+0.5*std::erf((x-norm_mu)/(norm_sd*std::sqrt(2.)));
}


} // namespace poisson

} // namespace testing
