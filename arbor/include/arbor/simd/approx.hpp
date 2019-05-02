#pragma once

#include <cfloat>

// Constants/magic numbers for the approximation of
// exponential, trigonometric and special functions.
//
// Polynomial coefficients and evaluation orders
// match those of the Cephes library,
// <URL: http://www.netlib.org/cephes>.
//
// Refer to the developer documentation for more
// detail concerning the approximations.

namespace arb {
namespace simd {
namespace detail {

// Exponential:
//
// Approximation of exponential by a Padé-like rational
// polynomial R(x)/R(-x) of order 6.
//
// P comprises the odd coefficients, and Q the even.

constexpr double P0exp = 9.99999999999999999910E-1;
constexpr double P1exp = 3.02994407707441961300E-2;
constexpr double P2exp = 1.26177193074810590878E-4;

constexpr double Q0exp = 2.00000000000000000009E0;
constexpr double Q1exp = 2.27265548208155028766E-1;
constexpr double Q2exp = 2.52448340349684104192E-3;
constexpr double Q3exp = 3.00198505138664455042E-6;

// Cancellation-corrected ln(2) = ln2C1 + ln2C2:

constexpr double ln2C1 = 6.93145751953125E-1;
constexpr double ln2C2 = 1.42860682030941723212E-6;

// 1/ln(2):

constexpr double ln2inv = 1.4426950408889634073599;

// Min and max argument values for finite and normal
// double-precision exponential.

constexpr double exp_minarg = -708.3964185322641;
constexpr double exp_maxarg = 709.782712893384;

// For expm1, minimum argument that gives a result
// over -1.

constexpr double expm1_minarg = -37.42994775023705;

// Logarithm:
//
// Positive denormal numbers are treated as zero
// for the purposes of the log function. 

constexpr double log_minarg = DBL_MIN;
constexpr double sqrt2 = 1.41421356237309504880;

// Cancellation-corrected ln(2) = ln2C3 + ln2C4:

constexpr double ln2C3 = 0.693359375;
constexpr double ln2C4 = -2.121944400546905827679e-4;

// Polynomial coefficients (Q is also order 5,
// but monic).

constexpr double P0log = 7.70838733755885391666E0;
constexpr double P1log = 1.79368678507819816313e1;
constexpr double P2log = 1.44989225341610930846e1;
constexpr double P3log = 4.70579119878881725854e0;
constexpr double P4log = 4.97494994976747001425e-1;
constexpr double P5log = 1.01875663804580931796e-4;

constexpr double Q0log = 2.31251620126765340583E1;
constexpr double Q1log = 7.11544750618563894466e1;
constexpr double Q2log = 8.29875266912776603211e1;
constexpr double Q3log = 4.52279145837532221105e1;
constexpr double Q4log = 1.12873587189167450590e1;

} // namespace detail
} // namespace simd
} // namespace arb
