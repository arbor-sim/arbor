#pragma once

// Constants/magic numbers for the approximation of
// exponential, trigonometric and special functions.

namespace arb {
namespace simd_detail {

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

// Cancellation-corrected ln(2) = ln2_C1 + ln2_ C2:

constexpr double ln2C1 = 6.93145751953125E-1;
constexpr double ln2C2 = 1.42860682030941723212E-6;

// 1/ln(2):

constexpr double ln2inv = 1.4426950408889634073599;

// Min and max argument values for finite and normal
// double-precision exp

const double exp_minarg = -708.3964185322641;
const double exp_maxarg = 709.782712893384;


} // namespace simd_detail
} // namespace arb
