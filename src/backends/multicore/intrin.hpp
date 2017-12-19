//
// Custom transcendental intrinsics
//
// Implementation inspired by the Cephes library:
//    - http://www.netlib.org/cephes/

#pragma once

#include <iostream>
#include <limits>

#include <immintrin.h>

namespace arb {
namespace multicore {

namespace detail {

constexpr double exp_limit = 708;

// P/Q polynomial coefficients for the exponential function
constexpr double P0exp = 9.99999999999999999910E-1;
constexpr double P1exp = 3.02994407707441961300E-2;
constexpr double P2exp = 1.26177193074810590878E-4;

constexpr double Q0exp = 2.00000000000000000009E0;
constexpr double Q1exp = 2.27265548208155028766E-1;
constexpr double Q2exp = 2.52448340349684104192E-3;
constexpr double Q3exp = 3.00198505138664455042E-6;

// P/Q polynomial coefficients for the log function
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
constexpr double ln2inv = 1.4426950408889634073599; // 1/ln(2)

// C1 + C2 = ln(2)
constexpr double C1 = 6.93145751953125E-1;
constexpr double C2 = 1.42860682030941723212E-6;

// C4 - C3 = ln(2)
constexpr double C3 = 2.121944400546905827679e-4;
constexpr double C4 = 0.693359375;

constexpr uint64_t dmant_mask = ((1UL<<52) - 1) | (1UL << 63); // mantissa + sign
constexpr uint64_t dexp_mask  = ((1UL<<11) - 1) << 52;
constexpr int exp_bias = 1023;
constexpr double dsqrth = 0.70710678118654752440;
}

#include "intrin_avx2.hpp"

#if defined(SIMD_KNL) || defined(SIMD_AVX512)
#include "intrin_avx512.hpp"
#endif

} // end namespace multicore
} // end namespace arb
