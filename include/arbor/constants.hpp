#pragma once

namespace arb {
namespace constant {

// TODO: handle change of constants over time. Take, for example, the values
// for the universal gas constant (R) and Faraday's constant as given by NIST:
//
//         R           F
// 1973    8.31441     96486.95
// 1986    8.314510    96485.309
// 1998    8.314472    96485.3415
// 2002    8.314472    96485.3383
// 2010    8.3144621   96485.3365
// 2014    8.3144598   96485.33289

// Universal gas constant (R)
// https://physics.nist.gov/cgi-bin/cuu/Value?r
constexpr double gas_constant = 8.3144598;  //  J.K^-1.mol^-1

// Faraday's constant (F)
// https://physics.nist.gov/cgi-bin/cuu/Value?f
constexpr double faraday = 96485.33289;     // C.mol^-1

// Temperature used in original Hodgkin-Huxley paper
//      doi:10.1113/jphysiol.1952.sp004764
constexpr double hh_squid_temp = 6.3+273.15; // K

} // namespace arb
} // namespace arb
