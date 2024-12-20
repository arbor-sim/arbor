#pragma once

#include <units/units.hpp>

namespace arb::units {

using quantity = ::units::precise_measurement;

// Allow unary minus on quantities. Seemingly doesn't catch literals such as -10_mV
inline quantity operator-(const quantity& q) { return (-1*q); }

using unit = ::units::precise_unit;
using ::units::to_string;
using ::units::unit_cast_from_string;


using ::units::precise::pico;
using ::units::precise::nano;
using ::units::precise::micro;
using ::units::precise::milli;
using ::units::precise::centi;
using ::units::precise::deci;

using ::units::precise::kilo;
using ::units::precise::mega;
using ::units::precise::giga;

using ::units::precise::deg;
using ::units::precise::rad;

constexpr inline auto Celsius = ::units::precise::degC;
constexpr inline auto Kelvin  = ::units::precise::Kelvin;

using ::units::precise::s;
using ::units::precise::ms;
constexpr inline auto us = micro * s;
using ::units::precise::ns;

using ::units::precise::m;
constexpr inline auto cm = centi * m;
constexpr inline auto mm = milli * m;
constexpr inline auto um = micro * m;
constexpr inline auto nm = nano  * m;

constexpr inline auto  Ohm = ::units::precise::ohm;
constexpr inline auto kOhm = kilo * Ohm;
constexpr inline auto MOhm = mega * Ohm;

// Siemens
constexpr inline auto S  = Ohm.pow(-1);
constexpr inline auto mS = milli * S;
constexpr inline auto uS = micro * S;

constexpr inline auto  A = ::units::precise::Ampere;
constexpr inline auto mA = milli * A;
constexpr inline auto uA = micro * A;
constexpr inline auto nA = nano  * A;
constexpr inline auto pA = pico  * A;

constexpr inline auto V = ::units::precise::volt;
constexpr inline auto mV = milli * V;
constexpr inline auto uV = micro * V;

constexpr inline auto  Hz = ::units::precise::second.pow(-1);
constexpr inline auto kHz = kilo * Hz;

constexpr inline auto F  = ::units::precise::farad;
constexpr inline auto mF = milli * F;
constexpr inline auto uF = micro * F;
constexpr inline auto nF = nano  * F;
constexpr inline auto pF = pico  * F;

constexpr inline auto m2 = m*m;
constexpr inline auto cm2 = cm*cm;
constexpr inline auto mm2 = mm*mm;
constexpr inline auto um2 = um*um;
constexpr inline auto nm2 = nm*nm;

constexpr inline auto nil = ::units::precise::one;

// Coulomb
constexpr inline auto C = ::units::precise::coulomb;

// mol and molarity
using ::units::precise::mol;
using ::units::precise::L;
constexpr inline auto M = mol / L;
constexpr inline auto mM = milli * M;

using ::units::is_valid;

namespace literals {
constexpr inline quantity operator ""_s(long double v) { return v*s; }
constexpr inline quantity operator ""_ms(long double v) { return v*ms; }
constexpr inline quantity operator ""_us(long double v) { return v*us; }
constexpr inline quantity operator ""_ns(long double v) { return v*ns; }

constexpr inline quantity operator ""_m(long double v) { return v*m; }
constexpr inline quantity operator ""_cm(long double v) { return v*cm; }
constexpr inline quantity operator ""_mm(long double v) { return v*mm; }
constexpr inline quantity operator ""_um(long double v) { return v*um; }
constexpr inline quantity operator ""_nm(long double v) { return v*nm; }

constexpr inline quantity operator ""_m2(long double v) { return v*m2; }
constexpr inline quantity operator ""_cm2(long double v) { return v*cm2; }
constexpr inline quantity operator ""_mm2(long double v) { return v*mm2; }
constexpr inline quantity operator ""_um2(long double v) { return v*um2; }
constexpr inline quantity operator ""_nm2(long double v) { return v*nm2; }

constexpr inline quantity operator ""_Ohm(long double v) { return v*Ohm; }
constexpr inline quantity operator ""_kOhm(long double v) { return v*kOhm; }
constexpr inline quantity operator ""_MOhm(long double v) { return v*MOhm; }

constexpr inline quantity operator ""_S(long double v) { return v*S; }
constexpr inline quantity operator ""_mS(long double v) { return v*mS; }
constexpr inline quantity operator ""_uS(long double v) { return v*uS; }

constexpr inline quantity operator ""_A(long double v) { return v*A; }
constexpr inline quantity operator ""_mA(long double v) { return v*mA; }
constexpr inline quantity operator ""_uA(long double v) { return v*uA; }
constexpr inline quantity operator ""_nA(long double v) { return v*nA; }
constexpr inline quantity operator ""_pA(long double v) { return v*pA; }

constexpr inline quantity operator ""_V(long double v) { return v*V; }
constexpr inline quantity operator ""_mV(long double v) { return v*mV; }

constexpr inline quantity operator ""_Hz(long double v) { return v*Hz; }
constexpr inline quantity operator ""_kHz(long double v) { return v*kHz; }

constexpr inline quantity operator ""_F(long double v)  { return v*F; }
constexpr inline quantity operator ""_mF(long double v) { return v*mF; }
constexpr inline quantity operator ""_uF(long double v) { return v*uF; }
constexpr inline quantity operator ""_nF(long double v) { return v*nF; }
constexpr inline quantity operator ""_pF(long double v) { return v*pF; }

constexpr inline quantity operator ""_mol(long double v) { return v*mol; }
constexpr inline quantity operator ""_M(long double v) { return v*M; }
constexpr inline quantity operator ""_mM(long double v) { return v*mM; }

constexpr inline quantity operator ""_C(long double v) { return v*C; }

constexpr inline quantity operator ""_s(unsigned long long v) { return v*s; }
constexpr inline quantity operator ""_ms(unsigned long long v) { return v*ms; }
constexpr inline quantity operator ""_us(unsigned long long v) { return v*us; }
constexpr inline quantity operator ""_ns(unsigned long long v) { return v*ns; }

constexpr inline quantity operator ""_m(unsigned long long v) { return v*m; }
constexpr inline quantity operator ""_cm(unsigned long long v) { return v*cm; }
constexpr inline quantity operator ""_mm(unsigned long long v) { return v*mm; }
constexpr inline quantity operator ""_um(unsigned long long v) { return v*um; }
constexpr inline quantity operator ""_nm(unsigned long long v) { return v*nm; }

constexpr inline quantity operator ""_m2(unsigned long long v) { return v*m2; }
constexpr inline quantity operator ""_cm2(unsigned long long v) { return v*cm2; }
constexpr inline quantity operator ""_mm2(unsigned long long v) { return v*mm2; }
constexpr inline quantity operator ""_um2(unsigned long long v) { return v*um2; }
constexpr inline quantity operator ""_nm2(unsigned long long v) { return v*nm2; }

constexpr inline quantity operator ""_Ohm(unsigned long long v) { return v*Ohm; }
constexpr inline quantity operator ""_kOhm(unsigned long long v) { return v*kOhm; }
constexpr inline quantity operator ""_MOhm(unsigned long long v) { return v*MOhm; }

constexpr inline quantity operator ""_S(unsigned long long v) { return v*S; }
constexpr inline quantity operator ""_mS(unsigned long long v) { return v*mS; }
constexpr inline quantity operator ""_uS(unsigned long long v) { return v*uS; }

constexpr inline quantity operator ""_A(unsigned long long v) { return v*A; }
constexpr inline quantity operator ""_mA(unsigned long long v) { return v*mA; }
constexpr inline quantity operator ""_uA(unsigned long long v) { return v*uA; }
constexpr inline quantity operator ""_nA(unsigned long long v) { return v*nA; }
constexpr inline quantity operator ""_pA(unsigned long long v) { return v*pA; }

constexpr inline quantity operator ""_V(unsigned long long v) { return v*V; }
constexpr inline quantity operator ""_mV(unsigned long long v) { return v*mV; }

constexpr inline quantity operator ""_Hz(unsigned long long v) { return v*Hz; }
constexpr inline quantity operator ""_kHz(unsigned long long v) { return v*kHz; }

constexpr inline quantity operator ""_F(unsigned long long v)  { return v*F; }
constexpr inline quantity operator ""_mF(unsigned long long v) { return v*mF; }
constexpr inline quantity operator ""_uF(unsigned long long v) { return v*uF; }
constexpr inline quantity operator ""_nF(unsigned long long v) { return v*nF; }
constexpr inline quantity operator ""_pF(unsigned long long v) { return v*pF; }

constexpr inline quantity operator ""_mol(unsigned long long v) { return v*mol; }
constexpr inline quantity operator ""_M(unsigned long long v) { return v*M; }
constexpr inline quantity operator ""_mM(unsigned long long v) { return v*mM; }

constexpr inline quantity operator ""_C(unsigned long long v) { return v*C; }
} // literals
} // units
