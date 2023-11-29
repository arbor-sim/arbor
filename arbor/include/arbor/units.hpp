#pragma once

#include <units/units.hpp>

namespace arb::units {
using quantity = ::units::measurement;
using ::units::unit;
using ::units::to_string;

using ::units::pico;
using ::units::nano;
using ::units::micro;
using ::units::milli;
using ::units::centi;

using ::units::kilo;
using ::units::mega;
using ::units::giga;

using ::units::deg;
using ::units::rad;

constexpr auto Celsius = ::units::degC;
constexpr auto Kelvin  = ::units::Kelvin;

using ::units::s;
using ::units::ms;
constexpr auto us = micro * s;
using ::units::ns;

inline quantity operator ""_s(long double v) { return v*s; }
inline quantity operator ""_ms(long double v) { return v*ms; }
inline quantity operator ""_us(long double v) { return v*us; }
inline quantity operator ""_ns(long double v) { return v*ns; }

using ::units::m;
constexpr auto cm = centi * m;
constexpr auto mm = milli * m;
constexpr auto um = micro * m;
constexpr auto nm = nano  * m;

inline quantity operator ""_m(long double v) { return v*m; }
inline quantity operator ""_cm(long double v) { return v*cm; }
inline quantity operator ""_mm(long double v) { return v*mm; }
inline quantity operator ""_um(long double v) { return v*um; }
inline quantity operator ""_nm(long double v) { return v*nm; }

constexpr auto  Ohm = ::units::ohm;
constexpr auto kOhm = kilo * Ohm;
constexpr auto MOhm = mega * Ohm;

inline quantity operator ""_Ohm(long double v) { return v*Ohm; }
inline quantity operator ""_kOhm(long double v) { return v*kOhm; }
inline quantity operator ""_MOhm(long double v) { return v*MOhm; }

// Siemens
constexpr auto S  = Ohm.pow(-1);
constexpr auto mS = milli * S;
constexpr auto uS = micro * S;

inline quantity operator ""_S(long double v) { return v*S; }
inline quantity operator ""_mS(long double v) { return v*mS; }
inline quantity operator ""_uS(long double v) { return v*uS; }

constexpr auto  A = ::units::Ampere;
constexpr auto mA = milli * A;
constexpr auto uA = micro * A;
constexpr auto nA = nano  * A;
constexpr auto pA = pico  * A;

inline quantity operator ""_A(long double v) { return v*A; }
inline quantity operator ""_mA(long double v) { return v*mA; }
inline quantity operator ""_uA(long double v) { return v*uA; }
inline quantity operator ""_nA(long double v) { return v*nA; }
inline quantity operator ""_pA(long double v) { return v*pA; }

constexpr auto V = ::units::volt;
constexpr auto mV = milli * V;

inline quantity operator ""_V(long double v) { return v*V; }
inline quantity operator ""_mV(long double v) { return v*mV; }

constexpr auto  Hz = ::units::second.pow(-1);
constexpr auto kHz = kilo * Hz;

inline quantity operator ""_Hz(long double v) { return v*Hz; }
inline quantity operator ""_kHz(long double v) { return v*kHz; }

constexpr auto F  = ::units::farad;
constexpr auto mF = milli * F;
constexpr auto uF = micro * F;
constexpr auto nF = nano  * F;
constexpr auto pF = pico  * F;

inline quantity operator ""_F(long double v) { return v*F; }
inline quantity operator ""_mF(long double v) { return v*mF; }
inline quantity operator ""_uF(long double v) { return v*uF; }
inline quantity operator ""_nF(long double v) { return v*nF; }
inline quantity operator ""_pF(long double v) { return v*pF; }

// Coulomb
constexpr auto C = ::units::coulomb;

inline quantity operator ""_C(long double v) { return v*C; }

// mol and molarity
using ::units::mol;
constexpr auto M = mol / m.pow(3);
constexpr auto mM = milli * M;

inline quantity operator ""_mol(long double v) { return v*mol; }
inline quantity operator ""_M(long double v) { return v*M; }
inline quantity operator ""_mM(long double v) { return v*mM; }

using ::units::is_valid;

}
