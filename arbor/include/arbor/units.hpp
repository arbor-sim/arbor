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

using ::units::m;
constexpr auto cm = centi * m;
constexpr auto mm = milli * m;
constexpr auto um = micro * m;
constexpr auto nm = nano  * m;

constexpr auto  Ohm = ::units::ohm;
constexpr auto kOhm = kilo * Ohm;
constexpr auto MOhm = mega * Ohm;

constexpr auto S  = Ohm.pow(-1);
constexpr auto mS = milli * S;
constexpr auto uS = micro * S;

constexpr auto  A = ::units::Ampere;
constexpr auto mA = milli * A;
constexpr auto uA = micro * A;
constexpr auto nA = nano  * A;
constexpr auto pA = pico  * A;

constexpr auto V = ::units::volt;
constexpr auto mV = milli * V;

constexpr auto  Hz = ::units::second.pow(-1);
constexpr auto kHz = kilo * Hz;

constexpr auto F  = ::units::farad;
constexpr auto uF = micro * F;
constexpr auto nF = nano  * F;
constexpr auto pF = pico  * F;

constexpr auto C = ::units::coulomb;

using ::units::mol;
constexpr auto M = mol / m.pow(3);
constexpr auto mM = milli * M;

}
