#include <pybind11/pybind11.h>
#include <pybind11/operators.h>

#include <arbor/units.hpp>

namespace pyarb {

namespace py = pybind11;

void register_units(py::module& m) {
    using namespace py::literals;

    auto u = m.def_submodule("units", "Units and quantities for driving the user interface.");

    py::class_<arb::units::unit> unit(u, "unit", "A unit.");
    py::class_<arb::units::quantity> quantity(u, "quantity", "A quantity, comprising a magnitude and a unit.");

    unit
        .def(py::self * py::self)
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def(py::self / py::self)
        .def(py::self * double())
        .def(py::self / double())
        .def(double() * py::self)
        .def(double() / py::self)
        .def("__pow__", [](const arb::units::unit &b, int e) { return b.pow(e); }, py::is_operator())
        .def("__str__",
             [](const arb::units::unit& u) { return arb::units::to_string(u) ; },
             "Convert unit to string.")
        .def("__repr__",
             [](const arb::units::unit& u) { return arb::units::to_string(u) ; },
             "Convert unit to string.");

    quantity
        .def(py::self * py::self)
        .def(py::self / py::self)
        .def(py::self == py::self)
        .def(py::self != py::self)
        .def(py::self + py::self)
        .def(py::self - py::self)
        .def(py::self * double())
        .def(py::self / double())
        .def(double() * py::self)
        .def(double() / py::self)
        .def(py::self * arb::units::unit())
        .def(py::self / arb::units::unit())
        .def("__pow__", [](const arb::units::unit &b, int e) { return b.pow(e); }, py::is_operator())
        .def("value_as",
             [](const arb::units::quantity& q, const arb::units::unit& u) { return q.value_as(u); },
             "unit"_a,
             "Convert quantity to given unit and return magnitude.")
        .def_property_readonly("value",
                               [](const arb::units::quantity& q) { return q.value(); },
                               "Return magnitude.")
        .def_property_readonly("units",
                               [](const arb::units::quantity& q) { return q.units(); },
                               "Return units.")
        .def("__str__",
             [](const arb::units::quantity& q) { return arb::units::to_string(q) ; },
             "Convert quantity to string.")
        .def("__repr__",
             [](const arb::units::quantity& q) { return arb::units::to_string(q) ; },
             "Convert quantity to string.");

    u.attr("m")   = py::cast(arb::units::m);
    u.attr("cm")  = py::cast(arb::units::cm);
    u.attr("mm")  = py::cast(arb::units::mm);
    u.attr("um")  = py::cast(arb::units::um);
    u.attr("nm")  = py::cast(arb::units::nm);

    u.attr("s")   = py::cast(arb::units::s);
    u.attr("ms")  = py::cast(arb::units::ms);
    u.attr("us")  = py::cast(arb::units::us);
    u.attr("ns")  = py::cast(arb::units::ns);
    u.attr("Hz")  = py::cast(arb::units::Hz);
    u.attr("kHz") = py::cast(arb::units::kHz);

    u.attr("Ohm")  = py::cast(arb::units::Ohm);
    u.attr("kOhm") = py::cast(arb::units::kOhm);
    u.attr("MOhm") = py::cast(arb::units::MOhm);

    u.attr("S")  = py::cast(arb::units::S);
    u.attr("mS")  = py::cast(arb::units::mS);
    u.attr("uS")  = py::cast(arb::units::uS);

    u.attr("F")  = py::cast(arb::units::F);
    u.attr("uF") = py::cast(arb::units::uF);
    u.attr("nF") = py::cast(arb::units::nF);
    u.attr("pF") = py::cast(arb::units::pF);

    u.attr("A")  = py::cast(arb::units::A);
    u.attr("mA") = py::cast(arb::units::mA);
    u.attr("uA") = py::cast(arb::units::uA);
    u.attr("nA") = py::cast(arb::units::nA);
    u.attr("pA") = py::cast(arb::units::pA);

    u.attr("V")  = py::cast(arb::units::V);
    u.attr("mV") = py::cast(arb::units::mV);

    u.attr("C")  = py::cast(arb::units::C);

    u.attr("rad") = py::cast(arb::units::rad);
    u.attr("deg") = py::cast(arb::units::deg);

    u.attr("Kelvin")  = py::cast(arb::units::Kelvin);
    u.attr("Celsius") = py::cast(arb::units::Celsius);

    u.attr("mol") = py::cast(arb::units::mol);
    u.attr("M")   = py::cast(arb::units::M);
    u.attr("mM")  = py::cast(arb::units::mM);

    u.attr("pico")  = py::cast(arb::units::pico);
    u.attr("nano")  = py::cast(arb::units::nano);
    u.attr("micro") = py::cast(arb::units::micro);
    u.attr("milli") = py::cast(arb::units::milli);
    u.attr("kilo")  = py::cast(arb::units::kilo);
    u.attr("mega")  = py::cast(arb::units::mega);
    u.attr("giga")  = py::cast(arb::units::giga);
}
} // pyarb
