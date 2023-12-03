"""
Units and quantities for driving the user interface.
"""
from __future__ import annotations
import typing

__all__ = [
    "A",
    "C",
    "Celsius",
    "F",
    "Hz",
    "Kelvin",
    "M",
    "MOhm",
    "Ohm",
    "S",
    "V",
    "cm",
    "cm2",
    "deg",
    "giga",
    "kHz",
    "kOhm",
    "kilo",
    "m",
    "m2",
    "mA",
    "mM",
    "mS",
    "mV",
    "mega",
    "micro",
    "milli",
    "mm",
    "mm2",
    "mol",
    "ms",
    "nA",
    "nF",
    "nano",
    "nm",
    "nm2",
    "ns",
    "pA",
    "pF",
    "pico",
    "quantity",
    "rad",
    "s",
    "uA",
    "uF",
    "uS",
    "um",
    "um2",
    "unit",
    "us",
]

class quantity:
    """
    A quantity, comprising a magnitude and a unit.
    """

    __hash__: typing.ClassVar[None] = None
    def __add__(self, arg0: quantity) -> quantity: ...
    def __eq__(self, arg0: quantity) -> bool: ...
    @typing.overload
    def __mul__(self, arg0: quantity) -> quantity: ...
    @typing.overload
    def __mul__(self, arg0: float) -> quantity: ...
    @typing.overload
    def __mul__(self, arg0: unit) -> quantity: ...
    def __ne__(self, arg0: quantity) -> bool: ...
    def __pow__(self: unit, arg0: int) -> unit: ...
    def __repr__(self) -> str:
        """
        Convert quantity to string.
        """
    def __rmul__(self, arg0: float) -> quantity: ...
    def __rtruediv__(self, arg0: float) -> quantity: ...
    def __str__(self) -> str:
        """
        Convert quantity to string.
        """
    def __sub__(self, arg0: quantity) -> quantity: ...
    @typing.overload
    def __truediv__(self, arg0: quantity) -> quantity: ...
    @typing.overload
    def __truediv__(self, arg0: float) -> quantity: ...
    @typing.overload
    def __truediv__(self, arg0: unit) -> quantity: ...
    def value_as(self, unit: unit) -> float:
        """
        Convert quantity to given unit and return magnitude.
        """
    @property
    def units(self) -> unit:
        """
        Return units.
        """
    @property
    def value(self) -> float:
        """
        Return magnitude.
        """

class unit:
    """
    A unit.
    """

    __hash__: typing.ClassVar[None] = None
    def __eq__(self, arg0: unit) -> bool: ...
    @typing.overload
    def __mul__(self, arg0: unit) -> unit: ...
    @typing.overload
    def __mul__(self, arg0: float) -> quantity: ...
    def __ne__(self, arg0: unit) -> bool: ...
    def __pow__(self, arg0: int) -> unit: ...
    def __repr__(self) -> str:
        """
        Convert unit to string.
        """
    def __rmul__(self, arg0: float) -> quantity: ...
    def __rtruediv__(self, arg0: float) -> quantity: ...
    def __str__(self) -> str:
        """
        Convert unit to string.
        """
    @typing.overload
    def __truediv__(self, arg0: unit) -> unit: ...
    @typing.overload
    def __truediv__(self, arg0: float) -> quantity: ...

A: unit  # value = A
C: unit  # value = C
Celsius: unit  # value = °C
F: unit  # value = F
Hz: unit  # value = Hz
Kelvin: unit  # value = K
M: unit  # value = mol/m^3
MOhm: unit  # value = 1/uS
Ohm: unit  # value = 1/S
S: unit  # value = S
V: unit  # value = V
cm: unit  # value = cm
cm2: unit  # value = cm^2
deg: unit  # value = deg
giga: unit  # value = 1000000000
kHz: unit  # value = kHz
kOhm: unit  # value = 1/mS
kilo: unit  # value = 1000
m: unit  # value = m
m2: unit  # value = m^2
mA: unit  # value = mA
mM: unit  # value = umol/L
mS: unit  # value = mS
mV: unit  # value = mV
mega: unit  # value = 1000000
micro: unit  # value = 9.99999997475242708e-07
milli: unit  # value = 0.00100000004749745131
mm: unit  # value = mm
mm2: unit  # value = mm^2
mol: unit  # value = mol
ms: unit  # value = ms
nA: unit  # value = nA
nF: unit  # value = nF
nano: unit  # value = 9.99999971718068537e-10
nm: unit  # value = nm
nm2: unit  # value = nm^2
ns: unit  # value = ns
pA: unit  # value = pA
pF: unit  # value = pF
pico: unit  # value = 9.999999960041972e-13
rad: unit  # value = rad
s: unit  # value = s
uA: unit  # value = uA
uF: unit  # value = uF
uS: unit  # value = uS
um: unit  # value = um
um2: unit  # value = um^2
us: unit  # value = us
