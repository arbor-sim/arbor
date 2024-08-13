.. _units:

Units in Arbor
==============

.. note::

   This is a work in progress. The near-term goal is to make this coverage
   complete but expect some exceptions. Notably, the interfaces of individual
   mechanisms are not yet integrated, since NMODL files -- despite explicitly
   specifying units -- do not make good use of the feature.

A large part of the interface of Arbor -- both in C++ and Python -- is covered
by units of measurement. This gives the API a way to communicate the intended
units of parameters and return values, and users can largely use their preferred
system of units, as automatic conversion is provided. For performance reasons,
this extends to the outermost layer only, after which Arbor uses its own
internal system of measurement.

We leverage the `units library <https://github.com/llnl/units>`_ published by LLNL.

Arbor is focused on SI units, and we provide the following presets for both the
C++ and Python modules.

.. table:: Provided dimensionalities and units
   :widths: auto

   =============   =======
   Dimension       Unit
   =============   =======
   Temperature     Kelvin
                   Celsius
   Length          m
                   cm
                   mm
                   um
                   nm
   Time            s
                   ms
                   us
                   ns
   Resistance      Ohm
                   kOhm
                   MOhm
   Conductivity    S
                   mS
                   uS
   Current         A
                   mA
                   uA
                   nA
                   pA
   Potential       V
                   mV
   Frequency       Hz
                   kHz
   Capacity        F
                   mF
                   uF
                   nF
                   pF
   Area            m2
                   cm2
                   mm2
                   um2
                   nm2
   Charge          C
   Mol             mol
   Molarity        M = mol/l
                   mM
   Angle           rad
                   deg
   =============   =======

Further units may be derived from existing ones by mean of multiplication and
division with the obvious semantics, the existing metric prefixes, or by extending
the catalogue of units via the underlying units library.

.. table:: Provided metric prefixes
   :widths: auto

   =============   =======   =============   =======
   Prefix          Scale     Prefix          Scale
   =============   =======   =============   =======
    pico            1e-12     kilo            1e3
    nano            1e-9      mega            1e6
    micro           1e-6      giga            1e9
    milli           1e-3
    centi           1e-2
   =============   =======   =============   =======

Parameters are passed into Arbor via a ``quantity``, which comprises a value and
a unit. We construct a quantity by multiplication of a scalar value by a unit.
Multiplication of two quantities will result in the pointwise product of the
values and units, as one would expect.

.. code-block:: python

    # two kilometers, dimension is the length
    l = 2 * km

    # three kilometers, but with the scaling factored out
    s = 3 * kilo * m

    # multiplication of two lengths gives an area
    a = l * s
    # is now 6 square kilometers

Units and quantities work intuitively and largely the same across C++ and
Python, but we provide some details below.

C++
---

Units are defined in the ``units`` namespace and exist at runtime, since we
need to cater to dynamical language bindings. In the ``units::literals``
namespace, we find user-defined literals for all units above, e.g. ``10_mV``.
Integral powers of units are constructed using the ``.pow(int)`` member, e.g.
``m2 = m.pow(2)``. Units and quantities can be converted to and from strings
using the ``std::string to_string(const T&)`` and ``T from_string_cast(const std::string&)``
functions. Conversion between different units is done like this

.. code-block:: c++

    namespace U = arb::units;

    // membrane capacitance in SI
    auto c_m = 42*U::F/U::m.pow(2) // same as 42*U::F*U::m.pow(-2)
    // convert to a different unit and extract value
    auto c_m_ = c_m.value_as(U::uF*U::cm.pow(-2))
    // invalid conversions result in NaN values, so check
    if (std::isnan(c_m_)) throw std::domain_error("Invalid value");

however, Arbor does this whenever values pass its interface.

.. cpp::namespace:: arb::units

.. cpp:class:: unit

    Describes a unit of measurement.

    .. method:: pow(int)

        Raise the unit to integral power.

.. cpp:class:: quantity

    A tuple of a value and a unit of measurement.

    .. method:: value_as(unit)

        Convert to another unit and return the converted value, possibly NaN, if
        malformed.


Python
------

Units are defined in the ``units`` sub-module. Integral powers of units are
constructed using the ``**`` operator, e.g. ``m2 = m ** 2``. Units and
quantities can be converted to a string using the ``str()`` function.
Conversion between different units is done like this

.. code-block:: python

    from arbor import units as U
    from math import isnan

    # membrane capacitance in SI
    c_m = 42*U.F/U.m**2
    # convert to a different unit and extract value
    c_m_ = c_m.value_as(U.uF*U.cm**-2)
    # invalid conversions result in NaN values, so check
    if isnan(c_m_):
        raise ValueError("Invalid value")

however, Arbor does this whenever values pass its interface.

.. currentmodule:: arbor.units

.. class:: unit

    Describes a unit of measurement.

.. class:: quantity

    A tuple of a value and a unit of measurement.

    .. method:: value_as

        Convert to another unit and return the converted value, possibly NaN, if
        malformed.
