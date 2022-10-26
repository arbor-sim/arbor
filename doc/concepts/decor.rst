.. _decor:

.. _cablecell-decoration:

Cable cell decoration
=====================

The distribution and placement of dynamics on a cable cell is called the *decor* of a cell.
A decor is composed of individual *decorations*, which associate a property or dynamic process
with a :term:`region` or :term:`locset`.
The choice of region or locset is reflected in the two broad classes of dynamics on cable cells:

* *Painted dynamics* are applied to regions of a cell, and are associated with
  an area of the membrane or volume of the cable.

  * :ref:`Cable properties <cablecell-properties>`.
  * :ref:`Density mechanisms <cablecell-density-mechs>`.
  * :ref:`Ion species <cablecell-ions>`.

* *Placed dynamics* are applied to locations on the cell, and are associated
  with entities that can be counted.

  * :ref:`Synapse mechanisms <cablecell-synapses>`.
  * :ref:`Gap junction mechanisms <cablecell-gj-mechs>`.
  * :ref:`Threshold detectors <cablecell-threshold-detectors>`
  * :ref:`Stimuli <cablecell-stimuli>`.
  * :ref:`Probes <cablecell-probes>`.

Decorations are described by a **decor** object in Arbor. It provides facilities for

* setting properties defined over the whole cell;
* descriptions of dynamics applied to regions and locsets.

.. note::

   All methods on decor objects (``paint``, ``place``, and ``set_default``)
   return a reference to the objects so you can chain them together. This saves
   some repetition. You can break long statements over multiple lines, but in
   Python this requires use of continuation lines ``\`` or wrapping the whole
   expression into parentheses.

.. _cablecell-paint:

Painted dynamics
----------------

Painted dynamics are applied to a subset of the surface or volume of cells.
They can be specified at three different levels:

* *globally*: a global default for all cells in a model.
* *per-cell*: override the global defaults for a specific cell.
* *per-region*: specialize on specific cell regions.

This hierarchical approach for resolving parameters and properties allows
us to, for example, define a global default value for calcium concentration,
then provide a different values on specific cell regions.

Some dynamics, such as membrane capacitance and the initial concentration of ion species
must be defined for all CVs. Others need only be applied where they are
present, for example ion channels.
The types of dynamics, and where they can be defined, are
:ref:`tabulated <cablecell-painted-resolution>` below.

.. _cablecell-painted-resolution:

.. csv-table:: Painted property resolution options.
   :widths: 20, 10, 10, 10

                  ,       **region**, **cell**, **global**
   cable properties,       ✓, ✓, ✓
   ion initial conditions, ✓, ✓, ✓
   density mechanism,       ✓, --, --
   scaled-mechanism (density),  ✓, --, --
   ion rev pot mechanism,  --, ✓, ✓
   ion valence,            --, --, ✓

If a property is defined at multiple levels, the most local definition will be chosen:
a cell-local definition will override a global definition, and a definition on a region
will override any cell-local or global definition on that region.

.. warning::
    If a property is defined on two regions that overlap, it is not possible to
    deterministically choose the correct definition, and an error will be
    raised during model instantiation.

.. _cablecell-properties:

1. Cable properties
~~~~~~~~~~~~~~~~~~~

There are four cable properties that must be defined everywhere on a cell:

* *Vm*: Initial membrane voltage [mV].
* *cm*: Membrane capacitance [F/m²].
* *rL*: Axial resistivity of cable [Ω·cm].
* *tempK*: Temperature [Kelvin].

Each of the cable properties can be defined as a cell-wide default, that is then
specialised on specific regions.

.. note::

    In Python, the :py:class:`decor` interface provides the :py:func:`decor.set_properties` method
    for setting cell-wide defaults for properties, and the
    :py:meth:`decor.paint` interface for overriding properties on specific regions.

    .. code-block:: Python

        import arbor

        # Create an empty decor.
        decor = arbor.decor()

        # Set cell-wide properties that will be applied by default to the entire cell.
        decor.set_properties(Vm=-70, cm=0.02, rL=30, tempK=30+273.5)

        # Override specific values on regions named "soma" and "axon".
        decor.paint('"soma"', Vm=-50, cm=0.01, rL=35)
        decor.paint('"axon"', Vm=-60, rL=40)

.. _cablecell-density-mechs:

3. Density mechanisms
~~~~~~~~~~~~~~~~~~~~~

Regions can have density mechanisms defined over their extents.
:ref:`Density mechanisms <mechanisms-density>` are a kind of
:ref:`NMODL mechanism <formatnmodl>` which describe biophysical processes.
These are processes that are distributed in space, but whose behaviour is
defined purely by the state of the cell and the process at any given point.

The most common use for density mechanisms is to describe ion channel dynamics,
for example the ``hh`` and ``pas`` mechanisms provided by NEURON and Arbor,
which model classic Hodgkin-Huxley and passive leaky currents respectively.

Mechanisms have two types of parameters that can be set by users

* *Global* parameters are a single scalar value that is the
  same everywhere a mechanism is defined.
* *Range* parameters can vary spatially.

Every mechanism is described by a string with its name, and
an optional list of key-value pairs that define its range parameters.

Because a global parameter is fixed over the entire spatial extent
of a density mechanism, a new mechanism has to be created for every
combination of global parameter values.

Take for example the built-in mechanism for passive leaky dynamics:

* Name: ``"pas"``
* Global variable: reversal potential ``"e"``.
* Range variable: conductance ``"g"``.

.. code-block:: Python

    # Create passive mechanism with default parameter values (set in NMODL file).
    m1 = arbor.mechanism('pas')

    # Create mechanism with custom conductance (range)
    m2 = arbor.mechanism('pas', {'g': 0.1})

    # Create a new passive mechanism with that changes reversal potential (global)
    m3 = arbor.mechanism('pas/e=-45')

    # Create an instance of the same mechanism, that also sets conductance (range)
    m4 = arbor.mechanism('pas/e=-45', {'g': 0.1})

    # And the mechanisms in `density` mechanism objects and add them to the decor.
    decor = arbor.decor()
    decor.paint('"soma"', arbor.density(m1))
    decor.paint('"soma"', arbor.density(m2)) # error: can't place the same mechanism on overlapping regions
    decor.paint('"soma"', arbor.density(m3)) # error: can't have overlap between two instances of a mechanism
                                             #        with different values for a global parameter.

.. _cablecell-ions:

.. _cablecell-scaled-mechs:

4. Scaled mechanisms
~~~~~~~~~~~~~~~~~~~~~
Mechanism parameters are usually homogeneous along a cell. However, sometimes it is useful to scale parameters based on inhomogeneous properties.
:ref:`Inhomogeneous expressions  <labels-iexpr>` provide a way to describe a desired scaling formula, which for example can include the cell radius or the distance to a given set of locations.
The name is inspired by NeuroML's https://docs.neuroml.org/Userdocs/Schemas/Cells.html#schema-inhomogeneousparameter.
Such an expression is evaluated along the cell and yields a scaling factor, which is multiplied with the base value of the selected parameter.
Internally, this evaluation and scaling is done at mid-points of the cable partition of the cell.
Currently, only parameters of :ref:`density mechanisms <cablecell-density-mechs>` can be scaled.


.. code-block:: Python

    # Create mechanism with custom conductance (range)
    m = arbor.mechanism('pas', {'g': 0.1})

    decor = arbor.decor()
    # paint a scaled density mechanism, where 'g' is scaled with the distance from the root.
    decor.paint('"dend"', arbor.scaled_mechanism(arbor.density(m), {'g': '(distance 1.0 (root))'}))

5. Ion species
~~~~~~~~~~~~~~

Arbor allows arbitrary ion species to be defined, to extend the default
calcium, potassium and sodium ion species.
A ion species is defined globally by its name and valence, which
can't be overridden at cell or region level.

.. csv-table:: Default ion species in Arbor
   :widths: 15, 10, 10

   **Ion**,     **name**, **Valence**
   *Calcium*,   ca,       2
   *Potassium*,  k,       1
   *Sodium*,    na,       1

Each ion species has the following properties:

1. *internal concentration*: concentration on interior of the membrane [mM].
2. *external concentration*: concentration on exterior of the membrane [mM].
3. *reversal potential*: reversal potential [mV].
4. *reversal potential mechanism*:  method for calculating reversal potential.
5. *diffusivity*: diffusion coefficient for marker concentration, defaults to zero [m^2/s].

Properties 1, 2 and 3 must be defined, and are used as the initial values for
each quantity at the start of the simulation. They are specified globally,
then specialized at cell and region level.

The reversal potential of an ion species is calculated by an
optional *reversal potential mechanism*.
If no reversal potential mechanism is specified for an ion species, the initial
reversal potential values are maintained for the course of a simulation.
Otherwise, the mechanism does the work.

Reversal potential mechanisms are density mechanisms subject to some strict restrictions.
Specifically, a reversal potential mechanism described in NMODL:

* May not maintain any STATE variables.
* Can only write to the "eX" value associated with an ion.
* Can not be a POINT mechanism.

Essentially, reversal potential mechanisms must be pure functions of cellular
and ionic state.

.. note::
    Arbor imposes greater restrictions on mechanisms that update ionic reversal potentials
    than NEURON. Doing so simplifies reasoning about interactions between
    mechanisms that share ionic species, by virtue of having one mechanism, and one
    mechanism only, that calculates reversal potentials according to concentrations
    that the other mechanisms use and modify.

If a reversal potential mechanism writes to multiple ions,
it must be given for either no ions, or all of the ions it writes.

Arbor's default catalogue includes a *nernst* reversal potential, which is
parameterized over a single ion. For example, to bind it to the calcium
ion at the cell level using the Python interface:

.. code-block:: Python

    decor = arbor.decor()

    # Method 1: create the mechanism explicitly.
    ca = arbor.mechanism('nernst/x=ca')
    decor.set_ion(ion='ca', method=ca)

    # Method 2: set directly using a string description.
    decor.set_ion(ion='ca', method='nernst/x=ca')

    cell = arbor.cable_cell(morph, decor)


The NMODL code for the
`Nernst mechanism  <https://github.com/arbor-sim/arbor/blob/master/mechanisms/mod/nernst.mod>`_
can be used as a guide for how to calculate reversal potentials.

While the reversal potential mechanism must be the same for a whole cell,
the initial concentrations and reversal potential can be localized for regions
using the *paint* interface:

.. code-block:: Python

    # decor is an arbor.decor

    # It is possible to define all of the initial condition values
    # for a ion species.
    decor.paint('(tag 1)', arbor.ion('ca', int_con=2e-4, ext_con=2.5, rev_pot=114))

    # Alternatively, one can selectively overwrite the global defaults.
    decor.paint('(tag 2)', arbor.ion('ca', rev_pot=126)

To enable diffusion of ion species along the morphology (axial diffusion) one
sets the per-species diffusivity to a positive value. It can be changed per
region and defaults to zero. This is strictly passive transport according to the
diffusion equation ``X' = ß ∆X`` where ``X`` is the species' diffusive
concentration and ``ß`` the diffusivity constant.

.. code-block:: Python

    decor = arbor.decor()
    decor.set_ion('ca', diff=23.0)
    decor.paint('"region"', 'ca', diff=42.0)

Be aware of the consequences of setting ``ß > 0`` only in some places, namely
pile-up effects similar to reflective bounds.

The diffusive concentration is *separate* from the internal concentration for
reasons innate to the cable model, which require resetting it to its starting
point at every timestep. It can be accessed from NMODL density and point
mechanisms as an independent quantity, see :ref:`NMODL mechanism <formatnmodl>`. It is
present on the full morphology if its associated diffusivity is set to a
non-zero value on any subset of the morphology, ie ``region``. It is initialised
to the value of the internal concentration at time zero.

.. _cablecell-place:

Placed dynamics
---------------

Placed dynamics are discrete countable items that affect or record the dynamics of a cell,
and are assigned to :term:`locsets <locset>`. Because locsets can contain multiple locations
on the cell, and the exact number of these locations can not be known until the model is built,
each placed dynamic is given a string label, used to refer to the group of items on the underlying
locset.

.. _cablecell-synapses:

1. Connection sites
~~~~~~~~~~~~~~~~~~~

Similar to how regions can have density mechanisms defined over their extents,
locsets can have point mechanisms placed on their individual locations.
:ref:`Point mechanisms <mechanisms-point>` are a kind of :ref:`NMODL mechanism <formatnmodl>`
which describe synaptic processes such as the ``expsyn`` mechanism provided by
NEURON and Arbor, which models an exponential synapse.

A point mechanism (synapse) can form the target of a :term:`connection` on a cell.

.. code-block:: Python

    decor = arbor.decor()

    # Create an 'expsyn' mechanism with default parameter values (set in NMODL file).
    expsyn = arbor.mechanism('expsyn')

    # Wrap the 'expsyn' mechanism in a `synapse` object and add it to the decor.
    decor.place('"syn_loc_0"', arbor.synapse(expsyn))

    # Create an 'expsyn' mechanism with default parameter values as a `synapse` object, and add it to the decor.
    decor.place('"syn_loc_1"', arbor.synapse("expsyn"))

    # Create an 'expsyn' mechanism with modified 'tau' parameter as a `synapse` object, and add it to the decor.
    decor.place('"syn_loc_2"', arbor.synapse("expsyn", {"tau": 1.0}))


.. _cablecell-threshold-detectors:

2. Threshold detectors.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Threshold detectors have a dual use: they can be used to record spike times, but are also used in propagating signals
between cells. An example where we're interested in when a threshold of ``-10 mV`` is reached:

.. code-block:: Python

    # Placing a threshold detector might look like this.
    decor = arbor.decor()
    decor.place('"root"', arbor.threshold_detector(-10), "detector")

    # At this point, "detector" could be connected to another cell,
    # and it would transmit events upon the voltage crossing the threshold.

    # Just printing those spike times goes as follows.
    sim = arbor.simulation(...)
    sim.record(arbor.spike_recording.all)
    sim.run(...)
    print("spikes:")
    for sp in sim.spikes():
        print(" ", sp)

See also :term:`threshold detector`.

.. _cablecell-gj-mechs:

3. Gap junction connection sites
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Locsets can also have junction mechanisms placed on their individual locations.
:ref:`Junction mechanisms <mechanisms-junction>` are a kind of :ref:`NMODL mechanism <formatnmodl>`
which describe gap-junction processes such as the ``gj`` mechanism provided by Arbor,
which models a basic, linear, constant-conductance based gap-junction.

A junction mechanism can form each of the endpoints of a :term:`gap junction connection`
on two separate cells.

.. code-block:: Python

    decor = arbor.decor()

    # Create a 'gj' mechanism with modified 'g' value.
    gj = arbor.mechanism("gj", {"g": 2.0})

    # Wrap the 'gj' mechanism in a `junction` object and add it to the decor.
    decor.place('"gj_loc_0"', arbor.junction(gj))

    # Create a 'gj' mechanism with modified 'g' parameter as a `junction` object, and add it to the decor.
    decor.place('"gj_loc_1"', arbor.junction("gj", {"g": 1.5}))

.. _cablecell-stimuli:

4. Stimuli
~~~~~~~~~~

A current stimulus is a DC or sinusoidal current of fixed frequency with a time-varying amplitude
governed by a piecewise-linear envelope.

The stimulus is described by three parameters:
a sequence of points (*t*\ :sub:`i`\ , *a*\ :sub:`i`\ ) describing the envelope, where the times
*t*\ :sub:`i` are in milliseconds and the amplitudes *a*\ :sub:`i` are in nanoamperes;
a frequency in kilohertz, where a value of zero denotes DC; and the phase in radians at time zero.

The stimulus starts at the first timepoint *t*\ :sub:`0` with amplitude *a*\ :sub:`0`, and the amplitude
is then interpolated linearly between successive points. The last envelope point
(*t*\ :sub:`n`\ , *a*\ :sub:`n`\ ) describes a constant amplitude *a*\ :sub:`n` from
the time *t*\ :sub:`n` onwards.

Stimulus objects in the C++ and Python interfaces have simple constructors for describing
constant stimuli and constant amplitude stimuli restricted to a fixed time interval.

.. code-block:: Python

    # Constant stimulus, amplitude 10 nA.
    decor.place('(root)', arbor.iclamp(10), "iclamp0")

    # Constant amplitude 10 nA stimulus at 20 Hz, with initial phase of π/4 radians.
    decor.place('(root)', arbor.iclamp(10, frequency=0.020, phase=math.pi/4), "iclamp1")

    # Stimulus at 1 kHz, amplitude 10 nA, for 40 ms starting at t = 30 ms.
    decor.place('(root)', arbor.iclamp(30, 40, 10, frequency=1), "iclamp2")

    # Piecewise linear stimulus with amplitude ranging from 0 nA to 10 nA,
    # starting at t = 30 ms and stopping at t = 50 ms.
    decor.place('(root)', arbor.iclamp([(30, 0), (37, 10), (43, 8), (50, 0)]), "iclamp3")


.. _cablecell-probes:

5. Probes
~~~~~~~~~

See :ref:`probesample`.

API
---

* :ref:`Python <pycablecell-decor>`
