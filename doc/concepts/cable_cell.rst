.. _cablecell:

Cable cells
===========

An Arbor *cable cell* is a full description of a cell with morphology and cell
dynamics, where cell dynamics include ion species and their properties, ion
channels, synapses, gap junction sites, stimuli and spike detectors.
Arbor cable cells are constructed from a morphology and a label dictionary,
and provide a rich interface for specifying the cell's dynamics.

.. note::
    The cable cell has more than *one* dedicated page, it has a few more! This page describes how to build a full
    description of a cable cell, based on three components that are broken out into their own pages:

    * :ref:`Morphology descriptions <morph-morphology>`
    * :ref:`Label dictionaries <labels-dictionary>`
    * :ref:`Mechanisms <mechanisms>`

    It can be helpful to consult those pages for some of the sections of this page.

.. _cablecell-decoration:

Decoration
----------------

A cable cell is *decorated* by specifying the distribution and placement of dynamics
on the cell. The decorations, coupled with a description of a cell morphology, are all
that is required to build a standalone single-cell model, or a cell that is part of
a larger network.

Decorations use :ref:`region <labels-region>` and :ref:`locset <labels-locset>`
descriptions, with their respective use for this purpose reflected in the two broad
classes of dynamics in Arbor:

* *Painted dynamics* are applied to regions of a cell, and are associated with
  an area of the membrane or volume of the cable.

  * :ref:`Cable properties <cable-properties>`.
  * :ref:`Density mechanisms <cable-density-mechs>`.
  * :ref:`Ion species <cable-ions>`.

* *Placed dynamics* are applied to locations on the cell, and are associated
  with entities that can be counted.

  * :ref:`Synapses <cable-synapses>`.
  * :ref:`Gap junction sites <cable-gj-sites>`.
  * :ref:`Threshold detectors <cable-threshold-detectors>` (spike detectors).
  * :ref:`Stimuli <cable-stimuli>`.
  * :ref:`Probes <cable-probes>`.

.. _cablecell-paint:

Painted dynamics
''''''''''''''''

Painted dynamics are applied to a subset of the surface and/or volume of cells.
They can be specified at three different levels:

* *globally*: a global default for all cells in a model.
* *per-cell*: override the global defaults for a specific cell.
* *per-region*: specialize on specific cell regions.

This hierarchical approach for resolving parameters and properties allows
us to, for example, define a global default value for calcium concentration,
then provide a different values on specific cell regions.

Some dynamics, such as membrane capacitance and the initial concentration of ion species
must be defined for all compartments. Others need only be applied where they are
present, for example ion channels.
The types of dynamics, and where they can be defined, are
:ref:`tabulated <cable-painted-resolution>` below.

.. _cable-painted-resolution:

.. csv-table:: Painted property resolution options.
   :widths: 20, 10, 10, 10

                  ,       **region**, **cell**, **global**
   cable properties,       ✓, ✓, ✓
   ion initial conditions, ✓, ✓, ✓
   density mechanism,       ✓, --, --
   ion rev pot mechanism,  --, ✓, ✓
   ion valence,            --, --, ✓

If a property is defined at multiple levels, the most local definition will be chosen:
a cell-local definition will override a global definition, and a definition on a region
will override any cell-local or global definition on that region.

.. warning::
    If a property is defined on two regions that overlap, it is not possible to
    deterministically choose the correct definition, and an error will be
    raised during model instantiation.

.. _cable-properties:

Cable properties
~~~~~~~~~~~~~~~~

There are four cable properties that are defined everywhere on all cables:

* *Vm*: Initial membrane voltage [mV].
* *cm*: Membrane capacitance [F/m²].
* *rL*: Axial resistivity of cable [Ω·cm].
* *tempK*: Temperature [Kelvin].

In Python, the :py:class:`cable_cell` interface provides the :py:func:`cable_cell.set_properties` method
for setting cell-wide defaults for properties, and the
:py:meth:`cable_cell.paint` interface for overriding properties on specific regions.

.. code-block:: Python

    import arbor

    # Load a morphology from file and define basic regions.
    tree = arbor.load_swc('granule.swc')
    morph = arbor.morphology(tree, spherical_root=True)
    labels = arbor.label_dict({'soma': '(tag 1)', 'axon': '(tag 2)', 'dend': '(tag 3)'})

    # Create a cable cell.
    cell = arbor.cable_cell(morph, labels)

    # Set cell-wide properties that will be applied by default to # the entire cell.
    cell.set_properties(Vm=-70, cm=0.02, rL=30, tempK=30+273.5)

    # Override specific values on the soma and axon
    cell.paint('"soma"', Vm=-50, cm=0.01, rL=35)
    cell.paint('"axon"', Vm=-60, rL=40)

.. _cable-discretisation:

Discretisation
~~~~~~~~~~~~~~~~

For the purpose of simulation, cable cells are decomposed into discrete
subcomponents called *control volumes* (CVs), following the finite volume method
terminology. Each control volume comprises a connected subset of the
morphology. Each fork point in the morphology will be the responsibility of
a single CV, and as a special case a zero-volume CV can be used to represent
a single fork point in isolation.

.. _cable-density-mechs:

Density mechanisms
~~~~~~~~~~~~~~~~~~~~~~

Regions can have density mechanisms defined over their extents.
Density mechanisms are :ref:`NMODL mechanisms <nmodl>`
which describe biophysical processes. These are processes
that are distributed in space, but whose behaviour is defined purely
by the state of the cell and the process at any given point.

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

Take for example a mechanism passive leaky dynamics:

* Name: ``"passive"``.
* Global variable: reversal potential ``"el"``.
* Range variable: conductance ``"g"``.

.. code-block:: Python

    # Create pas mechanism with default parameter values (set in NMODL file).
    m1 = arbor.mechanism('passive')

    # Create default mechanism with custom conductance (range)
    m2 = arbor.mechanism('passive', {'g': 0.1})

    # Create a new pas mechanism with that changes reversal potential (global)
    m3 = arbor.mechanism('passive/el=-45')

    # Create an instance of the same mechanism, that also sets conductance (range)
    m4 = arbor.mechanism('passive/el=-45', {'g': 0.1})

    cell.paint('"soma"', m1)
    cell.paint('"soma"', m2) # error: can't place the same mechanism on overlapping regions
    cell.paint('"soma"', m3) # error: technically a different mechanism?

.. _cable-ions:

Ion species
~~~~~~~~~~~

Arbor allows arbitrary ion species to be defined, to extend the default
calcium, potassium and sodium ion species.
A ion species is defined globally by its name and valence, which
can't be overridden at cell or region level.

.. csv-table:: Default ion species in Arbor
   :widths: 15, 10, 10

   **Ion**,     **name**, **Valence**
   *Calcium*,   ca,       1
   *Potassium*,  k,       1
   *Sodium*,    na,       2

Each ion species has the following properties:

1. *internal concentration*: concentration on interior of the membrane [mM].
2. *external concentration*: concentration on exterior of the membrane [mM].
3. *reversal potential*: reversal potential [mV].
4. *reversal potential mechanism*:  method for calculating reversal potential.

Properties 1, 2 and 3 must be defined, and are used as the initial values for
each quantity at the start of the simulation. They are specified globally,
then specialized at cell and region level.

The reversal potential of an ion species is calculated by an
optional *reversal potential mechanism*.
If no reversal potential mechanism is specified for an ion species, the initial
reversal potential values are maintained for the course of a simulation.
Otherwise, the mechanism does the work.

but it is subject to some strict restrictions.
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

If a reversal potential mechanism that writes to multiple ions,
it must be given for either no ions, or all of the ions it writes.

Arbor's default catalogue includes a *nernst* reversal potential, which is
parameterized over a single ion. For example, to bind it to the calcium
ion at the cell level using the Python interface:

.. code-block:: Python

    cell = arbor.cable_cell(morph, labels)

    # method 1: create the mechanism explicitly.
    ca = arbor.mechanism('nernst/x=ca')
    cell.set_ion(ion='ca', method=ca)

    # method 2: set directly using a string description
    cell.set_ion(ion='ca', method='nernst/x=ca')


The NMODL code for the
`Nernst mechanism  <https://github.com/arbor-sim/arbor/blob/master/mechanisms/mod/nernst.mod>`_
can be used as a guide for how to calculate reversal potentials.

While the reversal potential mechanism must be the same for a whole cell,
the initial concentrations and reversal potential can be localized for regions
using the *paint* interface:

.. code-block:: Python

    # cell is an arbor.cable_cell

    # It is possible to define all of the initial condition values
    # for a ion species.
    cell.paint('(tag 1)', arbor.ion('ca', int_con=2e-4, ext_con=2.5, rev_pot=114))

    # Alternatively, one can selectively overwrite the global defaults.
    cell.paint('(tag 2)', arbor.ion('ca', rev_pot=126)

.. _cablecell-place:

Placed dynamics
''''''''''''''''

Placed dynamics are discrete countable items that affect or record the dynamics of a cell,
and are assigned to specific locations.

.. _cable-synapses:

Connection sites
~~~~~~~~~~~~~~~~

Connections (synapses) are instances of NMODL POINT mechanisms. See also :ref:`modelconnections`.

.. _cable-gj-sites:

Gap junction sites
~~~~~~~~~~~~~~~~~~

See :ref:`modelgapjunctions`.

.. _cable-threshold-detectors:

Threshold detectors (spike detectors).
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _cable-stimuli:

Stimuli
~~~~~~~~

.. _cable-probes:

Probes
~~~~~~

API
---

* :ref:`Python <pycable_cell>`
* :ref:`C++ <cppcable_cell>`

