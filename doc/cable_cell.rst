.. _cablecell:

Cable Cells
===========

An Arbor *cable cell* is a full description of a cell with morphology and cell
dynamics, where cell dynamics include ion species and their properties, ion
channels, synapses, gap junction sites, stimulii and spike detectors.
Arbor cable cells are constructed from a morphology and a label dictionary,
and provide a rich interface for specifying the cell's dynamics.

.. note::
    Before reading this page, it is recommended that you first read about
    :ref:`morphology descriptions <morph-morphology>`, and also
    :ref:`label dictionary <labels-dictionary>` that are used to describe
    :ref:`locations <labels-locset>` and :ref:`regions <labels-region>` on a cell.

.. _cablecell-decoration:

Decoration
----------------

A cable cell is *decorated* by specifying the distribution and placement of dynamics
on the cell to produce a full description
of a cell morphology and its dynamics with all information required to build
a standalone single-cell model, or as part of a larger network.

Decoration uses region and locset descriptions, with
their respective use for this purpose reflected in the two broad classes
of dynamics in Arbor:

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
  * :ref:`Stimulii <cable-stimulii>`.
  * :ref:`Probes <cable-probes>`.

.. _cablecell-paint:

Painted Dynamics
''''''''''''''''

Painted dynamics are applied to a subset of the surface and/or volume of cells.
They can be specified at three different levels:

* *globally*: a global default for all cells in a model.
* *per-cell*: overide the global defaults for a specific cell.
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
   density mechnism,       ✓, --, --
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
    cell.paint('soma', Vm=-50, cm=0.01, rL=35)
    cell.paint('axon', Vm=-60, rL=40)

.. _cable-density-mechs:

Density mechanisms
~~~~~~~~~~~~~~~~~~~~~~

Regions can have density mechanisms defined over their extents.
Density mechanisms are :ref:`NMODL mechanisms <nmodl>`
which describe biophysical processes. These are processes
that are distributed in space, but whose behaviour is defined purely
by the state of the cell and the process at any given point.

The most common use for density mecahnisms is to describe ion channel dynamics,
for example the ``hh`` and ``pas`` mechanisms provided by NEURON and Arbor,
which model classic Hodgkin-Huxley and passive leaky currents respectively.

Mechanisms have two types of parameters that can be set by users

* *Global* parameters are a single scalar value that is the
  same everywhere a mechanism is defined.
* *Range* parameters can vary spatially.

Every mechanism is described by a string with its name, and
an optional list of key-value pairs that define its range parameters.

Because a global parameter is fixed over the entire spatial extent
of a density mechanism, a new mechanism has to created for every
combination of global parameter values.

Take for example a mechanism passive leaky dynamics:

* Name: ``"passive"``.
* Global variable: reversal potential ``"el"``.
* Range variable: conductance ``"g"``.

.. code-block:: Python

    # Create pas mechanism with default parameter values (set in NOMDL file).
    m1 = arbor.mechanism('passive')

    # Create default mechainsm with custom conductance (range)
    m2 = arbor.mechanism('passive', {'g', 0.1})

    # Create a new pas mechanism with that changes reversal potential (global)
    m3 = arbor.mechanism('passive/el=-45')

    # Create an instance of the same mechanism, that also sets conductance (range)
    m4 = arbor.mechanism('passive/el=-45', {'g', 0.1})

    cell.paint('soma', m1)
    cell.paint('soma', m2) # error: can't place the same mechanism on overlapping regions
    cell.paint('soma', m3) # error: technically a different mechanism?

.. _cable-ions:

Ion species
~~~~~~~~~~~

Arbor allows arbitrary ion species to be defined, to extend the default
calcium, potassium and sodium ion species.
A ion species is defined globally by its name and valence, which
can't be overriden at cell or region level.

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
then specialised at cell and region level.

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
the initial concentrations and reversal potential can be localised for regions
using the *paint* interface:

.. code-block:: Python

    # cell is an arbor.cable_cell

    # It is possible to define all of the initial condition values
    # for a ion species.
    cell.paint('soma', arbor.ion('ca', int_con=2e-4, ext_con=2.5, rev_pot=114))

    # Alternatively, one can selectively overwrite the global defaults.
    cell.paint('axon', arbor.ion('ca', rev_pot=126)

.. _cablecell-place:

Placed Dynamices
''''''''''''''''

Placed dynamics are discrete countable items that affect or record the dynamics of a cell,
and are asigned to specific locations.

.. _cable-synapses:

Synapses
~~~~~~~~

Synapses are instances of NMODL POINT mechanisms.

.. _cable-gj-sites:

Gap junction sites
~~~~~~~~~~~~~~~~~~

.. _cable-threshold-detectors:

Threshold detectors (spike detectors).
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _cable-stimulii:

Stimulii
~~~~~~~~

.. _cable-probes:

Probes
~~~~~~

Python API
----------

Creating a cable cell
'''''''''''''''''''''

.. py:class:: cable_cell

    A cable cell is constructed from a :ref:`morphology <morph-morphology>`
    and an optional :ref:`label dictionary <labels-dictionary>`.

    .. note::
        The regions and locsets defined in the label dictionary are
        :ref:`concretised <labels-concretise>` when the cable cell is constructed,
        and an exception will be thrown if an invalid label expression is found.

        There are two reasons an expression might be invalid:

        1. Explicitly refers to a location of cable that does not exist in the
           morphology, for example ``(branch 12)`` on a cell with 6 branches.
        2. Incorrect label reference: circular reference, or a label that does not exist.


    .. code-block:: Python

        import arbor

        # Construct the morphology from an SWC file.
        tree = arbor.load_swc('granule.swc')
        morph = arbor.morphology(tree, spherical_root=True)

        # Define regions using standard SWC tags
        labels = arbor.label_dict({'soma': '(tag 1)',
                                   'axon': '(tag 2)',
                                   'dend': '(join (tag 3) (tag 4))'})

        # Construct a cable cell.
        cell = arbor.cable_cell(morph, labels)

    .. method:: set_properties(Vm=None, cm=None, rL=None, tempK=None)

        Set default values of cable properties on the whole cell.
        Overrides the default global values, and can be overriden by painting
        the values onto regions.

        :param str region: name of the region.
        :param Vm: Initial membrane voltage [mV].
        :type Vm: float or None
        :param cm: Membrane capacitance [F/m²].
        :type cm: float or None
        :param rL: Axial resistivity of cable [Ω·cm].
        :type rL: float or None
        :param tempK: Temperature [Kelvin].
        :type tempK: float or None

        .. code-block:: Python

            # Set cell-wide values for properties
            cell.set_properties(Vm=-70, cm=0.01, rL=100, tempK=280)


    .. method:: paint(region, [Vm=None, cm=None, rL=None, tempK=None])

        Set cable properties on a region.

        :param str region: name of the region.
        :param Vm: Initial membrane voltage [mV].
        :type Vm: float or None
        :param cm: Membrane capacitance [F/m²].
        :type cm: float or None
        :param rL: Axial resistivity of cable [Ω·cm].
        :type rL: float or None
        :param tempK: Temperature [Kelvin].
        :type tempK: float or None

        .. code-block:: Python

            # Specialize resistivity on soma
            cell.paint('soma', rL=100)
            # Specialize resistivity and capacitance on the axon
            cell.paint('axon', cm=0.05, rL=80)

.. py:class:: ion

    properties of an ionic species.

