.. _cablecell:

.. py:module:: arbor

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

Decoration
------------

To define the dynamic behavior of a cell, the electrical properties, ion
channels, synapses, stimulii and gap junctions are associated with regions and
locations on the cell. In Arbor, this *decoration* is performed through three
broad classes of action:

* Setting default properties that apply to an *entire cell*.
* *Painting* properties and dynamics on *regions* of a cell.
* *Placing* discrete items such as synapses, stimulii and spike detectors on
  *locsets* of a cell.

Cable properties and ions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every cable cell model has ion species and physical properties that must be defined
on all cables to form a consistent model.
These properties include ion species valence, the method used to calculate
ion species reversal potential, and cable properties like capacitance, resistivity
and temperature.

Parameters can be set at three different levels:

* *globally*: set default global values for all cells in a model.
* *per-cell*: overide the global defaults for a single cell.
* *per-region*: specialize values on specific cell regions.

Cable Properties
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

    cell = arbor.cable_cell(morph, labels)

    # Set cell-wide properties that will be applied by default to # the entire cell.
    cell.set_properties(Vm=-70, cm=0.02, rL=30, tempK=30+273.5)

    # Override specific values on the soma and axon
    cell.paint('soma', Vm=-50, cm=0.01, rL=35)
    cell.paint('axon', Vm=-60, rL=40)

.. warning::
    It is not possible to deterministicly find which value should be chosen for a
    parameter when it is specified separately on two overlapping regions on a cell
    using the paint interface.
    If this occurs, an *exception is thrown*.

.. note::
    No global default values are used for cable properties, which must be expliclty
    must be set by the user. It is an error when a model is constructed and any
    region of any cell has no value set at global, cell-local or region-local level.

    It was a deliberate choice to not use the same default values as NEURON; while
    some values have historic significance, for example the default
    temperature of 6.3 Celcius from early work on squid axons, they are not broadly
    applicable. Also, by explicitly setting parameters, there are less surprises
    caused by behavior that depends on hidden values.

    .. csv-table:: Default property values from NEURON
       :widths: 15, 10, 10

       **Property**, **Value**, **Units**
       *Vm*,         -65,       mV
       *cm*,         0.01,      F/m²
       *rL*,         35.4,      Ω·cm
       *tempK*,      279.45,    Kelvin

Ion species
~~~~~~~~~~~~~~~~~~~

Arbor defines calcium, pottasium and sodium ion species by default.

.. csv-table:: Default ion species in Arbor
   :widths: 15, 10, 10

   **Ion**,     **name**, **Valence**
   *Calcium*,   ca,       1
   *Pottasium*,  k,       1
   *Sodium*,    na,       2

Each ion species has the following properties:

1. *internal concentration*: concentration on interior of the membrane [mM].
2. *external concentration*: concentration on exterior of the membrane [mM].
3. *reversal potential*: reversal potential [mV].
4. *method*:  method for calculating reversal potential.

The first three  properties must be set, and the values provided will be initial
values for each quantity at the start of the simulation.

If no reversal potential mechanism is specified for an ion species, the initial
reversal potential values are maintained for the course of a simulation. Otherwise,
a provided mechanism does the work, but it is subject to some strict restrictions.
A reversal potential mechanism described in NMODL:

* May not maintain any STATE variables.
* Can only write to the "eX" value associated with an ion.
* Can not given as a POINT mechanism.

Essentially, reversal potential mechanisms must be pure functions of cellular
and ionic state.

.. note::
    Arbor imposes greater restrictions on mechanisms that update ionic reversal potentials
    than NEURON. Doing so simplifies reasoning about interactions between
    mechanisms that share ionic species, by virtue of having one mechanism, and one
    mechanism only, that calculates reversal potentials according to concentrations
    that the other mechanisms use and modify.

If a reversal potential mechanism that writes to multiple ions
is the *method* for one of the ions in the global or per-cell parameters,
it must be given for each of the ions.

Arbor's default catalogue includes a *nernst* reversal potential, which is
parameterized over a single ion, and so can be assigned to e.g. calcium:

.. code-block:: Python

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


Density mechanisms
~~~~~~~~~~~~~~~~~~~

Based on NEURON mechanisms.

Placing
--------

Synapses
~~~~~~~~~~~~~~~~~~~

Gap junction sites
~~~~~~~~~~~~~~~~~~~

Stimulii
~~~~~~~~~~~~~~~~~~~

Threshold detectors
~~~~~~~~~~~~~~~~~~~

*TODO*

* mechanisms

    * ``arbor.mechanism`` interface

* ions

    * cell and global defaults
    * concentrations, reversal potentials
    * setting locally on a region

* electrical properties

    * voltage, capacitance, etc
    * cell and global defaults

* painting

    * electrical properties
    * mechanisms
    * ion data

* placing

    * synapses
    * gap junctions sites
    * stimulii
    * threshold detectors

* probes


Python API
----------

Creating a cable cell
------------------------------

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

.. py:class:: mechanism

    Mechanisms describe physical processes, distributed over the membrane of the cell.
    *Density mechanisms* are associated with regions of the cell, whose dynamics are
    a function of the cell state and their own state where they are present.
    *Point mechanisms* are defined at discrete locations on the cell, which receive
    events from the network.
    A third, specific type of density mechanism, which describes ionic reversal potential
    behaviour, can be specified for cells or the whole model.

    The :class:`mechanism` type is a simple wrapper around a mechanism
    :attr:`mechanism.name` and a dictionary of named parameters.

    Mechanisms have two types of parameters:

    * global parameters: a scalar value that is the same for all instances
      of a mechanism.
    * range parameters: the value of range parameters is defined for each instance
      of the mechanism on a cell. For density mechanisms, this means one value for
      each compartment on which it is present.
      value per instance.

    The method for setting a paremeter varies depending on its type.
    If global parameters change, we are effectively defining a new type
    of mechanism, so global parameter information is encoded in the
    name.
    Range parameters are set using a dictionary of name-value pairs.

    .. code-block:: Python

        import arbor

        # hh dynamics with default parameters.
        hh = arbor.mechanism('hh')

        # A passive leaky channel with custom parameters
        pas = arbor.mechanism('pas', {'e': -55, 'gl': 0.02})

        # Reversal potential using Nernst equation with GLOBAL parameter values
        # for Faraday's constant and the target ion species, set with a '/' followed
        # by comma-separated list of parameter after the base mechanism name.
        rev = arbor.mechanism('nernst/F=96485,x=ca')

    .. py:method:: set(name, value)

        Set a parameter. Adds a new parameter if a parameter of the same
        name does not exist, or overwrites the existing value if one does.

        :param str name: name of the parameter.
        :param float value: value of the parameter.

    .. py:attribute:: name
        :type: str

        The name of the mechanism.

    .. py:attribute:: values
        :type: dict

        A dictionary of key-value pairs for the parameters.
