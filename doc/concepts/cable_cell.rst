.. _cablecell:

Cable cells
===========

An Arbor *cable cell* is a full :ref:`description <modelcelldesc>` of a cell
with morphology and cell dynamics like ion species and their properties, ion
channels, synapses, gap junction sites, stimuli and spike detectors.

Cable cells are constructed from three components:

* :ref:`Morphology <morph>`: a decription of the geometry and branching structure of the cell shape.
* :ref:`Label dictionary <labels>`: a set of rules that refer to regions and locations on the cell.
* :ref:`Decor <cablecell-decoration>`: a description of the dynamics on the cell, placed according to the named rules in the dictionary.

When a cable cell is constructued the following steps are performed using the inputs:

1. Concrete regions and locsets are generated for the morphology for each labeled region and locset in the dictionary
2. The default values for parameters specified in the decor, such as ion species concentration, are instantiated.
3. Dynamics (mechanisms, parameters, synapses, etc.) are instaniated on the regions and locsets as specified by the decor.

Once constructed, the cable cell can be queried for specific information about the cell, but it can't be modified (it is *immutable*).

.. Note::

    The inputs used to construct the cell (morphology, label definitions and decor) are orthogonal,
    which allows a broad range of individual cells to be constructed from a handful of simple rules
    encoded in the inputs.
    For example, take a model with the following:

    * three cell types: pyramidal, purkinje and granule.
    * two different morphologies for each cell type (a total of 6 morphologies).
    * all cells have the same basic region definitions: soma, axon, dendrites.
    * all cells of the same type (e.g. Purkinje) have the same dynamics defined on their respective regions.

    The basic building blocks required to construct all of the cells for the model would be:
    * 6 morphologies (2 for each of purkinje, granule and pyramidal).
    * 3 decors (1 for each of purkinje, granule and pyramidal).
    * 1 label dictionary that defines the region types.

.. _cablecell-decoration:

Decoration
----------

The distribution and placement of dynamics on a cable cell is called the *decor* of a cell.
A decor is composed of individual *decorations*, which associate a property or dynamic process
with a :ref:`region <labels-region>` or :ref:`locset <labels-locset>`.
The choice of region or locset is reflected in the two broad classes of dynamics on cable cells:

* *Painted dynamics* are applied to regions of a cell, and are associated with
  an area of the membrane or volume of the cable.

  * :ref:`Cable properties <cablecell-properties>`.
  * :ref:`Density mechanisms <cablecell-density-mechs>`.
  * :ref:`Ion species <cablecell-ions>`.

* *Placed dynamics* are applied to locations on the cell, and are associated
  with entities that can be counted.

  * :ref:`Synapses <cablecell-synapses>`.
  * :ref:`Gap junction sites <cablecell-gj-sites>`.
  * :ref:`Threshold detectors <cablecell-threshold-detectors>` (spike detectors).
  * :ref:`Stimuli <cablecell-stimuli>`.
  * :ref:`Probes <cablecell-probes>`.

Decorations are described by a **decor** object in Arbor.
Provides facility for
* setting properties defined over the whole cell
* descriptions of dynamics applied to regions and locsets

.. _cablecell-paint:

Painted dynamics
''''''''''''''''

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
        decor = arbor.decor

        # Set cell-wide properties that will be applied by default to the entire cell.
        decor.set_properties(Vm=-70, cm=0.02, rL=30, tempK=30+273.5)

        # Override specific values on regions named "soma" and "axon".
        decor.paint('"soma"', Vm=-50, cm=0.01, rL=35)
        decor.paint('"axon"', Vm=-60, rL=40)

.. _cablecell-density-mechs:

3. Density mechanisms
~~~~~~~~~~~~~~~~~~~~~

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

    decor = arbor.decor()
    decor.paint('"soma"', m1)
    decor.paint('"soma"', m2) # error: can't place the same mechanism on overlapping regions
    decor.paint('"soma"', m3) # error: can't have overlap between two instances of a mechanism
                              #        with different values for a global parameter.

.. _cablecell-ions:

4. Ion species
~~~~~~~~~~~~~~

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

    decor = arbor.decor()

    # Method 1: create the mechanism explicitly.
    ca = arbor.mechanism('nernst/x=ca')
    decor.set_ion(ion='ca', method=ca)

    # Method 2: set directly using a string description.
    decor.set_ion(ion='ca', method='nernst/x=ca')

    cell = arbor.cable_cell(morph, labels, decor)


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

.. _cablecell-place:

Placed dynamics
''''''''''''''''

Placed dynamics are discrete countable items that affect or record the dynamics of a cell,
and are assigned to specific locations.

.. _cablecell-synapses:

1. Connection sites
~~~~~~~~~~~~~~~~~~~

Connections (synapses) are instances of NMODL POINT mechanisms. See also :ref:`modelconnections`.

.. _cablecell-gj-sites:

2. Gap junction sites
~~~~~~~~~~~~~~~~~~~~~

See :ref:`modelgapjunctions`.

.. _cablecell-threshold-detectors:

3. Threshold detectors (spike detectors).
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _cablecell-stimuli:

4. Stimuli
~~~~~~~~~~

.. _cablecell-probes:

5. Probes
~~~~~~~~~

.. _cablecell-cv-policies:

Discretisation and CV policies
------------------------------

For the purpose of simulation, cable cells are decomposed into discrete
subcomponents called *control volumes* (CVs). The CVs are
uniquely determined by a set of *B* ``mlocation`` boundary points.
For each non-terminal point *h* in *B*, there is a CV comprising the points
{*x*: *h* ≤ *x* and ¬∃ *y* ∈ *B* s.t *h* < *y* < *x*}, where < and ≤ refer to the
geometrical partial order of locations on the morphology. A fork point is
owned by a CV if and only if all of its corresponding representative locations
are in the CV.

The set of boundary points used by the simulator is determined by a *CV policy*.

Specific CV policies are created by functions that take a ``region`` parameter
that restrict the domain of applicability of that policy; this facility is useful
for specifying differing discretisations on different parts of a cell morphology.
When a CV policy is constrained in this manner, the boundary of the domain will
always constitute part of the CV boundary point set.

``cv_policy_single``
''''''''''''''''''''

Use one CV for each connected component of a region. When applied to the whole cell
will generate single CV for the whole cell.

``cv_policy_explicit``
''''''''''''''''''''''

Define CV boundaries according to a user-supplied set of locations, optionally
restricted to a region.

``cv_policy_every_segment``
'''''''''''''''''''''''''''

Use every segment in the morphology to define CVs, optionally
restricted to a region. Each fork point in the domain is
represented by a trivial CV.

``cv_policy_fixed_per_branch``
''''''''''''''''''''''''''''''

For each branch in each connected component of the region (or the whole cell,
if no region is specified), evenly distribute boundary points along the branch so
as to produce an exact number of CVs per branch.

By default, CVs will terminate at branch ends. An optional flag
``cv_policy_flag::interior_forks`` can be passed to specify that fork points
will be included in non-trivial, branched CVs and CVs covering terminal points
in the morphology will be half-sized.


``cv_policy_max_extent``
''''''''''''''''''''''''

As for ``cv_policy_fixed_per_branch``, save that the number of CVs on any
given branch will be chosen to be the smallest number that ensures no
CV will have an extent on the branch longer than a user-provided CV length.

.. _cablecell-cv-composition:

Composition of CV policies
'''''''''''''''''''''''''''''

CV policies can be combined with ``+`` and ``|`` operators. For two policies
*A* and *B*, *A* + *B* is a policy which gives boundary points from both *A*
and *B*, while *A* | *B* is a policy which gives all the boundary points from
*B* together with those from *A* which do not within the domain of *B*.
The domain of *A* + *B* and *A* | *B* is the union of the domains of *A* and
*B*.


API
---

* :ref:`Python <pycablecell>`
* :ref:`C++ <cppcablecell>`

