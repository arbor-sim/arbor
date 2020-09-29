.. _mechanisms:

Cell Mechanisms
===============

Mechanisms describe biophysical processes such as ion channels and synapses.
Mechanisms are assigned to regions and locations on a cell morphology
through a process that called :ref:`decoration <cablecell-decoration>`.
Mechanisms are described using a dialect of the :ref:`NMODL <nmodl>` domain
specific language that is similarly used in `NEURON <https://neuron.yale.edu/neuron/>`_.

Mechanism Catalogues
----------------------

A *mechanism catalogue* is a collection of mechanisms that maintains:

1. Collection of mechanism metadata indexed by name.
2. A further hierarchy of *derived* mechanisms, that allow specialization of
   global parameters, ion bindings, and implementations.
3. A map for looking up a concrete mechanism implementation on a target hardware back end.

A derived mechanism can be given a new name. Alternatively derived mechanisms can
be created implicitly.
When a mechanism name of the form ``"mech/param=value,..."`` is :ref:`requested <mechanisms-name-note>`,
if the mechanism of that name does not already exist in the catalogue, it will be
implicitly derived from an existing mechanism ``"mech"``, with global parameters
and ion bindings overridden by the supplied assignments that follow the slash.
If the mechanism in question has a single ion dependence, then that ion name
can be omitted in the assignments; ``"mech/oldion=newion"`` will make the same
derived mechanism as simply ``"mech/newion"``.

In additional being able to derive new mechanisms, catalogtues provide and interface
for looking up a mechanism by name, and querying the following properties:

* Global parameter: name, units and default value.
* Range parameters: name, units and default value.
* State variables: name, units and default value.
* Ion dependency: for each ion whether it writes concentrations or reversal potential, and
  whether the mechanism reads the reversal potential.

Default Mechanisms
''''''''''''''''''

Arbor provides a default catalogue with the following mechanisms:

* *pas*: Leaky current (:ref:`density mechanism <mechanisms-density>`).
* *hh*:  Classic Hodgkin-Huxley dynamics (:ref:`density mechanism <mechanisms-density>`).
* *nernst*: Calculate reversal potential for an ionic species using the Nernst equation (:ref:`reversal potential mechanism <mechanisms-revpot>`)
* *expsyn*: Synapse with discontinuous change in conductance at an event followed by an exponential decay (:ref:`point mechanism <mechanisms-point>`).
* *exp2syn*: Two state kinetic scheme synapse described by two time constants: rise and decay (:ref:`point mechanism <mechanisms-point>`).

With the exception of *nernst*, these mechanisms are the same as those available in NEURON.

Parameters
''''''''''

Mechanism behavior can be tuned using parameters and ion channel dependencies,
prescribed in the NMODL description.
Parameters and ion species are set initially before a simulation starts, and remain
unchanged thereafter, for the duration of the simulation.
There are two types of parameters that can be set by users:

* *Global* parameters are a single scalar value that is the same everywhere a mechanism is defined.
* *Range* parameters can vary spatially.

Every mechanism is described with a *mechanism description*, a
``(name, range_parameters)`` tuple, where ``name`` is a string,
and ``range_parameters`` is an optional dictionary of key-value pairs
that specifies values for range parameters.
For example, consider a mechanism that models passive leaky dynamics with
the following parameters:

* *Name*: ``"passive"``.
* *Global parameter*: reversal potential ``el``, default -65 mV.
* *Range parameter*: conductance ``g``, default 0.001 S⋅cm⁻².

The following example mechanism descriptions for our passive mechanism show that parameters and
ion species dependencies only need to be specified when they differ from their defaults:

* ``("passive")``: the passive mechanism with default parameters.
* ``("passive/el=-80")``: derive a new passive mechanism with a non-default value for global parameter.
* ``("passive", {"gl": 0.005})``: passive mechanism with a new a non-default range parameter value.
* ``("passive/el=-80", {"gl": 0.005})``: derive a new passive mechanism that overrides both

Similarly to global parameters, ion species can be renamed in the mechanism name.
This allows the use of generic mechanisms that can be adapted to a specific species
during model instantiation.
For example, the ``nernst`` mechanism in Arbor's default mechanism catalogue calculates
the reversal potential of a generic ionic species ``x`` according to its internal
and external concentrations and valence. To specialize ``nersnt`` for calcium name it
``("nernst/x=ca")``, or if there is only one ions species in the mechanism the following
shorthand ``("nernst/ca")`` can be used unambiguously.

.. _mechanisms-name-note:

.. note::
    Global parameter values and ionic dependencies are the same for each instance of
    a mechanism, so when these are redifeind a new mechanism is created, derived from
    the parent mechanism.
    For this reason, new global parameter values and ion renaming are part of the name of
    the new mechanism, or a mechanism with a new unique name must be defined.


Mechanism Types
---------------

There are two broad categories of mechanism, density mechanisms and
point mechanisms, and a third special density mechanism for
computing ionic reversal potentials.

.. _mechanisms-density:

Density mechanisms
''''''''''''''''''''''

Density mechanisms are :ref:`NMODL mechanisms <nmodl>`
which describe biophysical processes that are distributed in space, but whose behaviour
is defined purely by the state of the cell and the process at any given point.

Density mechanisms are commonly used to describe ion channel dynamics,
for example the ``hh`` and ``pas`` mechanisms provided by NEURON and Arbor,
which model classic Hodgkin-Huxley and passive leaky currents respectively.

.. _mechanisms-revpot:

Ion reversal potential mechanisms
'''''''''''''''''''''''''''''''''

These mechanisms, which describe ionic reversal potential
behaviour, can be specified for cells or the whole model.

The reversal potential of an ion species is calculated by an
optional *reversal potential mechanism*.
If no such mechanism is specified for an ion species, the initial
reversal potential values are maintained for the course of a simulation.
Otherwise, the mechanism does the work.

Reversal potential mechanisms are density mechanisms subject to some strict restrictions.
Specifically, a reversal potential mechanism described in NMODL:

* May not maintain any state variables.
* Can only write to the reversal potential (``eX``) value of the ion species.
* Can not be a :ref:`point mechanism <mechanisms-point>`.

Essentially, reversal potential mechanisms must be pure functions of cellular
and ionic state.

.. note::
    Arbor imposes greater restrictions on mechanisms that update ionic reversal potentials
    than NEURON. Doing so simplifies reasoning about interactions between
    mechanisms that share ionic species, by virtue of having one mechanism, and one
    mechanism only, that calculates reversal potentials according to concentrations
    that the other mechanisms use and modify.

.. _mechanisms-point:

Point mechanisms
'''''''''''''''''''''''''''''''''

*Point mechanisms*, which are associated with connection end points on a
cable cell, are placed at discrete locations on the cell.
Unlike density mechanisms, whose behaviour is defined purely by the state of the cell and the process,
their behavior is additionally governed by the timing and weight of events delivered
via incoming connections.


API
---

* :ref:`Python <py_mechanisms>`
* :ref:`C++ <cpp_mechanisms>`
