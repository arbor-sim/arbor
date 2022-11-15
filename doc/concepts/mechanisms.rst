.. _mechanisms:

Cable cell mechanisms
=====================

Mechanisms describe biophysical processes such as ion channels, synapses and gap-junctions.
Mechanisms are assigned to regions and locations on a cell morphology
through the process of :ref:`decoration <cablecell-decoration>`.
Mechanisms are described using a dialect of the :ref:`NMODL <formatnmodl>` domain
specific language that is similarly used in `NEURON <https://neuron.yale.edu/neuron/>`_.

Arbor supports mechanism descriptions using the NMODL language through our ``modcc``
compiler. ``modcc`` supports many of NMODL's features but there are a few
additional :ref:`guidelines <formatnmodl>`.
for users who wish to compile their own mechanisms for Arbor. Out-of-tree mechanism
building is available in Arbor (See: :ref:`mechanisms_dynamic`). We also have built-in
mechanisms, which are organized in *mechanism catalogues*.

Mechanism catalogues
--------------------

A *mechanism catalogue* is a collection of mechanisms that maintains:

1. A collection of mechanism metadata indexed by name.
2. A further hierarchy of *derived* mechanisms, that allow specialization of
   global parameters, ion bindings, and implementations.
3. A map for looking up a concrete mechanism implementation on a target hardware back end.

Derived mechanisms will always have a different name to the mechanism from which they are derived.
This name is given explicitly when the derivation is constructed, or implicitly when a mechanism
is :ref:`requested <mechanisms-name-note>` with a name of the form ``"mech/param=value,..."``.
In this instance, if a mechanism of that name does not already exist in the catalogue, it will be
implicitly derived from an existing mechanism ``"mech"`` with global parameters and ion bindings
set according to the assignments following the slash. If the mechanism ``"mech"`` depends upon
only a single ion, the name of that ion can be omitted in the assignments:
``"mech/oldion=newion"`` and ``"mech/newion"`` are equivalent derivations.


Catalogues provide an interface for querying mechanism metadata, which includes the following information:

* Global parameter names, units, and default values.
* Range parameter names, units, and default values.
* State variable names, units and default values.
* Ion dependencies: for each ion used by the mechanism, information on whether the mechanism writes
  to its internal or external concentration or to its reversal potential value, and whether it reads
  or asserts the ionic charge.

Arbor provides a default catalogue of mechanisms as well as two other catalogues containing the sets of common mechanisms
used by the `Allen Institute <https://alleninstitute.org/>`_ and the `Blue Brain Project <https://portal.bluebrain.epfl.ch/>`_.
(Find the NMODL descriptions of the `default mechanisms <https://github.com/arbor-sim/arbor/tree/master/mechanisms/default>`_,
the `Allen institute mechanisms <https://github.com/arbor-sim/arbor/tree/master/mechanisms/allen>`_ and
the `BBP mechanisms <https://github.com/arbor-sim/arbor/tree/master/mechanisms/bbp>`_ at the provided links.)

.. _mechanisms_builtins:

Built-in Catalogues
'''''''''''''''''''

Arbor provides the ``default_catalogue`` with the following mechanisms:

* *pas*: Leaky current (:ref:`density mechanism <mechanisms-density>`).
* *hh*: Classic Hodgkin-Huxley dynamics (:ref:`density mechanism
  <mechanisms-density>`).
* *nernst*: Calculate reversal potential for an ionic species using the Nernst
  equation (:ref:`reversal potential mechanism <mechanisms-revpot>`). **NB**
  This is not meant to be used directly
* *expsyn*: Synapse with discontinuous change in conductance at an event
  followed by an exponential decay (:ref:`point mechanism <mechanisms-point>`).
* *exp2syn*: Bi-exponential conductance synapse described by two time constants:
  rise and decay (:ref:`point mechanism <mechanisms-point>`).
* *gj*: Linear gap-junction mechanism with constant conductance (:ref:`junction mechanism <mechanisms-junction>`).

With the exception of *nernst*, these mechanisms are the same as those available in NEURON.

Two catalogues are provided that collect mechanisms associated with specific projects and model databases:

* ``bbp_catalogue`` For models published by the Blue Brain Project (BBP).
* ``allen_catalogue`` For models published on the Allen Brain Atlas Database.

A fourth catalogue ``stochastic_catalogue`` provides mechanisms expressed as stochastic differential
equations:

* *ou_input* Synapse mechanism that can stochastically account for a population of *ou_input*
  synapses.  The mechanism is similar to *expsyn_curr* but with the exponential decay being subject
  to noise due to a Ornstein-Uhlenbeck process.


.. _mechanisms_dynamic:

Adding Catalogues to Arbor
''''''''''''''''''''''''''

.. Note::

   If you are coming from NEURON this is the equivalent of ``nrnivmodl``.

This will produce a catalogue loadable at runtime by calling ``load_catalogue``
with a filename in both C++ and Python. The steps are

1. Prepare a directory containing your NMODL files (.mod suffixes required)
2. Call ``arbor-build-catalogue`` installed by arbor

   .. code-block :: bash

     arbor-build-catalogue <name> <path/to/nmodl>

All files with the suffix ``.mod`` located in ``<path/to/nmodl>`` will be baked
into a catalogue named ``lib<name>-catalogue.so`` and placed into your current
working directory. Note that these files are platform-specific and should only
be used on the combination of OS, compiler, arbor, and machine they were built
with. See our internal documentation for more advanced usage of the builder.
Errors might be diagnosable by passing the ``-v`` flag.

This catalogue can then be used similarly to the built-in ones

   .. code-block :: python

     import arbor as A

     c = A.load_catalogue('bbp2-catalogue.so')

     [n for n in c]
     >> ['Ca_LVAst',
         'Nap_Et2',
         'NaTa_t',
         'SKv3_1',
         'K_Tst',
         'Ih',
         'SK_E2',
         'Ca_HVA',
         'CaDynamics_E2',
         'Im',
         'NaTs2_t',
         'K_Pst']

See also the demonstration in ``python/example/dynamic-catalogue.py`` for an example.

Parameters
''''''''''

Mechanism behaviour can be tuned using parameters and ion channel dependencies,
as defined in the NMODL description.
Parameters and ion species are set initially before a simulation starts, and remain
unchanged thereafter, for the duration of the simulation.
There are two types of parameters that can be set by users:

* *Global* parameters are a single scalar value that is the same everywhere a mechanism is defined.
* *Range* parameters can vary spatially.

Every mechanism is applied to a cell via a *mechanism description*, a
``(name, range_parameters)`` tuple, where ``name`` is a string,
and ``range_parameters`` is an optional dictionary of key-value pairs
that specifies values for range parameters.
For example, consider a mechanism that models passive leaky dynamics with
the following parameters:

* *Name*: ``"pas"``.
* *Global parameter*: reversal potential ``e``, default -70 mV.
* *Range parameter*: conductance ``g``, default 0.001 S⋅cm⁻².

The following example mechanism descriptions for our passive mechanism show that parameters and
ion species dependencies only need to be specified when they differ from their defaults:

* ``("pas")``: the passive mechanism with default parameters.
* ``("pas/e=-80")``: derive a new passive mechanism with a non-default value for global parameter.
* ``("pas", {"g": 0.005})``: passive mechanism with a new a non-default range parameter value.
* ``("pas/e=-80", {"g": 0.005})``: derive a new passive mechanism that overrides both

Similarly to global parameters, ion species can be renamed in the mechanism name.
This allows the use of generic mechanisms that can be adapted to a specific species
during model instantiation.
For example, the ``nernst`` mechanism in Arbor's default mechanism catalogue calculates
the reversal potential of a generic ionic species ``x`` according to its internal
and external concentrations and valence. To specialize ``nernst`` for calcium name it
``("nernst/x=ca")``, or as there is only one ion species in the mechanism the
shorthand ``("nernst/ca")`` can be used unambiguously.

.. _mechanisms-name-note:

.. note::
    Global parameter values and ionic dependencies are the same for each instance of
    a mechanism; changing these requires the derivation of a new mechanism, implicitly or explicitly.
    For this reason, new global parameter values and ion renaming are part of the name of
    the new mechanism, or a mechanism with a new unique name must be defined.


Mechanism types
---------------

There are three broad categories of mechanism: density mechanisms, point mechanisms,
gap-junction mechanisms and a fourth special density mechanism for computing ionic
reversal potential.

.. _mechanisms-density:

Density mechanisms
''''''''''''''''''''''

Density mechanisms describe biophysical processes that are distributed in space,
but whose behaviour is defined purely by the state of the cell and the process
at any given point.

Density mechanisms are commonly used to describe ion channel dynamics,
for example the ``hh`` and ``pas`` mechanisms provided by NEURON and Arbor,
which model classic Hodgkin-Huxley and passive leaky currents respectively.

In NMODL, density mechanisms are identified using the ``SUFFIX`` keyword in the
``NEURON`` block.

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
''''''''''''''''

*Point mechanisms*, which are associated with connection end points on a
cable cell, are placed at discrete locations on the cell.
Unlike density mechanisms, whose behaviour is defined purely by the state of the cell
and the process, their behaviour is additionally governed by the timing and weight of
events delivered via incoming connections.

In NMODL, point mechanisms are identified using the ``POINT_PROCESS`` keyword in the
``NEURON`` block.

.. _mechanisms-junction:

Junction mechanisms
'''''''''''''''''''

*Junction mechanisms*, which are associated with gap-junction connection end points on a
cable cell, are placed at discrete locations on the cell.
A junction mechanism contributes a current at the discrete location of the cell on which it is placed.
This current contribution depends on the state of the mechanism and the process, as well as the membrane
potential at the discrete location which forms the other end of the gap-junction connection and the weight
of that connection.

In NMODL, junction mechanisms are identified using the ``JUNCTION_PROCESS`` keyword in the
``NEURON`` block.

.. note::
    ``JUNCTION_PROCESS`` is an Arbor-specific extension to NMODL. The NMODL description of gap-junction
    mechanisms in arbor is not identical to NEURON's though it is similar.

.. _mechanisms-sde:

Stochastic Processes
''''''''''''''''''''

Arbor offers support for stochastic processes at the level of
:ref:`point mechanisms <mechanisms-point>` and :ref:`density mechanisms <mechanisms-density>`.
These processes can be modelled as systems of stochastic differential equations (SDEs). In general,
such equations have the differential form:

.. math::

    d\textbf{X}(t) = \textbf{f}(t, \textbf{X}(t)) dt + \sum_{i=0}^{M-1} \textbf{l}_i(t,\textbf{X}(t)) d B_i(t),

where :math:`\textbf{X}` is the vector of state variables, while the vector valued function
:math:`\textbf{f}` represents the deterministic differential. The *M* functions :math:`\textbf{l}_i`
are each associated with the Brownian Motion :math:`B_i` (Wiener process). The Brownian motions are
assumed to be standard: 

.. math::

    \begin{align*}
    B_i(0) &= 0 \\
    E[B_i(t)] &= 0 \\
    E[B_i^2(t)] &= t
    \end{align*}

The above differential form is an informal way of expressing the corresponding integral equation,

.. math::

    \textbf{X}(t+s) = \textbf{X}(t) + \int_t^{t+s} \textbf{f}(\tau, \textbf{X}(\tau)) d\tau + \sum_{i=0}^{M-1} \int_t^{t+s} \textbf{l}_i(\tau,\textbf{X}(\tau)) d B_i(\tau).


By defining a random process called **stationary white noise** as the formal derivative
:math:`W_i(t) = \dfrac{d B_i(t)}{dt}`, we can write the system of equations using a shorthand
notation as

.. math::

    \textbf{X}^\prime(t) = \textbf{f}(t, \textbf{X}(t)) + \sum_{i=0}^{M-1} \textbf{l}_i(t,\textbf{X}(t)) W_i(t)

Since we used standard Brownian Motions above, the withe noises :math:`W_i(t)` are Gaussian for all
*t* with :math:`\mu=0`, :math:`\sigma^2=1`.

In Arbor, the white noises :math:`W_i` are assumed to be independent of each other. Furthermore,
each connection end point (point mechanism) or control volume (density mechanism) are assumed to
generate independent noise, as well. The system of stochastic equations is interpreted in the `Itô
sense <https://en.wikipedia.org/wiki/It%C3%B4_calculus>`_ and numerically solved using the
Euler-Maruyama method.
For specifics about the notation to define stochastic processes, please
consult the :ref:`Arbor-specific NMODL extension <format-sde>`.

.. note::

   While the units of :math:`\textbf{f}(t, \textbf{X}(t))` represent the deterministic rate of
   change (per millisecond),

   .. math::

        \left[\textbf{f}(t, \textbf{X}(t))\right] = \frac{\left[\textbf{X}(t)\right]}{ms},

   the stochastic terms scale with :math:`t^{-1/2}`,

   .. math::

        \left[\textbf{l}_i(t, \textbf{X}(t))\right] = \frac{\left[\textbf{X}(t)\right]}{\sqrt{ms}}.


**Example:** The popular Ornstein-Uhlenbeck process is described by a scalar linear mean-reverting SDE
and can be written as

.. math::

    X^\prime = -\frac{1}{\tau} (X - \mu) + \sqrt{\frac{2}{τ}}  \sigma W,

with white noise :math:`W`, and constant model parameters :math:`\tau`, :math:`\mu` and
:math:`\sigma`. The relaxation time :math:`\tau` determines how fast the process reverts back to its
mean value :math:`\mu`, and :math:`\sigma` controls the volatility (:math:`\mu` and :math:`\sigma`
have the same units as :math:`X`). The expected value and variance can be computed analytically and
yield

.. math::

    \begin{align*}
    E[X]   &= \mu - \left( \mu - X_0\right) e^{-t/\tau}, \\
    Var[X] &= \sigma^2 \left( 1 - e^{-2 t/\tau} \right),
    \end{align*}

which in the limit :math:`t \rightarrow \infty` converge to

.. math::

    \begin{align*}
    E[X]   &= \mu, \\
    Var[X] &= \sigma^2.
    \end{align*}

API
---

* :ref:`Python <py_mechanisms>`
