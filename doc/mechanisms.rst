.. _mechanisms:

Mechanisms
===========

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
be created implitlitly.
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

Python API
----------

Mechanism Catalogues
''''''''''''''''''''

.. py:class:: catalogue

    A *mechanism catalogue* is a collection of mechanisms that maintains:

    1. Collection of mechanism metadata indexed by name.
    2. A further hierarchy of *derived* mechanisms, that allow specialization of
       global parameters, ion bindings, and implementations.

    .. py:method:: has(name)

        Test if mechanism with *name* is in the catalogue.

        :param name: name of mechanism.
        :type name: str
        :return: bool

    .. py:method:: is_derived(name)

        Is *name* a derived mechanism or can it be implicitly derived?

        :param name: name of mechanism.
        :type name: str
        :return: bool

    .. py:method:: __getitem__(name)

        Look up mechanism meta data with *name*.

        .. code-block:: Python

            import arbor

            cat = arbor.default_catalogue()

            # Print default value and units for gnabar parameter of hh.
            print(cat['hh'].parameters['gnabar'])

        :param name: name of mechanism.
        :type name: str
        :return: mechanism metadata
        :rtype: :class:`mechanism_info`

    .. py:method:: derive(name, parent, globals={}, ions={})

        Derive a new mechanism with *name* from the mechanism *parent*.

        If no parameters or ion renaming are specified with *globals* or *ions*,
        the method will attempt to implicitly derive a new mechanism from parent by parsing global and
        ions from the parent string.

        .. code-block:: Python

            import arbor

            cat = arbor.default_catalogue()

            # Use the value of the Faraday constant as published by CODATA in 1986,
            # and bind to pottasium ion species.
            cat.derive('krev',  'nernst', globals={'F': 96485.309}, ions={'x': 'k'})

            # Derive a reversal potential mechanism for sodium from the one we defined
            # for potasium, which will inherit the redefined Faraday constant.
            cat.derive('narev', 'krev', ions={'k': 'na'})

            # Alternatively, we can derive a mechanism with global parameters and ion renaming
            # specified in the parent name string.
            cat.derive('krev_imp', 'nernst/F=96485.309,k')
            cat.derive('carev', 'krev_imp/ca')

        :param name: name of new derived mechanism.
        :type name: str
        :param parent: name of parent mechanism.
        :type parent: str
        :param globals: a dictionary mapping global parameter names to their values, if any.
        :type globals: dict[str, float]
        :param ions: a dictionary renaming ion species, if any.
        :type ions: dict[str, str]

.. py:class:: mechanism_info

    Meta data about the fields and ion dependencies of a mechanism.
    The data is presented as read-only attributes.

    .. code-block:: Python

        import arbor
        cat = arbor.default_catalogue()

        # Get mechanism_info for the 'expsyn' mechanism.
        mech = cat['expsyn']

        # Query the mechanism_info for information about parameters.

        print(mech.parameters.keys())
        # dict_keys(['e', 'tau'])

        print(mech.parameters['tau'].units)
        # 'ms'

        print(mech.parameters['tau'].default)
        # 2.0

    .. py:attribute:: globals
        :type: dict[str, mechanism_field]

        Global fields have one value common to an instance of a mechanism, are constant in time and set at instantiation.

    .. py:attribute:: parameters
        :type: dict[str, mechanism_field]

        Parameter fields may vary across the extent of a mechanism, but are constant in time and set at instantiation.

    .. py:attribute:: state
        :type: dict[str, mechanism_field]

        State fields vary in time and across the extent of a mechanism, and potentially can be sampled at run-time.

    .. py:attribute:: ions
        :type: dict[str, ion_dependency]

        Ion dependencies.

    .. py:attribute:: linear
        :type: bool

        True if a synapse mechanism has linear current contributions so that multiple instances on the same compartment can be coalesced.


.. py:class:: ion_dependency

    Meta data about a mechanism's dependence on an ion species,
    presented as read-only attributes.

    .. code-block:: Python

        import arbor
        cat = arbor.default_catalogue()

        # Get ion_dependency for the 'hh' mechanism.
        ions = cat['hh'].ions

        # Query the ion_dependency.

        print(ions.keys())
        # dict_keys(['k', 'na'])

        print(ions['k'].write_rev_pot)
        # False

        print(ions['k'].read_rev_pot)
        # True

    .. py:attribute:: write_int_con
        :type: bool

        If the mechanism contributes to the internal concentration of the ion species.

    .. py:attribute:: write_ext_con
        :type: bool

        If the mechanism contributes to the external concentration of the ion species.

    .. py:attribute:: write_rev_pot
        :type: bool

        If the mechanism calculates the reversal potential of the ion species.

    .. py:attribute:: read_rev_pot
        :type: bool

        If the mechanism depends on the reversal potential of the ion species.


.. py:class:: mechanism_field

    Meta data about a specific field of a mechanism, presented as read-only attributes.

    .. py:attribute:: units
        :type: string

        The units of the field.

    .. py:attribute:: default
        :type: float

        The default value of the field.

    .. py:attribute:: min
        :type: float

        The minimum permissible value of the field.

    .. py:attribute:: max
        :type: float

        The maximum permissible value of the field.

The :py:class:`mechanism_info` type above presents read-only information about a mechanism that is available in a catalogue.

When :ref:`decorating <cablecell-decoration>` a cable cell, we use a :py:class:`mechanism` type to describe a
mechanism that is to be painted or placed on the cable cell.

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

    The method for setting a parameter depends on its type.
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

    .. method:: mechanism(name, params)

        constructor for mechanism with *name* and range parameter overrides *params*,
        for example: ``arbor.mechanism(name='pas', params={'g': 0.01})``.

        :param name: name of mechanism.
        :type name: str
        :param params: A dictionary of parameter values, with parameter name as key.
        :type params: dict[str, double]

    .. method:: mechanism(name)
        :noindex:

        constructor for mechanism.
        The *name* can be either the name of a mechanism in the catalogue,
        e.g.  ``arbor.mechanism('pas')``, or an implicitly derived mechanism,
        e.g. ``arbor.mechanism('nernst/k')``.

    .. method:: set(name, value)

        Set new value for a parameter.

        :param name: name of the parameter.
        :type name: str
        :param value: value of the parameter.
        :type value: float

    .. py:attribute:: name
        :type: str

        The name of the mechanism.

    .. py:attribute:: values
        :type: dict

        A dictionary of key-value pairs for the parameters.

    .. code-block:: Python

        import arbor

        # Create pas mechanism with default parameter values (set in NOMDL file).
        m1 = arbor.mechanism('passive')

        # Create default mechainsm with custom conductance (range).
        m2 = arbor.mechanism('passive', {'g', 0.1})

        # Create a new pas mechanism with that changes reversal potential (global).
        m3 = arbor.mechanism('passive/el=-45')

        # Create an instance of the same mechanism, that also sets conductance (range).
        m4 = arbor.mechanism('passive/el=-45', {'g', 0.1})

        # This is an equivalent to m4, using set method to specify range parameters.
        m5 = arbor.mechanism('passive/el=-45')
        m5.set('g', 0.1)

        # Decorate the 'soma' on a cable_cell.

        cell.paint('"soma"', m1)
        cell.paint('"soma"', m2) # Error: can't place the same mechanism on overlapping regions
        cell.paint('"soma"', m3) # This would be ok: m3 is a new, derived mechanism by virtue of
                                 # having a different name, i.e. 'passive/el=-45' vs. 'passive'.

