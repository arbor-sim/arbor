.. _py_mechanisms:

Cable cell mechanisms
=====================

.. py:class:: mechanism

    Mechanisms describe physical processes, distributed over the membrane of the cell.
    *Density mechanisms* are associated with regions of the cell, whose dynamics are
    a function of the cell state and their own state where they are present.
    *Point mechanisms* are defined at discrete locations on the cell, which receive
    events from the network.
    *Junction mechanisms* are defined at discrete locations on the cell, which define the
    behavior of a gap-junction mechanism.
    A fourth, specific type of density mechanism, which describes ionic reversal potential
    behaviour, can be specified for cells or the whole model.

    The :class:`mechanism` type is a simple wrapper around a mechanism
    :attr:`mechanism.name` and a dictionary of named parameters.

    Mechanisms have two types of parameters:

    * global parameters: a scalar value that is the same for all instances
      of a mechanism.
    * range parameters: the value of range parameters is defined for each instance
      of the mechanism on a cell. For density mechanisms, this means one value for
      each :term:`control volume` on which it is present.

    The method for setting a parameter depends on its type.
    If global parameters change, we are effectively defining a new type
    of mechanism, so global parameter information is encoded in the
    name.
    Range parameters are set using a dictionary of name-value pairs.

    .. code-block:: Python

        import arbor

        # A passive leaky channel with default parameter values (set in NOMDL file).
        pas_0 = arbor.mechanism('pas')

        # A passive leaky channel with custom conductance (range).
        pas_1 = arbor.mechanism('pas', {'g': 0.02})

        # A passive leaky channel with custom reversal potential (global).
        pas_2 = arbor.mechanism('pas/e=-45')

        # A passive leaky channel with custom reversal potential (global), and custom conductance (range).
        pas_3 = arbor.mechanism('pas/e=-45', {'g', 0.1})

        # This is an equivalent to pas_3, using set method to specify range parameters.
        pas_4 = arbor.mechanism('pas/e=-45')
        pas_4.set('g', 0.1)

        # Reversal potential using Nernst equation with GLOBAL parameter values
        # for Faraday's constant and the target ion species, set with a '/' followed
        # by comma-separated list of parameter after the base mechanism name.
        rev = arbor.mechanism('nernst/F=96485,x=ca')

        # An exponential synapse with default parameter values (set in NOMDL file).
        expsyn = arbor.mechanism("expsyn")

        # A gap-junction mechanism with default parameter values (set in NOMDL file).
        gj = arbor.mechanism("gj")

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

.. py:class:: density

   When :ref:`decorating <cablecell-decoration>` a cable cell, we use a :py:class:`density` type to
   wrap a density :py:class:`mechanism` that is to be painted on the cable cell.

   Different :py:class:`density` mechanisms can be painted on top of each other.

    .. code-block:: Python

        import arbor

        pas = arbor.mechanism('pas')
        pas.set('g', 0.2)

        # Decorate the 'soma' with (multiple) density mechanisms
        decor.paint('"soma"', density(pas))
        decor.paint('"soma"', density('pas', {'g': 0.1})) # Error: can't place the same mechanism on overlapping regions
        decor.paint('"soma"', density('pas/e=-45'))       # This is ok: pas/e=-45 is a new, derived mechanism by virtue of
                                                          # having a different name, i.e. 'pas/e=-45' vs. 'pas'.

    .. py:attribute:: mech
        :type: mechanism

        The underlying mechanism.

    .. method:: density(name)

        constructs :attr:`mech` with *name* and default parameters.

        :param name: name of mechanism.
        :type name: str

    .. method:: density(name, params)
        :noindex:

        constructs :attr:`mech` with *name* and range parameter overrides *params*.
        for example: ``arbor.density('pas', {'g': 0.01})``.

        :param name: name of mechanism.
        :type name: str
        :param params: A dictionary of parameter values, with parameter name as key.
        :type params: dict[str, double]

    .. method:: density(mech)
        :noindex:

        constructs :attr:`mech` from *mech*.

        :param mech: mechanism description.
        :type mech: :py:class:`mechanism`

    .. method:: density(mech, params)
        :noindex:

        constructs :attr:`mech` from *mech* and sets the range parameter overrides *params*.

        :param mech: mechanism description.
        :type mech: :py:class:`mechanism`
        :param params: A dictionary of parameter values, with parameter name as key.
        :type params: dict[str, double]

.. py:class:: synapse

   When :ref:`decorating <cablecell-decoration>` a cable cell, we use a :py:class:`synapse` type to
   wrap a point :py:class:`mechanism` that is to be placed on the cable cell.

    .. py:attribute:: mech
        :type: mechanism

        The underlying mechanism.

    .. method:: synapse(name)

        constructs :attr:`mech` with *name* and default parameters.

        :param name: name of mechanism.
        :type name: str

    .. method:: synapse(name, params)
        :noindex:

        constructs :attr:`mech` with *name* and range parameter overrides *params*.
        for example: ``arbor.synapse('expsyn', {'tau': 0.01})``.

        :param name: name of mechanism.
        :type name: str
        :param params: A dictionary of parameter values, with parameter name as key.
        :type params: dict[str, double]

    .. method:: synapse(mech)
        :noindex:

        constructs :attr:`mech` from *mech*.

        :param mech: mechanism description.
        :type mech: :py:class:`mechanism`

    .. method:: synapse(mech, params)
        :noindex:

        constructs :attr:`mech` from *mech* and sets the range parameter overrides *params*.

        :param mech: mechanism description.
        :type mech: :py:class:`mechanism`
        :param params: A dictionary of parameter values, with parameter name as key.
        :type params: dict[str, double]


.. py:class:: junction

   When :ref:`decorating <cablecell-decoration>` a cable cell, we use a :py:class:`junction` type to
   wrap a gap-junction :py:class:`mechanism` that is to be placed on the cable cell.

    .. py:attribute:: mech
        :type: mechanism

        The underlying mechanism.

    .. method:: junction(name)

        constructs :attr:`mech` with *name* and default parameters.

        :param name: name of mechanism.
        :type name: str

    .. method:: junction(name, params)
        :noindex:

        constructs :attr:`mech` with *name* and range parameter overrides *params*.
        for example: ``arbor.junction('gj', {'g': 2})``.

        :param name: name of mechanism.
        :type name: str
        :param params: A dictionary of parameter values, with parameter name as key.
        :type params: dict[str, double]

    .. method:: junction(mech)
        :noindex:

        constructs :attr:`mech` from *mech*.

        :param mech: mechanism description.
        :type mech: :py:class:`mechanism`

    .. method:: junction(mech, params)
        :noindex:

        constructs :attr:`mech` from *mech* and sets the range parameter overrides *params*.

        :param mech: mechanism description.
        :type mech: :py:class:`mechanism`
        :param params: A dictionary of parameter values, with parameter name as key.
        :type params: dict[str, double]

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

    .. py:attribute:: kind
        :type: string

        String representation of the kind of the mechanism: density, point or reversal potential.

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

        True if a synapse mechanism has linear current contributions so that multiple instances on the same :term:`control volume` can be coalesced.

    .. py:attribute:: post_events
        :type: bool

        True if a synapse mechanism has a `POST_EVENT` procedure defined.


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


Mechanism catalogues
''''''''''''''''''''

.. py:class:: catalogue

    A *mechanism catalogue* is a collection of mechanisms that maintains:

    1. Collection of mechanism metadata indexed by name.
    2. A further hierarchy of *derived* mechanisms, that allow specialization of
       global parameters, ion bindings, and implementations.

    .. py:method:: __init__(catalogue=None)

        Create an empty or copied catalogue.

        :param catalogue: ``catalogue`` to copy
        :type catalogue: :class:`catalogue`
        :return: empty or copied catalogue
        :rtype: :class:`catalogue`

    .. py:method:: __contains__(name)

        Test if mechanism with *name* is in the catalogue.

        Note: This enables the following idiom

        .. code-block:: Python

            import arbor

            if 'hh' in arbor.default_catalogue():
              print("Found HH mechanism.")

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

    .. py:method:: __iter___()

        Return a list names of all the mechanisms in the catalogue.

        Note: This enables the following idiom

        .. code-block:: Python

            import arbor

            for name in arbor.default_catalogue():
              print(name)

        :return: :class:`py_mech_cat_iterator`


    .. py:method:: extend(other, prefix)

        Import another catalogue, possibly with a prefix. Will raise an exception
        in case of name collisions.

        .. code-block:: Python

            import arbor

            cat = arbor.default_catalogue()
            cat.extend(arbor.allen_catalogue(), "")

        :param other: reference to other catalogue.
        :type other: :class:`mechanism_catalogue`
        :param prefix: prefix for mechanism names in ``other``
        :type prefix: str

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
