.. _cpp_mechanisms:

Cable cell mechanisms
=====================

Mechanisms describe physical processes distributed over the membrane of the cell.
*Density mechanisms* are associated with regions of the cell, whose dynamics are
a function of the cell state and their own state where they are present.
*Point mechanisms* are defined at discrete locations on the cell, which receive
events from the network.
*Junction mechanisms* are defined at discrete locations on the cell, which define the
behavior of a gap-junction mechanism.
A fourth, specific type of density mechanism, which describes ionic reversal potential
behaviour, can be specified for cells or the whole model.

.. cpp:namespace:: arb

.. cpp:class:: density

   When :ref:`decorating <cablecell-decoration>` a cable cell, we use a :cpp:class:`density` type to
   wrap a density :cpp:class:`mechanism` that is to be painted on the cable cell.

   Different :cpp:class:`density` mechanisms can be painted on top of each other.


.. cpp:class:: synapse

   When :ref:`decorating <cablecell-decoration>` a cable cell, we use a :cpp:class:`synapse` type to
   wrap a point :cpp:class:`mechanism` that is to be placed on the cable cell.

.. cpp:class:: junction

   When :ref:`decorating <cablecell-decoration>` a cable cell, we use a :cpp:class:`junction` type to
   wrap a gap-junction :cpp:class:`mechanism` that is to be placed on the cable cell.

.. cpp:class:: mechanism_info

    Meta data about the fields and ion dependencies of a mechanism.
    The data is presented as read-only attributes.

    .. cpp:member:: arb_mechanism_kind kind

        Mechanism kind

    .. cpp:member:: std::unordered_map<std::string, mechanism_field_spec> globals

        Global fields have one value common to an instance of a mechanism, are constant in time and set at instantiation.

    .. cpp:member:: std::unordered_map<std::string, mechanism_field_spec> parameters

        Parameter fields may vary across the extent of a mechanism, but are constant in time and set at instantiation.

    .. cpp:member:: std::unordered_map<std::string, mechanism_field_spec> state

        State fields vary in time and across the extent of a mechanism, and potentially can be sampled at run-time.

    .. cpp:member:: std::unordered_map<std::string, ion_dependency> ions

        Ion dependencies.

    .. cpp:member:: std::unordered_map<std::string, arb_size_type> random_variables

        Random variables

    .. cpp:member:: bool linear = false

    .. cpp:member:: bool post_events = false



.. cpp:class:: ion_dependency

    Metadata about a mechanism's dependence on an ion species,
    presented as read-only attributes.

    .. code-block:: Python

        import arbor as A
        cat = A.default_catalogue()

        # Get ion_dependency for the 'hh' mechanism.
        ions = cat['hh'].ions

        # Query the ion_dependency.

        print(ions.keys())
        # dict_keys(['k', 'na'])

        print(ions['k'].write_rev_pot)
        # False

        print(ions['k'].read_rev_pot)
        # True

    .. cpp:attribute:: write_int_con
        :type: bool

        If the mechanism contributes to the internal concentration of the ion species.

    .. cpp:attribute:: write_ext_con
        :type: bool

        If the mechanism contributes to the external concentration of the ion species.

    .. cpp:attribute:: write_rev_pot
        :type: bool

        If the mechanism calculates the reversal potential of the ion species.

    .. cpp:attribute:: read_rev_pot
        :type: bool

        If the mechanism depends on the reversal potential of the ion species.


.. cpp:class:: mechanism_field

    Metadata about a specific field of a mechanism is presented as read-only attributes.

    .. cpp:attribute:: units
        :type: string

        The units of the field.

    .. cpp:attribute:: default
        :type: float

        The default value of the field.

    .. cpp:attribute:: min
        :type: float

        The minimum permissible value of the field.

    .. cpp:attribute:: max
        :type: float

        The maximum permissible value of the field.

The :cpp:class:`mechanism_info` type above presents read-only information about a mechanism that is available in a catalogue.


Mechanism catalogues
--------------------

.. cpp:namespace:: arb

.. cpp:class:: catalogue

    A *mechanism catalogue* is a collection of mechanisms that maintains:

    1. Collection of mechanism metadata indexed by name.
    2. A further hierarchy of *derived* mechanisms, that allow specialization of
       global parameters, ion bindings, and implementations.

    .. cpp:function:: mechanism_catalogue(mechanism_catalogue&& other)

        Create an empty, copied or moved catalogue.

    .. cpp:method:: bool has(const std::string& name)

        Test if mechanism with *name* is in the catalogue.

    .. cpp:method:: is_derived(name)

        Is *name* a derived mechanism or can it be implicitly derived?

    .. cpp:method:: mechanism_info operator[](const std::string& name)

        Look up mechanism metadata with *name*.

    .. cpp:method:: void add(const std::string& name, mechanism_info)

         Add mechanism metadata with *name*.


    .. cpp:method:: std::vector<std::string> mechanism_names() const

        Return a list of names of all the mechanisms in the catalogue.

   .. cpp:method:: extend(other, prefix="")

        Import another catalogue, possibly with a prefix. Will raise an exception
        in case of name collisions.

    .. cpp:method:: void derive(const std::string& name, const std::string& parent, const std::vector<std::pair<std::string, double>>& global_params, const std::vector<std::pair<std::string, std::string>>& ion_remap = {});

        Derive a new mechanism with *name* from the mechanism *parent*.

        If no parameters or ion renaming are specified with *globals* or *ions*,
        the method will attempt to implicitly derive a new mechanism from the
        parent by parsing global and ions from the parent string.

.. cpp:function:: const mechanism_catalogue& global_default_catalogue()

    Return the default catalogue.

.. cpp:function:: const mechanism_catalogue& global_allen_catalogue()

    Return the Allen Institute catalogue.
    
.. cpp:function:: const mechanism_catalogue& global_bbp_catalogue()

    Return the Blue Brain Project catalogue.

.. cpp:function:: const mechanism_catalogue& global_stochastic_catalogue()

    Return a catalogue with stochastic mechanisms.

.. cpp:function:: const mechanism_catalogue load_catalogue(const std::filesystem::path&)

    Load catalogue from disk.
