.. _pycable_cell:

Cable cells
===========

.. currentmodule:: arbor

.. py:class:: decor

    A decor object contains a description of the cell dynamics, to be applied
    to a morphology when used to instantiate a :py:class:`cable_cell`

    .. method:: __init__()

        Construct an empty decor description.

    Properties for which defaults can be defined over the entire cell, specifically
    :ref:`cable properties <cable-properties>` and :ref:`ion properties <cable-ions>`,
    are set with ``set_property`` and ``set_ion`` methods.

    .. method:: set_property(Vm=None, cm=None, rL=None, tempK=None)

        Set default values of cable properties on the whole cell.
        Overrides the default global values, and can be overridden by painting
        the values onto regions.

        :param str region: description of the region.
        :param Vm: Initial membrane voltage [mV].
        :type Vm: float or None
        :param cm: Membrane capacitance [F/m²].
        :type cm: float or None
        :param rL: Axial resistivity of cable [Ω·cm].
        :type rL: float or None
        :param tempK: Temperature [Kelvin].
        :type tempK: float or None

        .. code-block:: Python

            # Set cell-wide values for properties for resistivity and capacitance
            decor.set_property(rL=100, cm=0.1)

    .. method:: set_ion(ion, int_con=None, ext_con=None, rev_pot=None, method=None)

        Set default value for one or more properties of a specific ion on the whole cell.
        Set the properties of ion species named ``ion`` that will be applied
        by default everywhere on the cell. Species concentrations and reversal
        potential can be overridden on specific regions using the paint interface,
        while the method for calculating reversal potential is global for all
        compartments in the cell, and can't be overriden locally.

        :param str ion: description of the ion species.
        :param float int_con: initial internal concentration [mM].
        :type int_con: float or None.
        :param float ext_con: initial external concentration [mM].
        :type ext_con: float or None.
        :param float rev_pot: reversal potential [mV].
        :type rev_pot: float or None
        :param method: method for calculating reversal potential.
        :type method: :py:class:`mechanism` or None

        .. code-block:: Python

            # Set nernst reversal potential method for calcium.
            decor.set_ion('ca', method=mech('nernst/x=ca'))

            # Set reversal potential and concentration for sodium.
            # The reversal potential is fixed, so we set the method to None.
            decor.set_ion('na', int_con=5.0, rev_pot=70, method=None)

    Verious specialisations of the ``paint`` method are available for setting properties
    and mechanisms that are applied to regions.

    .. method:: paint(region, Vm=None, cm=None, rL=None, tempK=None)

        Set cable properties on a region.

        :param str region: description of the region.
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
            decor.paint('"soma"', rL=100)
            # Specialize resistivity and capacitance on the axon, where
            # axon is defined using a region expression.
            decor.paint('(tag 2)', cm=0.05, rL=80)

    .. method:: paint(region, name, int_con=None, ext_con=None, rev_pot=None)

        Set ion species properties initial conditions on a region.

        :param str name: name of the ion species.
        :param float int_con: initial internal concentration [mM].
        :type int_con: float or None.
        :param float ext_con: initial external concentration [mM].
        :type ext_con: float or None.
        :param float rev_pot: reversal potential [mV].
        :type rev_pot: float or None

    .. method:: paint(region, mechanism)

        Apply a mechanism with a region.

        :param str region: description of the region.
        :param mechanism: the mechanism.
        :type mechanism: :py:class:`mechanism`

    .. method:: paint(region, mech_name)

        Apply a mechanism with a region using the name of the mechanism.
        The mechanism will use the parameter values set in the mechanism catalogue.

        :param str region: description of the region.
        :param str mechanism: the name of the mechanism.

    .. method:: place(locations, const arb::mechanism_desc& d)

        Place one instance of synapse described by ``mechanism`` to each location in ``locations``.

        :param str locations: description of the locset.
        :param str mechanism: the name of the mechanism.

    .. method:: place(locations, mechanism)

        Place one instance of synapse described by ``mechanism`` to each location in ``locations``.

        :param str region: description of the region.
        :param mechanism: the mechanism.
        :type mechanism: :py:class:`mechanism`

    .. method:: place(locations, const arb::gap_junction_site& site)

        Place one gap junction site at each location in ``locations``.

        "locations"_a, "gapjunction"_a,

    .. method:: place(locations, const arb::i_clamp& stim)

        Add a current stimulus at each location in ``locations``.

        "locations"_a, "iclamp"_a,

    .. method:: place(locations, const arb::threshold_detector& d)

        Add a voltage spike detector at each location in locations.

        "locations"_a, "detector"_a,

    .. method:: discretization(policy)

        Set the cv_policy used to discretise the cell into compartments for simulation.

        :param policy: The cv_policy.
        :type policy: :py:class:`cv_policy`

.. py:class:: cable_cell

    A cable cell is constructed from a :ref:`morphology <morph-morphology>`
    and an optional :ref:`label dictionary <labels-dictionary>`.

    .. note::
        The regions and locsets defined in the label dictionary are
        :ref:`concretised <labels-concretise>` when the cable cell is constructed,
        and an exception will be thrown if an invalid label expression is found.

        There are two reasons an expression might be invalid:

        1. Explicit reference to a location of cable that does not exist in the
           morphology, for example ``(branch 12)`` on a cell with 6 branches.
        2. Reference to an incorrect label: circular reference, or a label that does not exist.


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

.. py:class:: ion

    properties of an ionic species.


