.. _pycablecell-decor:

Cable cell decoration
=====================

.. currentmodule:: arbor

.. py:class:: decor

    A decor object contains a description of the cell dynamics, to be applied
    to a morphology when used to instantiate a :py:class:`cable_cell`

    .. method:: __init__()

        Construct an empty decor description.

    Properties for which defaults can be defined over the entire cell, specifically
    :ref:`cable properties <cablecell-properties>` and :ref:`ion properties <cablecell-ions>`,
    are set with ``set_property`` and ``set_ion`` methods.

    .. method:: set_property(Vm=None, cm=None, rL=None, tempK=None)

        Set default values of cable properties on the whole cell.
        Overrides the default global values, and can be overridden by painting
        the values onto regions.

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
        CVs in the cell, and can't be overridden locally.

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

    Various specialisations of the ``paint`` method are available for setting properties
    and density mechanisms that are applied to regions.

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
        :noindex:

        Set ion species properties initial conditions on a region.

        :param str name: name of the ion species.
        :param float int_con: initial internal concentration [mM].
        :type int_con: float or None.
        :param float ext_con: initial external concentration [mM].
        :type ext_con: float or None.
        :param float rev_pot: reversal potential [mV].
        :type rev_pot: float or None

    .. method:: paint(region, density)
        :noindex:

        Apply a density mechanism on a region.

        :param str region: description of the region.
        :param density: the density mechanism.
        :type density: :py:class:`density`


    .. method:: place(locations, synapse, label)
        :noindex:

        Place one instance of the synapse mechanism described by ``synapse`` to each location in ``locations``
        and label the group of synapses with ``label``. The label can be used to form connections to one of the
        synapses in the :py:class:`arbor.recipe` by creating a :py:class:`arbor.connection`.

        :param str locations: description of the locset.
        :param synapse: the synapse.
        :type synapse: :py:class:`synapse`
        :param str label: the label of the group of synapses on the locset.

    .. method:: place(locations, junction, label)
        :noindex:

        Place one instance of the gap junction mechanism described by ``junction`` at each location in ``locations``
        and label the group of gap junction sites with ``label``. The label can be used to form gap junction
        connections to/from one of labeled sites in the :py:class:`arbor.recipe` by creating a
        :py:class:`arbor.gap_junction_connection`.

        :param str locations: description of the locset.
        :param junction: the gap junction mechanism.
        :type junction: :py:class:`junction`
        :param str label: the label of the group of gap junction sites on the locset.

    .. method:: place(locations, stim, label)
        :noindex:

        Add a current stimulus at each location in ``locations`` and label the group of stimuli with ``label``.

        :param str locations: description of the locset.
        :param stim: the current stim.
        :type stim: :py:class:`iclamp`
        :param str label: the label of the group of stimuli on the locset.

    .. method:: place(locations, d, label)
        :noindex:

        Add a voltage threshold detector at each location in ``locations`` and label the group of detectors with ``label``.
        The label can be used to form connections from one of the detectors in the :py:class:`arbor.recipe` by creating
        a :py:class:`arbor.connection`.

        :param str locations: description of the locset.
        :param d: description of the detector.
        :type d: :py:class:`threshold_detector`
        :param str label: the label of the group of detectors on the locset.

    .. method:: discretization(policy)

        Set the cv_policy used to discretise the cell into control volumes for simulation.

        :param policy: The cv_policy.
        :type policy: :py:class:`cv_policy`

    .. method:: discretization(policy)
        :noindex:

        Set the cv_policy used to discretise the cell into control volumes for simulation.

        :param str policy: :ref:`string representation <morph-cv-sexpr>` of a cv_policy.

    .. method:: paintings()

        Returns a list of tuples ``(region, painted_object)`` for inspection.

    .. method:: placements()

        Returns a list of tuples ``(locset, placed_object)`` for inspection.

    .. method:: defaults()

        Returns a list of all set defaults for inspection.
