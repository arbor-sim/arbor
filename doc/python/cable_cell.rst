.. _pycable_cell:

Cable cells
===========

.. currentmodule:: arbor

.. py:class:: cable_cell

    .. method:: cable_cell(morphology, label_dict=None)
    Construct a cable cell from a :ref:`morphology <morph-morphology>`
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

    .. method:: set_properties(Vm=None, cm=None, rL=None, tempK=None)

        Set default values of cable properties on the whole cell.
        Overrides the default global values, and can be overridden by painting
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

    .. method:: compartments_length(length)

        Adjust the :ref:`compartments length <cable-discretisation>`.

        :param int length: length of compartments [μm].

        Defaults to one control volume per branch.

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
            cell.paint('"soma"', rL=100)
            # Specialize resistivity and capacitance on the axon, where
            # axon is defined using a region expression.
            cell.paint('(tag 2)', cm=0.05, rL=80)

.. py:class:: ion

    properties of an ionic species.


