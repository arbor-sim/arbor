.. _pycablecell:

Cable cells
===========

.. toctree::
   :maxdepth: 1

   morphology
   labels
   mechanisms
   decor
   probe_sample
   cable_cell_format

.. currentmodule:: arbor

.. py:class:: cable_cell

    A cable cell is constructed from a :ref:`morphology <morph-morphology>`,
    a :ref:`label dictionary <labels-dictionary>` and a decor.

    .. note::
        The regions and locsets defined in the label dictionary are
        :ref:`thingified <labels-thingify>` when the cable cell is constructed,
        and an exception will be thrown if an invalid label expression is found.

        There are two reasons an expression might be invalid:

        1. Explicit reference to a location of cable that does not exist in the
           morphology, for example ``(branch 12)`` on a cell with 6 branches.
        2. Reference to an incorrect label: circular reference, or a label that does not exist.


    .. code-block:: Python

        import arbor

        # Construct the morphology from an SWC file.
        tree = arbor.load_swc_arbor('granule.swc')
        morph = arbor.morphology(tree)

        # Define regions using standard SWC tags
        labels = arbor.label_dict({'soma': '(tag 1)',
                                   'axon': '(tag 2)',
                                   'dend': '(join (tag 3) (tag 4))'})

        # Define decorations
        decor = arbor.decor()
        decor.paint('"dend"', arbor.density('pas'))
        decor.paint('"axon"', arbor.density('hh'))
        decor.paint('"soma"', arbor.density('hh'))

        # Construct a cable cell.
        cell = arbor.cable_cell(morph, decor, labels)

    .. method:: __init__(morphology, decorations, labels)

        Constructor.

        :param morphology: the morphology of the cell
        :type morphology: :py:class:`morphology` or :py:class:`segment_tree`
        :param labels: dictionary of labeled regions and locsets
        :type labels: :py:class:`label_dict`
        :param decorations: the decorations on the cell
        :type decorations: :py:class:`decor`

    .. method:: placed_lid_range(index)

        Returns the range of local indexes assigned to a placement in the decorations as a tuple of two integers,
        that define the range of indexes as a half open interval.

        :param index: the unique index of the placement.
        :type index: int
        :rtype: tuple(int, int)

.. py:class:: ion

    properties of an ionic species.


.. py:class:: cable_cell_global_properties

   .. property:: catalogue

       All mechanism names refer to mechanism instances in this mechanism
       catalogue. by default, this is set to point to ``default_catalogue()``.

   .. property:: membrane_voltage_limit

       Set a limiter ``U_max`` (mV) on the membrane potential; if ``U > U_max``
       at any point and location the simulation is aborted with an error.
       Defaults to ``None``, if set to a numeric value the limiter is armed.

   .. property:: ion_data

     Return a read-only view onto concentrations, diffusivity, and reversal potential settings.

  .. property:: ion_valence

     Return a read-only view onto species and charge.

  .. property:: ion_reversal_potential

     Return a read-only view onto reversal potential methods (if set).

   .. property:: ions

     Return a read-only view onto all settings.

   .. function:: set_ion(name,
                         charge,
                         internal_concentration, external_concentration,
                         reversal_potential, reversal_potential_method,
                         diffusivty)

      Add a new ion to the global set of known species.

   .. function:: unset_ion(name)

      Remove the named ion.

   .. property:: membrane_potential

    Set the default value for the membrane potential. (``mV``)

   .. property:: membrane_capacitance

    Set the default value for the membrane potential. (``F/m²``)

    .. property:: temperature

    Set the default value for the temperature (``K``).

    .. property:: axial_resisitivity

    Set the default value for the membrane axial resisitivity. (``Ω·cm``)


For convenience, ``neuron_cable_properties`` is a predefined value that holds
values that correspond to NEURON defaults.
