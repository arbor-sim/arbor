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
        decor.paint('"dend"', 'pas')
        decor.paint('"axon"', 'hh')
        decor.paint('"soma"', 'hh')

        # Construct a cable cell.
        cell = arbor.cable_cell(morph, labels, decor)

    .. method:: __init__(morphology, labels, decorations)

        Constructor.

        :param morphology: the morphology of the cell
        :type morphology: :py:class:`morphology`
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
