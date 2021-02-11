.. _pycablecell:

Cable cells
===========

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

.. _pycablecell-probes:

Cable cell probes
-----------------

Cable cell probe addresses are defined analagously to their counterparts in
the C++ API (see :ref:`cablecell-probes` for details). Sample data recorded
by the Arbor simulation object is returned in the form of a NumPy array,
with the first column holding sample times, and subsequent columns holding
the corresponding scalar- or vector-valued sample.

Location expressions will be realised as zero or more specific sites on
a cell; probe addresses defined over location expressions will describe zero,
one, or more probes, one per site. They are evaluated in the context of
the cell on which the probe is attached.

Each of the functions described below generates an opaque :class:`probe`
object for use in the recipe :py:func:`get_probes` method.

More information on probes, probe metadata, and sampling can be found
in the documentation for the class :class:`simulation`.

Membrane voltage
   .. py:function:: cable_probe_membrane_voltage(where)

   Cell membrane potential (mV) at the sites specified by the location
   expression string ``where``. This value is spatially interpolated.

   Metadata: the explicit :class:`location` of the sample site.

   .. py:function:: cable_probe_membrane_voltage_cell()

   Cell membrane potential (mV) associated with each cable in each CV of
   the cell discretization.

   Metadata: the list of corresponding :class:`cable` objects.

Axial current
   .. py:function:: cable_probe_axial_current(where)

   Estimation of intracellular current (nA) in the distal direction at the
   sites specified by the location expression string ``where``.

   Metadata: the explicit :class:`location` of the sample site.

Ionic current
   .. py:function:: cable_probe_ion_current_density(where, ion)

   Transmembrane current density (A/m²) associated with the given ``ion`` at
   sites specified by the location expression string ``where``.

   Metadata: the explicit :class:`location` of the sample site.

   .. py:function:: cable_probe_ion_current_cell(ion)

   Transmembrane current (nA) associated with the given ``ion`` across each
   cable in each CV of the cell discretization.

   Metadata: the list of corresponding :class:`cable` objects.

Total ionic current
   .. py:function:: cable_probe_total_ion_current_density(where)

   Transmembrane current density (A/m²) _excluding_ capacitive currents at the
   sites specified by the location expression string ``where``.

   Metadata: the explicit :class:`location` of the sample site.

   .. py:function:: cable_probe_total_ion_current_cell()

   Transmembrane current (nA) _excluding_ capacitive currents across each
   cable in each CV of the cell discretization.

   Metadata: the list of corresponding :class:`cable` objects.

Total transmembrane current
   .. py:function:: cable_probe_total_current_cell()

   Transmembrane current (nA) _including_ capacitive currents across each
   cable in each CV of the cell discretization.

   Metadata: the list of corresponding :class:`cable` objects.

Density mechanism state variable
   .. py:function:: cable_probe_density_state(where, mechanism, state)

   The value of the state variable ``state`` in the density mechanism ``mechanism``
   at the sites specified by the location expression ``where``.

   Metadata: the explicit :class:`location` of the sample site.

   .. py:function:: cable_probe_density_state_cell(mechanism, state)

   The value of the state variable ``state`` in the density mechanism ``mechanism``
   on each cable in each CV of the cell discretixation.

   Metadata: the list of corresponding :class:`cable` objects.

Point process state variable
   .. py:function:: cable_probe_point_state(target, mechanism, state)

   The value of the state variable ``state`` in the point process ``mechanism``
   associated with the target index ``target`` on the cell. If the given mechanism
   is not associated with the target index, no probe will be generated.

   Metadata: an object of type :class:`cable_point_probe_info`, comprising three fields:

   * ``target``: target index on the cell;

   * ``multiplicity``: number of targets sharing the same state in the discretization;

   * ``location``: :class:`location` object corresponding to the target site.

   .. py:function:: cable_probe_point_state_cell(mechanism, state)

   The value of the state variable ``state`` in the point process ``mechanism``
   at each of the targets where that mechanism is defined.

   Metadata: a list of :class:`cable_point_probe_info` values, one for each matching
   target.

Ionic internal concentration
   .. py:function:: cable_probe_ion_int_concentration(where, ion)

   Ionic internal concentration (mmol/L) of the given ``ion`` at the
   sites specified by the location expression string ``where``.

   Metadata: the explicit :class:`location` of the sample site.

   .. py:function:: cable_probe_ion_int_concentration_cell(ion)

   Ionic internal concentration (mmol/L) of the given ``ion`` in each able in each
   CV of the cell discretization.

   Metadata: the list of corresponding :class:`cable` objects.

Ionic external concentration
   .. py:function:: cable_probe_ion_ext_concentration(where, ion)

   Ionic external concentration (mmol/L) of the given ``ion`` at the
   sites specified by the location expression string ``where``.

   Metadata: the explicit :class:`location` of the sample site.

   .. py:function:: cable_probe_ion_ext_concentration_cell(ion)

   Ionic external concentration (mmol/L) of the given ``ion`` in each able in each
   CV of the cell discretization.

   Metadata: the list of corresponding :class:`cable` objects.

