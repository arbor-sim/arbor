.. _pycablecell:

Cable cells
===========

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
        CVs in the cell, and can't be overriden locally.

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
        :noindex:

        Set ion species properties initial conditions on a region.

        :param str name: name of the ion species.
        :param float int_con: initial internal concentration [mM].
        :type int_con: float or None.
        :param float ext_con: initial external concentration [mM].
        :type ext_con: float or None.
        :param float rev_pot: reversal potential [mV].
        :type rev_pot: float or None

    .. method:: paint(region, mechanism)
        :noindex:

        Apply a mechanism with a region.
        Returns a unique identifier that can be used to query the local indexes (see :gen:`index`) assigned to the placed items on the cable cell.

        :param str region: description of the region.
        :param mechanism: the mechanism.
        :type mechanism: :py:class:`mechanism`

    .. method:: paint(region, mech_name)
        :noindex:

        Apply a mechanism with a region using the name of the mechanism.
        The mechanism will use the parameter values set in the mechanism catalogue.
        Returns a unique identifier that can be used to query the local indexes (see :gen:`index`) assigned to the placed items on the cable cell.

        :param str region: description of the region.
        :param str mechanism: the name of the mechanism.

    .. method:: place(locations, const arb::mechanism_desc& d)

        Place one instance of synapse described by ``mechanism`` to each location in ``locations``.
        Returns a unique identifier that can be used to query the local indexes (see :gen:`index`) assigned to the placed items on the cable cell.

        :param str locations: description of the locset.
        :param str mechanism: the name of the mechanism.
        :rtype: int

    .. method:: place(locations, mechanism)
        :noindex:

        Place one instance of synapse described by ``mechanism`` to each location in ``locations``.
        Returns a unique identifier that can be used to query the local indexes (see :gen:`index`) assigned to the placed items on the cable cell.

        :param str locations: description of the locset.
        :param mechanism: the mechanism.
        :type mechanism: :py:class:`mechanism`
        :rtype: int

    .. method:: place(locations, site)
        :noindex:

        Place one gap junction site at each location in ``locations``.
        Returns a unique identifier that can be used to query the local indexes (see :gen:`index`) assigned to the placed items on the cable cell.

        :param str locations: description of the locset.
        :param site: indicates a gap junction site..
        :type site: :py:class:`gap_junction_site`
        :rtype: int

    .. method:: place(locations, stim)
        :noindex:

        Add a current stimulus at each location in ``locations``.
        Returns a unique identifier that can be used to query the local indexes (see :gen:`index`) assigned to the placed items on the cable cell.

        :param str locations: description of the locset.
        :param stim: the current stim.
        :type stim: :py:class:`i_clamp`
        :rtype: int

    .. method:: place(locations, d)
        :noindex:

        Add a voltage spike detector at each location in ``locations``.
        Returns a unique identifier that can be used to query the local indexes (see :gen:`index`) assigned to the placed items on the cable cell.

        :param str locations: description of the locset.
        :param d: description of the detector.
        :type d: :py:class:`threshold_detector`
        :rtype: int

    .. method:: discretization(policy)

        Set the cv_policy used to discretise the cell into control volumes for simulation.

        :param policy: The cv_policy.
        :type policy: :py:class:`cv_policy`

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


.. _pycablecell-cv-policies:

Discretisation and CV policies
------------------------------

The set of boundary points used by the simulator is determined by a
:ref:`CV policy <cablecell-cv-policies>`. These are objects of type
:cpp:class:`cv_policy`, which has the following public methods:

.. py:class:: cv_policy

   .. attribute:: domain

       A read only string expression describing the subset of a cell morphology
       (region) on which this policy has been declared.

   CV policies can be :ref:`composed <cablecell-cv-composition>` with
   ``+`` and ``|`` operators.

   .. code-block:: Python

       # The plus operator applies 
       policy = arbor.cv_policy_single('"soma"') + cv_policy('"dend"')

       # The | operator uses CVs of length 10 μm everywhere, except
       # on the soma, to which a single CV policy is applied.
       policy = arbor.cv_policy_max_extent(10) | cv_policy_single('"soma"')

Specific CV policy objects are created by functions described below.
These all take a ``region`` parameter that restrict the
domain of applicability of that policy; this facility is useful for specifying
differing discretisations on different parts of a cell morphology. When a CV
policy is constrained in this manner, the boundary of the domain will always
constitute part of the CV boundary point set.

.. py:function:: cv_policy_single(domain='(all)')

    Use one CV for the whole cell, or one for each connected component of the
    supplied domain.

    .. code-block:: Python

        # Use one CV for the entire cell (a single compartment model)
        single_comp = arbor.cv_policy_single()

        # Use a single CV for the soma.
        single_comp_soma = arbor.cv_policy_single('"soma"')

    :param str domain: The region on which the policy is applied.

.. py:function:: cv_policy_every_segment(domain='(all)')

    Use every sample point in the morphology definition as a CV boundary, optionally
    restricted to the supplied domain. Each fork point in the domain is
    represented by a trivial CV.

    :param str domain: The region on which the policy is applied.

.. py:function:: cv_policy_fixed_per_branch(cv_per_branch, domain='(all)')

    For each branch in each connected component of the domain (or the whole cell,
    if no domain is given), evenly distribute boundary points along the branch so
    as to produce exactly ``cv_per_branch`` CVs.

    :param int cv_per_branch: The number of CVs per branch.
    :param str domain: The region on which the policy is applied.

.. py:function:: cv_policy_max_extent(max_extent, domain='(all)')

    As for :py:func:`cv_policy_fixed_per_branch`, save that the number of CVs on any
    given branch will be chosen to be the smallest number that ensures no
    CV will have an extent on the branch longer than ``max_extent`` micrometres.

    :param float max_etent: The maximum length for generated CVs.
    :param str domain: The region on which the policy is applied.
