.. _cppcablecell:

Cable cells
===========

.. Warning::
   The interface for building and modifying cable cell objects
   has changed significantly; some of the documentation below is
   out of date.

Cable cells, which use the :cpp:enum:`cell_kind` :cpp:expr:`cable`,
represent morphologically-detailed neurons as 1-d trees, with
electrical and biophysical properties mapped onto those trees.

A single cell is represented by an object of type :cpp:type:`cable_cell`.
Properties shared by all cable cells, as returned by the recipe
:cpp:expr:`get_global_properties` method, are described by an
object of type :cpp:type:`cable_cell_global_properties`.


The :cpp:type:`cable_cell` object
---------------------------------

Cable cells are constructed from a :cpp:type:`morphology`; an optional
:cpp:type:`label_dict` that associates names with particular points
(:cpp:type:`locset` objects) or subsets (:cpp:type:`region` objects) of the
morphology; and an optional :ref:`decor <cablecell-decoration>`.

Morphologies are constructed from a :cpp:type:`segment_tree`, but can also
be generated via the :cpp:type:`stitch_builder`, which offers a slightly
higher level interface. Details are described in :ref:`cppcablecell-morphology-construction`.

Each cell has particular values for its electrical and ionic properties. These
are determined first by the set of global defaults, then the defaults
associated with the cell, and finally by any values specified explicitly for a
given subsection of the morphology via the ``paint`` interface of the decor
(see :ref:`cppcablecell-electrical-properties` and :ref:`cppcablecell-paint-properties`).

Ion channels and other distributed dynamical processes are also specified
on the cell via the ``paint`` method; while synapses, current clamps,
gap junction locations, and the site for testing the threshold potential
are specified via the ``place`` method. See :ref:`cppcablecell-dynamics`, below.

.. _cppcablecell-dynamics:

Cell dynamics
-------------

Each segment in a cell may have attached to it one or more density *mechanisms*,
which describe biophysical processes. These are processes
that are distributed in space, but whose behaviour is defined purely
by the state of the cell and the process at any given point.

Cells may also have *point* mechanisms, describing the dynamics
at post-synaptic sites.

A third type of mechanism, which describes ionic reversal potential
behaviour, can be specified for cells or the whole model via cell parameter
settings, described below.

Mechanisms are described by a :cpp:type:`mechanism_desc` object. These specify the
name of the mechanism (used to find the mechanism in the mechanism catalogue)
and parameter values for the mechanism that apply within a segment.
A :cpp:type:`mechanism_desc` is effectively a wrapper around a name and
a dictionary of parameter/value settings.

Mechanism descriptions can be constructed implicitly from the
mechanism name, and mechanism parameter values then set with the
:cpp:expr:`set` method. Relevant :cpp:type:`mechanism_desc` methods:

.. cpp:function:: mechanism_desc::mechanism_desc(std::string name)

   Construct a mechanism description for the mechanism named `name`.

.. cpp:function:: mechanism_desc& mechanism_desc::set(const std::string& key, double value)

   Sets the parameter associated with :cpp:expr:`key` in the description.
   Returns a reference to the mechanism description, so that calls to
   :cpp:expr:`set` can be chained in a single expression.


Density mechanisms are associated with a cable cell object with:

.. cpp:function:: void cable_cell::paint(const region&, mechanism_desc)

Point mechanisms, which are associated with connection end points on a
cable cell, are attached to a cell with:

.. cpp:function:: void cable_cell::place(const locset&, mechanism_desc)

.. todo::

   TODO: describe other ``place``-able things: current clamps, gap junction
   sites, threshold potential measurement point.

.. _cppcablecell-electrical-properties:

Electrical properties and ion values
-------------------------------------

On each cell segment, electrical and ion properties can be specified by the
:cpp:expr:`parameters` field, of type :cpp:type:`cable_cell_local_parameter_set`.

The :cpp:type:`cable_cell_local_parameter_set` has the following members,
where an empty optional value or missing map key indicates that the corresponding
value should be taken from the cell or global parameter set.

.. cpp:class:: cable_cell_local_parameter_set

   .. cpp:member:: std::unordered_map<std::string, cable_cell_ion_data> ion_data

   The keys of this map are names of ions, whose parameters will be locally overridden.
   The struct :cpp:type:`cable_cell_ion_data` has three fields:
   :cpp:type:`init_int_concentration`, :cpp:type:`init_ext_concentration`, and
   :cpp:type:`init_reversal_potential`.

   Internal and external concentrations are given in millimolars, i.e. mol/m³.
   Reversal potential is given in millivolts.

   .. cpp:member:: util::optional<double> init_membrane_potential

   Initial membrane potential in millivolts.

   .. cpp:member:: util::optional<double> temperature_K

   Local temperature in Kelvin.

   .. cpp:member:: util::optional<double> axial_resistivity

   Local resistivity of the intracellular medium, in ohm-centimetres.

   .. cpp:member:: util::optional<double> membrane_capacitance

   Local areal capacitance of the cell membrane, in Farads per square metre.

   .. cpp:member:: util::optional<cv_policy> discretisation

   Method by which CV boundaries are determined when the cell is discretised.
   See :ref:`cv-policies`.

Default parameters for a cell are given by the :cpp:expr:`default_parameters`
field in the :cpp:type:`cable_cell` object. This is a value of type :cpp:type:`cable_cell_parameter_set`,
which extends :cpp:type:`cable_cell_local_parameter_set` by adding an additional
field describing reversal potential computation:

.. cpp:class:: cable_cell_parameter_set: public cable_cell_local_parameter_set

   .. cpp:member:: std::unordered_map<std::string, mechanism_desc> reversal_potential_method

   Maps the name of an ion to a 'reversal potential' mechanism that describes
   how it should be computed. When no mechanism is provided for an ionic reversal
   potential, the reversal potential will be kept at its initial value.

Default parameters for all cells are supplied in the :cpp:type:`cable_cell_global_properties`
struct.

Global properties
-----------------

.. cpp:class:: cable_cell_global_properties

   .. cpp:member:: const mechanism_catalogue* catalogue

   all mechanism names refer to mechanism instances in this mechanism catalogue.
   by default, this is set to point to `global_default_catalogue()`, the catalogue
   that contains all mechanisms bundled with arbor.

   .. cpp:member:: double membrane_voltage_limit_mv

   if non-zero, check to see if the membrane voltage ever exceeds this value
   in magnitude during the course of a simulation. if so, throw an exception
   and abort the simulation.

   .. cpp:member:: bool coalesce_synapses

   when synapse dynamics are sufficiently simple, the states of synapses within
   the same discretised element can be combined for better performance. this
   is true by default.

   .. cpp:member:: std::unordered_map<std::string, int> ion_species

   every ion species used by cable cells in the simulation must have an entry in
   this map, which takes an ion name to its charge, expressed as a multiple of
   the elementary charge. by default, it is set to include sodium "na" with
   charge 1, calcium "ca" with charge 2, and potassium "k" with charge 1.

   .. cpp:member:: cable_cell_parameter_set default_parameters

   the default electrical and physical properties associated with each cable
   cell, unless overridden locally. in the global properties, *every
   optional field must be given a value*, and every ion must have its default
   values set in :cpp:expr:`default_parameters.ion_data`.

   .. cpp:function:: add_ion(const std::string& ion_name, int charge, double init_iconc, double init_econc, double init_revpot)

   convenience function for adding a new ion to the global :cpp:expr:`ion_species`
   table, and setting up its default values in the `ion_data` table.

   .. cpp:function:: add_ion(const std::string& ion_name, int charge, double init_iconc, double init_econc, mechanism_desc revpot_mechanism)

   As above, but set the initial reversal potential to zero, and use the given mechanism
   for reversal potential calculation.


For convenience, :cpp:expr:`neuron_parameter_defaults` is a predefined :cpp:type:`cable_cell_local_parameter_set`
value that holds values that correspond to NEURON defaults. To use these values,
assign them to the :cpp:expr:`default_parameters` field of the global properties
object returned in the recipe.


Reversal potential dynamics
---------------------------

If no reversal potential mechanism is specified for an ion species, the initial
reversal potential values are maintained for the course of a simulation. Otherwise,
a provided mechanism does the work, but it subject to some strict restrictions.
A reversal potential mechanism described in NMODL:

* May not maintain any STATE variables.
* Can only write to the "eX" value associated with an ion.
* Can not given as a POINT mechanism.

Essentially, reversal potential mechanisms must be pure functions of cellular
and ionic state.

If a reversal potential mechanism writes to multiple ions, then if the mechanism
is given for one of the ions in the global or per-cell parameters, it must be
given for all of them.

Arbor's default catalogue includes a "nernst" reversal potential, which is
parameterized over a single ion, and so can be assigned to e.g. calcium in
the global parameters via

.. code::

   cable_cell_global_properties gprop;
   // ...
   gprop.default_parameters.reversal_potential_method["ca"] = "nernst/ca";


This mechanism has global scalar parameters for the gas constant *R* and
Faraday constant *F*, corresponding to the exact values given by the 2019
redefinition of the SI base units. These values can be changed in a derived
mechanism in order to use, for example, older values of these physical
constants.

.. code::

   mechanism_catalogue mycat(global_default_catalogue());
   mycat.derive("nernst1998", "nernst", {{"R", 8.314472}, {"F", 96485.3415}});

   gprop.catalogue = &mycat;
   gprop.default_parameters.reversal_potential_method["ca"] = "nernst1998/ca";


.. _cppcablecell-paint-properties:

Overriding properties locally
-----------------------------

.. todo::

   TODO: using ``paint`` to specify electrical properties on subsections of
   the morphology.


.. _cppcablecell--probes:

Cable cell probes
-----------------

Various properties of a a cable cell can be sampled via one of the cable cell
specific probe address described below. They fall into two classes: scalar
probes are associated with a single real value, such as a membrane voltage
or mechanism state value at a particular location; vector probes return
multiple values corresponding to a quantity sampled over a whole cell.

The sample data associated with a cable cell probe will either be a ``double``
for scalar probes, or a ``cable_sample_range`` describing a half-open range
of ``double`` values:

.. code::

   using cable_sample_range = std::pair<const double*, const double*>

The probe metadata passed to the sampler will be a const pointer to:

*   ``mlocation`` for most scalar probes;

*   ``cable_probe_point_info`` for point mechanism state queries;

*   ``mcable_list`` for most vector queries;

*   ``std::vector<cable_probe_point_info>`` for cell-wide point mechanism state queries.

The type ``cable_probe_point_info`` holds metadata for a single target on a cell:

.. code::

    struct cable_probe_point_info {
        // Target number of point process instance on cell.
        cell_lid_type target;

        // Number of combined instances at this site.
        unsigned multiplicity;

        // Point on cell morphology where instance is placed.
        mlocation loc;
    };

Note that the ``multiplicity`` will always be 1 if synapse coalescing is
disabled.

Cable cell probes that contingently do not correspond to a valid measurable
quantity are ignored: samplers attached to them will receive no values.
Mechanism state queries however will throw a ``cable_cell_error`` exception
at simulation initialization if the requested state variable does not exist
on the mechanism.

Cable cell probe addresses that are described by a ``locset`` may generate more
than one concrete probe: there will be one per location in the locset that is
satisfiable. Sampler callback functions can distinguish between different
probes with the same address and id by examining their index and/or
probe-specific metadata found in the ``probe_metadata`` parameter.

Membrane voltage
^^^^^^^^^^^^^^^^

.. code::

    struct cable_probe_membrane_voltage {
        locset locations;
    };

Queries cell membrane potential at each site in ``locations``.

*  Sample value: ``double``. Membrane potential in millivolts.

*  Metadata: ``mlocation``. Location of probe.


.. code::

    struct cable_probe_membrane_voltage_cell {};

Queries cell membrane potential across whole cell.

*  Sample value: ``cable_sample_range``. Each value is the
   average membrane potential in millivolts across an unbranched
   component of the cell, as determined by the discretisation.

*  Metadata: ``mcable_list``. Each cable in the cable list describes
   the unbranched component for the corresponding sample value.

Axial current
^^^^^^^^^^^^^

.. code::

    struct cable_probe_axial_current {
        locset locations;
    };

Estimate intracellular current at each site in ``locations``,
in the distal direction.

*  Sample value: ``double``. Current in nanoamperes.

*  Metadata: ``mlocation``. Location as of probe.


Transmembrane current
^^^^^^^^^^^^^^^^^^^^^

.. code::

    struct cable_probe_ion_current_density {
        locset locations;
        std::string ion;
    };

Membrane current density attributed to a particular ion at
each site in ``locations``.

*  Sample value: ``double``. Current density in amperes per square metre.

*  Metadata: ``mlocation``. Location of probe.


.. code::

    struct cable_probe_ion_current_cell {
        std::string ion;
    };

Membrane current attributed to a particular ion across components of the cell.

*  Sample value: ``cable_sample_range``. Each value is the current in
   nanoamperes across an unbranched component of the cell, as determined
   by the discretisation.

*  Metadata: ``mcable_list``. Each cable in the cable list describes
   the unbranched component for the corresponding sample value.


.. code::

    struct cable_probe_total_ion_current_density {
        locset locations;
    };

Membrane current density at given locations _excluding_ capacitive currents.

*  Sample value: ``double``. Current density in amperes per square metre.

*  Metadata: ``mlocation``. Location of probe.


.. code::

    struct cable_probe_total_ion_current_cell {};

Membrane current _excluding_ capacitive currents across components of the cell.

*  Sample value: ``cable_sample_range``. Each value is the current in
   nanoamperes across an unbranched component of the cell, as determined
   by the discretisation.

*  Metadata: ``mcable_list``. Each cable in the cable list describes
   the unbranched component for the corresponding sample value.


.. code::

    struct cable_probe_total_current_cell {};

Total membrance current across components of the cell.

*  Sample value: ``cable_sample_range``. Each value is the current in
   nanoamperes across an unbranched component of the cell, as determined
   by the discretisation.

*  Metadata: ``mcable_list``. Each cable in the cable list describes
   the unbranched component for the corresponding sample value.


Ion concentration
^^^^^^^^^^^^^^^^^

.. code::

    struct cable_probe_ion_int_concentration {
        locset locations;
        std::string ion;
    };

Ionic internal concentration of ion at each site in ``locations``.

*  Sample value: ``double``. Ion concentration in millimoles per litre.

*  Metadata: ``mlocation``. Location of probe.


.. code::

    struct cable_probe_ion_int_concentration_cell {
        std::string ion;
    };

Ionic external concentration of ion across components of the cell.

*  Sample value: ``cable_sample_range``. Each value is the concentration in
   millimoles per lire across an unbranched component of the cell, as determined
   by the discretisation.

*  Metadata: ``mcable_list``. Each cable in the cable list describes
   the unbranched component for the corresponding sample value.


.. code::

    struct cable_probe_ion_ext_concentration {
        mlocation location;
        std::string ion;
    };

Ionic external concentration of ion at each site in ``locations``.

*  Sample value: ``double``. Ion concentration in millimoles per litre.

*  Metadata: ``mlocation``. Location of probe.


.. code::

    struct cable_probe_ion_ext_concentration_cell {
        std::string ion;
    };

Ionic external concentration of ion across components of the cell.

*  Sample value: ``cable_sample_range``. Each value is the concentration in
   millimoles per lire across an unbranched component of the cell, as determined
   by the discretisation.

*  Metadata: ``mcable_list``. Each cable in the cable list describes
   the unbranched component for the corresponding sample value.



Mechanism state
^^^^^^^^^^^^^^^

.. code::

    struct cable_probe_density_state {
        locset locations;
        std::string mechanism;
        std::string state;
    };


Value of state variable in a density mechanism in each site in ``locations``.
If the mechanism is not defined at a particular site, that site is ignored.

*  Sample value: ``double``. State variable value.

*  Metadata: ``mlocation``. Location as given in the probe address.


.. code::

    struct cable_probe_density_state_cell {
        std::string mechanism;
        std::string state;
    };

Value of state variable in a density mechanism across components of the cell.

*  Sample value: ``cable_sample_range``. State variable values from the
   mechanism across unbranched components of the cell, as determined
   by the discretisation and mechanism extent.

*  Metadata: ``mcable_list``. Each cable in the cable list describes
   the unbranched component for the corresponding sample value.


.. code::

    struct cable_probe_point_state {
        cell_lid_type target;
        std::string mechanism;
        std::string state;
    };

Value of state variable in a point mechanism associated with the given target.
If the mechanism is not associated with this target, the probe is ignored.

*  Sample value: ``double``. State variable value.

*  Metadata: ``cable_probe_point_info``. Target number, multiplicity and location.


.. code::

    struct cable_probe_point_state_cell {
        std::string mechanism;
        std::string state;
    };

Value of state variable in a point mechanism for each of the targets in the cell
with which it is associated.

*  Sample value: ``cable_sample_range``. State variable values at each associated
   target.

*  Metadata: ``std::vector<cable_probe_point_info>``. Target metadata for each
   associated target.


.. _cv-policies:

Discretisation and CV policies
------------------------------

The set of boundary points used by the simulator is determined by a
:ref:`CV policy <cablecell-cv-policies>`. These are objects of type
:cpp:class:`cv_policy`, which has the following public methods:

.. cpp:class:: cv_policy

   .. cpp:function:: locset cv_boundary_points(const cable_cell&) const

   Return a locset describing the boundary points for CVs on the given cell.

   .. cpp:function:: region domain() const

   Give the subset of a cell morphology on which this policy has been declared,
   as a morphological ``region`` expression.

Specific CV policy objects are created by functions described below (strictly
speaking, these are class constructors for classes are implicit converted to
``cv_policy`` objects). These all take a ``region`` parameter that restrict the
domain of applicability of that policy; this facility is useful for specifying
differing discretisations on different parts of a cell morphology. When a CV
policy is constrained in this manner, the boundary of the domain will always
constitute part of the CV boundary point set.

CV policies can be :ref:`composed <cablecell-cv-composition>` with ``+`` and ``|`` operators.
For two policies
*A* and *B*, *A* + *B* is a policy which gives boundary points from both *A*
and *B*, while *A* | *B* is a policy which gives all the boundary points from
*B* together with those from *A* which do not within the domain of *B*.
The domain of *A* + *B* and *A* | *B* is the union of the domains of *A* and
*B*.

``cv_policy_single``
^^^^^^^^^^^^^^^^^^^^

.. code::

    cv_policy_single(region domain = reg::all())

Use one CV for the whole cell, or one for each connected component of the
supplied domain.

``cv_policy_explicit``
^^^^^^^^^^^^^^^^^^^^^^

.. code::

   cv_policy_explicit(locset locs, region domain = reg::all())

Use the points given by ``locs`` for CV boundaries, optionally restricted to the
supplied domain.

``cv_policy_every_segment``
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code::

   cv_policy_every_segment(region domain = reg::all())

Use every segment in the morphology as a CV, optionally
restricted to the supplied domain. Each fork point in the domain is
represented by a trivial CV.

``cv_policy_fixed_per_branch``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code::

    cv_policy_fixed_per_branch(unsigned cv_per_branch, region domain, cv_policy_flag::value flags = cv_policy_flag::none);

    cv_policy_fixed_per_branch(unsigned cv_per_branch, cv_policy_flag::value flags = cv_policy_flag::none):

For each branch in each connected component of the domain (or the whole cell,
if no domain is given), evenly distribute boundary points along the branch so
as to produce exactly ``cv_per_branch`` CVs.

By default, CVs will terminate at branch ends. If the flag
``cv_policy_flag::interior_forks`` is given, fork points will be included in
non-trivial, branched CVs and CVs covering terminal points in the morphology
will be half-sized.


``cv_policy_max_extent``
^^^^^^^^^^^^^^^^^^^^^^^^

.. code::

    cv_policy_max_extent(double max_extent, region domain, cv_policy_flag::value flags = cv_policy_flag::none);

    cv_policy_max_extent(double max_extent, cv_policy_flag::value flags = cv_policy_flag::none):

As for ``cv_policy_fixed_per_branch``, save that the number of CVs on any
given branch will be chosen to be the smallest number that ensures no
CV will have an extent on the branch longer than ``max_extent`` micrometres.

