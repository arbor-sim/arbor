.. _cppcablecell:

Cable cells
===============

.. Warning::
   The interface for building and modifying cable cell objects
   will be thoroughly revised in the near future. The documentation
   here is primarily a place holder.

Cable cells, which use the :cpp:enum:`cell_kind` :cpp:expr:`cable`,
represent morphologically-detailed neurons as 1-d trees, with
electrical and biophysical properties mapped onto those trees.

A single cell is represented by an object of type :cpp:type:`cable_cell`.
Properties shared by all cable cells, as returned by the recipe
:cpp:expr:`get_global_properties` method, are described by an
object of type :cpp:type:`cable_cell_global_properties`.


The :cpp:type:`cable_cell` object
---------------------------------

Cable cells are built up from a series of :cpp:type:`segment`
objects, which themselves describe an unbranched component of the
cell morphology. These segments are added via the methods:

.. cpp:function:: soma_segment* cable_cell::add_soma(double radius)

   Add the soma to the cable cell with the given radius. There
   can be only one per cell.

   The soma segment has index 0, and must be added before any
   cable segments.

.. cpp:function:: cable_segment* cable_cell::add_cable(cell_lid_type index, Args&&... args)

   Add a unbranched section of the cell morphology, with its proximal
   end attached to the segment given by :cpp:expr:`index`. The
   following arguments are forwarded to the :cpp:type:`cable_segment`
   constructor.

Segment indices are exactly the order in which they have been added
to a cell, counting from zero (for the soma). Both :cpp:type:`soma_segment`
and :cpp:type:`cable_segment` are derived from the abstract base
class :cpp:type:`segment`.

.. todo::

   Describe cable_segment constructor arguments, unless we get to the
   replace cell building/morphology implementation first.

Each segment will inherit the electrical properties of the cell, unless
otherwise overriden (see below).


Cell dynamics
-------------

Each segment in a cell may have attached to it one or more density *mechanisms*,
which describe biophysical processes. These are processes
that are distributed in space, but whose behaviour is defined purely
by the state of the cell and the process at any given point.

Cells may also have *point* mechanisms, which are added directly to the
:cpp:type:`cable_cell` object.

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

.. cpp:function:: void segment::add_mechanism(mechanism_desc mech)

Point mechanisms, which are associated with connection end points on a
cable cell, are attached to a cell with:

.. cpp:function:: void cable_cell::add_synapse(mlocation loc, mechanism_desc mech)

where :cpp:type:`mlocation` is a simple struct holding a segment index
and a relative position (from 0, proximal, to 1, distal) along that segment:


Electrical properities and ion values
-------------------------------------

On each cell segment, electrical and ion properties can be specified by the
:cpp:expr:`parameters` field, of type :cpp:type:`cable_cell_local_parameter_set`.

The :cpp:type:`cable_cell_local_parameter_set` has the following members,
where an empty optional value or missing map key indicates that the corresponding
value should be taken from the cell or global parameter set.

.. cpp:class:: cable_cell_local_parameter_set

   .. cpp:member:: std::unordered_map<std::string, cable_cell_ion_data> ion_data

   The keys of this map are names of ions, whose parameters will be locally overriden.
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

   All mechanism names refer to mechanism instances in this mechanism catalogue.
   By default, this is set to point to `global_default_catalogue()`, the catalogue
   that contains all mechanisms bundled with Arbor.

   .. cpp:member:: double membrane_voltage_limit_mV

   If non-zero, check to see if the membrane voltage ever exceeds this value
   in magnitude during the course of a simulation. If so, throw an exception
   and abort the simulation.

   .. cpp:member:: bool coalesce_synapses

   When synapse dynamics are sufficiently simple, the states of synapses within
   the same discretized element can be combined for better performance. This
   is true by default.

   .. cpp:member:: std::unordered_map<std::string, int> ion_species

   Every ion species used by cable cells in the simulation must have an entry in
   this map, which takes an ion name to its charge, expressed as a multiple of
   the elementary charge. By default, it is set to include sodium "na" with
   charge 1, calcium "ca" with charge 2, and potassium "k" with charge 1.

   .. cpp:member:: cable_cell_parameter_set default_parameters

   The default electrical and physical properties associated with each cable
   cell, unless overridden locally. In the global properties, *every
   optional field must be given a value*, and every ion must have its default
   values set in :cpp:expr:`default_parameters.ion_data`.

   .. cpp:function:: add_ion(const std::string& ion_name, int charge, double init_iconc, double init_econc, double init_revpot)

   Convenience function for adding a new ion to the global :cpp:expr:`ion_species`
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
redifinition of the SI base units. These values can be changed in a derived
mechanism in order to use, for example, older values of these physical
constants.

.. code::

   mechanism_catalogue mycat(global_default_catalogue());
   mycat.derive("nernst1998", "nernst", {{"R", 8.314472}, {"F", 96485.3415}});

   gprop.catalogue = &mycat;
   gprop.default_parameters.reversal_potential_method["ca"] = "nernst1998/ca";


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

Membrane voltage
^^^^^^^^^^^^^^^^

.. code::

    struct cable_probe_membrane_voltage {
        mlocation location;
    };

Queries cell membrane potential at the specified location.

*  Sample value: ``double``. Membrane potential in millivolts.

*  Metadata: ``mlocation``. Location as given in the probe address.


.. code::

    struct cable_probe_membrane_voltage_cell {};

Queries cell membrane potential across whole cell.

*  Sample value: ``cable_sample_range``. Each value is the
   average membrane potential in millivolts across an unbranched
   component of the cell, as determined by the discretization.

*  Metadata: ``mcable_list``. Each cable in the cable list describes
   the unbranched component for the corresponding sample value.

Axial current
^^^^^^^^^^^^^

.. code::

    struct cable_probe_axial_current {
        mlocation location;
    };

Estimate intracellular current at given location in the distal direction.

*  Sample value: ``double``. Current in nanoamperes.

*  Metadata: ``mlocation``. Location as given in the probe address.


Transmembrane current
^^^^^^^^^^^^^^^^^^^^^

.. code::

    struct cable_probe_ion_current_density {
        mlocation location;
        std::string ion;
    };

Membrance current density attributed to a particular ion at a given location.

*  Sample value: ``double``. Current density in amperes per square metre.

*  Metadata: ``mlocation``. Location as given in the probe address.


.. code::

    struct cable_probe_ion_current_cell {
        std::string ion;
    };

Membrane current attributed to a particular ion across components of the cell.

*  Sample value: ``cable_sample_range``. Each value is the current in
   nanoamperes across an unbranched component of the cell, as determined
   by the discretization.

*  Metadata: ``mcable_list``. Each cable in the cable list describes
   the unbranched component for the corresponding sample value.


.. code::

    struct cable_probe_total_ion_current_density {
        mlocation location;
    };

Membrane current density at gvien location _excluding_ capacitive currents.

*  Sample value: ``double``. Current density in amperes per square metre.

*  Metadata: ``mlocation``. Location as given in the probe address.


.. code::

    struct cable_probe_total_ion_current_cell {};

Membrane current _excluding_ capacitive currents across components of the cell.

*  Sample value: ``cable_sample_range``. Each value is the current in
   nanoamperes across an unbranched component of the cell, as determined
   by the discretization.

*  Metadata: ``mcable_list``. Each cable in the cable list describes
   the unbranched component for the corresponding sample value.


.. code::

    struct cable_probe_total_current_cell {};

Total membrance current across components of the cell.

*  Sample value: ``cable_sample_range``. Each value is the current in
   nanoamperes across an unbranched component of the cell, as determined
   by the discretization.

*  Metadata: ``mcable_list``. Each cable in the cable list describes
   the unbranched component for the corresponding sample value.


Ion concentration
^^^^^^^^^^^^^^^^^

.. code::

    struct cable_probe_ion_int_concentration {
        mlocation location;
        std::string ion;
    };

Ionic internal concentration of ion at given location.

*  Sample value: ``double``. Ion concentration in millimoles per litre.

*  Metadata: ``mlocation``. Location as given in the probe address.


.. code::

    struct cable_probe_ion_int_concentration_cell {
        std::string ion;
    };

Ionic external concentration of ion across components of the cell.

*  Sample value: ``cable_sample_range``. Each value is the concentration in
   millimoles per lire across an unbranched component of the cell, as determined
   by the discretization.

*  Metadata: ``mcable_list``. Each cable in the cable list describes
   the unbranched component for the corresponding sample value.


.. code::

    struct cable_probe_ion_ext_concentration {
        mlocation location;
        std::string ion;
    };

Ionic internal concentration of ion at given location.

*  Sample value: ``double``. Ion concentration in millimoles per litre.

*  Metadata: ``mlocation``. Location as given in the probe address.


.. code::

    struct cable_probe_ion_ext_concentration_cell {
        std::string ion;
    };

Ionic external concentration of ion across components of the cell.

*  Sample value: ``cable_sample_range``. Each value is the concentration in
   millimoles per lire across an unbranched component of the cell, as determined
   by the discretization.

*  Metadata: ``mcable_list``. Each cable in the cable list describes
   the unbranched component for the corresponding sample value.



Mechanism state
^^^^^^^^^^^^^^^

.. code::

    struct cable_probe_density_state {
        mlocation location;
        std::string mechanism;
        std::string state;
    };


Value of state variable in a density mechanism in at given location.
If the mechanism is not defined at the location, the probe is ignored.

*  Sample value: ``double``. State variable value.

*  Metadata: ``mlocation``. Location as given in the probe address.


.. code::

    struct cable_probe_density_state_cell {
        std::string mechanism;
        std::string state;
    };

Value of state variable in adensity mechanism across components of the cell.

*  Sample value: ``cable_sample_range``. State variable values from the
   mechanism across unbranched components of the cell, as determined
   by the discretization and mechanism extent.

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
