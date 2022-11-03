.. _cppcablecell-probesample:

Cable cell probing and sampling
===============================

.. _cppcablecell-probes:

Cable cell probes
-----------------

Various properties of a cable cell can be sampled. They fall into two classes: scalar
probes are associated with a single real value, such as a membrane voltage
or mechanism state value at a particular location; vector probes return
multiple values corresponding to a quantity sampled at a set of points on the cell.

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

Cable cell probeset addresses that are described by a ``locset`` may generate more
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

Membrane current _excluding_ capacitive currents and stimuli across components of the cell.

*  Sample value: ``cable_sample_range``. Each value is the current in
   nanoamperes across an unbranched component of the cell, as determined
   by the discretisation.

*  Metadata: ``mcable_list``. Each cable in the cable list describes
   the unbranched component for the corresponding sample value.


.. code::

    struct cable_probe_total_current_cell {};

Total membrance current excluding current stimuli across components of the cell.

*  Sample value: ``cable_sample_range``. Each value is the current in
   nanoamperes across an unbranched component of the cell, as determined
   by the discretisation.

*  Metadata: ``mcable_list``. Each cable in the cable list describes
   the unbranched component for the corresponding sample value.

.. code::

    struct cable_probe_stimulus_current_cell {};

Total stimulus currents applied across components of the cell.

*  Sample value: ``cable_sample_range``. Each value is the current in
   nanoamperes across an unbranched component of the cell, as determined
   by the discretisation. Components of CVs where no stimulus is present
   will report a corresponding stimulus value of zero.

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

*  Metadata: ``mlocation``. Location as given in the probeset address.


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


.. _sampling_api:

Sampling API
------------

The new API replaces the flexible but irreducibly inefficient scheme
where the next sample time for a sampling was determined by the
return value of the sampler callback.


Definitions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

probe
    A location or component of a cell that is available for monitoring.

sample
    A record of data corresponding to the value at a specific *probe* at a specific time.

sampler
    A function or function object that receives a sequence of *sample* records.

schedule
    A function or function object that, given a time interval, returns a list of sample times within that interval.



Probes
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Probes are specified in the recipe objects that are used to initialize a
simulation; the specification of the item or value that is subjected to a
probe will be specific to a particular cell type.

.. container:: api-code

   .. code-block:: cpp

           using probe_tag = int;

           struct probe_info {
               probe_tag tag;         // opaque key, returned in sample record
               any address;           // cell-type specific location info

               template <typename X>
               probe_info(X&& a, probe_tag tag = 0):
                  tag(tag), address(std::forward<X>(x)) {}
           };

           std::vector<probe_info> recipe::get_probes(cell_gid_type gid);


The ``tag`` field has no semantics for the engine. It is provided merely
as a way of passing additional metadata about a probe to any sampler
that polls it, with a view to samplers that handle multiple probes,
possibly with different value types.

Probeset addresses are decoupled from the cell descriptions themselves —
this allows a recipe implementation to construct probes independently
of the cells themselves. It is the responsibility of a cell group implementation
to parse the probeset address objects wrapped in the ``any address`` field,
thus the order of probes returned is important.

The _k_th element of the vector returned by ``get_probes(gid)`` is
identified with a probe-id: ``cell_member_type{gid, k}``.

One probeset address may describe more than one concrete probe, depending
upon the interpretation of the probeset address by the cell group. In this
instance, each of the concrete probes will be associated with the
same probe-id. Samplers can distinguish between different probes with
the same id by their probe index (see below).


Samplers and sample records
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Data collected from probes (according to a schedule described below)
will be passed to a sampler function or function object:

.. container:: api-code

    .. code-block:: cpp

            struct probe_metadata {
                cell_member_type id; // probeset id
                probe_tag tag;       // probe tag associated with id
                unsigned index;      // index of probe source within those supplied by probeset id
                util::any_ptr meta;  // probe-specific metadata
            };

            using sampler_function =
                std::function<void (probe_metadata, size_t, const sample_record*)>;

where the parameters are respectively the probe metadata, the number of
samples, and finally a pointer to the sequence of sample records.

The ``probeset_id``, identifies the probe by its probe-id (see above).

The ``probe_tag`` in the metadata is the key given in the ``probe_info``
returned by the recipe.

The ``index`` identifies which of the possibly multiple probes associated
with the probe-id is the source of the samples.

The ``any_ptr`` value in the metadata points to const probe-specific metadata;
the type of the metadata will depend upon the probeset address specified in the
``probe_info`` provided by the recipe.

One ``sample_record`` struct contains one sample of the probe data at a
given simulation time point:

.. container:: api-code

    .. code-block:: cpp

            struct sample_record {
                time_type time;    // simulation time of sample
                any_ptr data;      // sample data
            };

The ``data`` field points to the sample data, wrapped in ``any_ptr`` for
type-checked access. The exact representation will depend on the nature of
the object that is being probed, but it should depend only on the cell type and
probeset address.

The data pointed to by ``data``, and the sample records themselves, are
only guaranteed to be valid for the duration of the call to the sampler
function. A simple sampler implementation for ``double`` data, assuming
one probe per probeset id, might be as follows:

.. container:: example-code

    .. code-block:: cpp

            using sample_data = std::map<cell_member_type, std::vector<std::pair<double, double>>>;

            struct scalar_sampler {
                sample_data& samples;

                explicit scalar_sample(sample_data& samples): samples(samples) {}

                void operator()(probe_metadata pm, size_t n, const sample_record* records) {
                    for (size_t i=0; i<n; ++i) {
                        const auto& rec = records[i];

                        const double* data = any_cast<const double*>(rec.data);
                        assert(data);
                        samples[pm.id].emplace_back(rec.time, *data);
                    }
                }
            };

The use of ``any_ptr`` allows type-checked access to the sample data, which
may differ in type from probe to probe.


Model and cell group interface
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Polling rates, policies and sampler functions are set through the
``simulation`` interface, after construction from a recipe.

.. container:: api-code

    .. code-block:: cpp

            using sampler_association_handle = std::size_t;
            using cell_member_predicate = std::function<bool (cell_member_type)>;

            sampler_association_handle simulation::add_sampler(
                cell_member_predicate probeset_ids,
                schedule sched,
                sampler_function fn,
                sampling_policy policy = sampling_policy::lax);

            void simulation::remove_sampler(sampler_association_handle);

            void simulation::remove_all_samplers();

Multiple samplers can then be associated with the same probe locations.
The handle returned is only used for managing the lifetime of the
association. The ``cell_member_predicate`` parameter defines the
set of probeset ids in terms of a membership test.

Two helper functions are provided for making ``cell_member_predicate`` objects:

.. container:: api-code

   .. code-block:: cpp

           // Match all probeset ids.
           cell_member_predicate all_probes = [](cell_member_type pid) { return true; };

           // Match just one probeset id.
           cell_member_predicate one_probe(cell_member_type pid) {
               return [pid](cell_member_type x) { return pid==x; };
           }


The ``sampling_policy`` policy is used to modify sampling behaviour: by
default, the ``lax`` policy is to perform a best-effort sampling that
minimizes sampling overhead and which will not change the numerical
behaviour of the simulation. The ``exact`` policy requests that samples
are provided for the exact time specified in the schedule, even if this
means disrupting the course of the simulation. Other policies may be
implemented in the future, but cell groups are in general not required
to support any policy other than ``lax``.

The simulation object will pass on the sampler setting request to the cell
group that owns the given probeset id. The ``cell_group`` interface will be
correspondingly extended:

.. container:: api-code

   .. code-block:: cpp

           void cell_group::add_sampler(sampler_association_handle h, cell_member_predicate probeset_ids, sample_schedule sched, sampler_function fn, sampling_policy policy);

           void cell_group::remove_sampler(sampler_association_handle);

           void cell_group::remove_all_samplers();

Cell groups will invoke the corresponding sampler function directly, and
may aggregate multiple samples with the same probeset id in one call to the
sampler. Calls to the sampler are synchronous, in the sense that
processing of the cell group state does not proceed while the sampler
function is being executed, but the times of the samples given to the
sampler will typically precede the time corresponding to the current
state of the cell group. It should be expected that this difference in
time should be no greater the the duration of the integration period
(i.e. ``mindelay/2``).

If a cell group does not support a given ``sampling_policy``, it should
raise an exception. All cell groups should support the ``lax`` policy,
if they support probes at all.


Schedules
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Schedules represent a non-negative, monotonically increasing sequence
of time points, and are used to specify the sampling schedule in any
given association of a sampler function to a set of probes.

A ``schedule`` object has two methods:

.. container:: api-code

   .. code-block:: cpp

       void schedule::reset();

       time_event_span events(time_type t0, time_type t1)

A ``time_event_span`` is a ``std::pair`` of pointers `const time_type*`,
representing a view into an internally maintained collection of generated
time values.

The ``events(t0, t1)`` method returns a view of monotonically
increasing time values in the half-open interval ``[t0, t1)``.
Successive calls to ``events`` — without an intervening call to ``reset()``
—  must request strictly subsequent intervals.

The data represented by the returned ``time_event_span`` view is valid
for the lifetime of the ``schedule`` object, and is invalidated by any
subsequent call to ``reset()`` or ``events()``.

The ``reset()`` method resets the state such that events can be retrieved
from again from time zero. A schedule that is reset must then produce
the same sequence of time points, that is, it must exhibit repeatable
and deterministic behaviour.

The ``schedule`` object itself uses type-erasure to wrap any schedule
implementation class, which can be any copy--constructable class that
provides the methods ``reset()`` and ``events(t0, t1)`` above. Three
schedule implementations are provided by the engine:

.. container:: api-code

   .. code-block:: cpp


           // Schedule at integer multiples of dt:
           schedule regular_schedule(time_type dt);

           // Schedule at a predetermined (sorted) sequence of times:
           template <typename Seq>
           schedule explicit_schedule(const Seq& seq);

           // Schedule according to Poisson process with lambda = 1/mean_dt
           template <typename RandomNumberEngine>
           schedule poisson_schedule(time_type mean_dt, const RandomNumberEngine& rng);

The ``schedule`` class and its implementations are found in ``schedule.hpp``.


Helper classes for probe/sampler management
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``simulation`` and ``mc_cell_group`` classes use classes defined in
``scheduler_map.hpp`` to simplify the management of sampler--probe associations
and probe metadata.

``sampler_association_map`` wraps an ``unordered_map`` between sampler association
handles and tuples (*schedule*, *sampler*, *probe set*, *policy*), with thread-safe
accessors.


Batched sampling in ``mc_cell_group``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``fvm_multicell`` implementations for CPU and GPU simulation of multi-compartment
cable neurons perform sampling in a batched manner: when their integration is
initialized, they take a sequence of ``sample_event`` objects which are used to
populate an implementation-specific ``multi_event_stream`` that describes for each
cell the sample times and what to sample over the integration interval.

When an integration step for a cell covers a sample event on that cell, the sample
is satisfied with the value from the cell state at the beginning of the time step,
after any postsynaptic spike events have been delivered.

It is the responsibility of the ``mc_cell_group::advance()`` method to create the sample
events from the entries of its ``sampler_association_map``, and to dispatch the
sampled values to the sampler callbacks after the integration is complete.
Given an association tuple (*schedule*, *sampler*, *probe set*, *policy*) where the *schedule*
has (non-zero) *n* sample times in the current integration interval, the ``mc_cell_group`` will
call the *sampler* callback once for probe in *probe set*, with *n* sample values.

In addition to the ``lax`` sampling policy, ``mc_cell_group`` supports the ``exact``
policy. Integration steps will be shortened such that any sample times associated
with an ``exact`` policy can be satisfied precisely.

LIF cell probing and sampling
===============================

Membrane voltage
----------------

.. code::

    struct lif_probe_voltage {};

Queries cell membrane potential.

* Sample value: ``double``. Membrane potential (mV).

* Metadata: none
