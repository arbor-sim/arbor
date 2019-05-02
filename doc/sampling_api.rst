.. _sampling_api:

Sampling API
============

The new API replaces the flexible but irreducibly inefficient scheme
where the next sample time for a sampling was determined by the
return value of the sampler callback.


Definitions
-----------

probe
    A location or component of a cell that is available for monitoring.

sample
    A record of data corresponding to the value at a specific *probe* at a specific time.

sampler
    A function or function object that receives a sequence of *sample* records.

schedule
    A function or function object that, given a time interval, returns a list of sample times within that interval.



Probes
------

Probes are specified in the recipe objects that are used to initialize a
simulation; the specification of the item or value that is subjected to a
probe will be specific to a particular cell type.

.. container:: api-code

   .. code-block:: cpp

           using probe_tag = int;

           struct probe_info {
               cell_member_type id;   // cell gid, index of probe
               probe_tag tag;         // opaque key, returned in sample record
               any address;           // cell-type specific location info
           };

           probe_info recipe::get_probe(cell_member_type probe_id);


The ``id`` field in the ``probe_info`` struct will be the same value as
the ``probe_id`` used in the ``get_probe`` call.

The ``get_probe()`` method is intended for use by cell group
implementations to set up sampling data structures ahead of time and for
putting in place any structures or information in the concrete cell
implementations to allow monitoring.

The ``tag`` field has no semantics for the engine. It is provided merely
as a way of passing additional metadata about a probe to any sampler
that polls it, with a view to samplers that handle multiple probes,
possibly with different value types.

Probe addresses are now decoupled from the cell descriptions themselves —
this allows a recipe implementation to construct probes independently
of the cells themselves. It is the responsibility of a cell group implementation
to parse the probe address objects wrapped in the ``any address`` field.


Samplers and sample records
---------------------------

Data collected from probes (according to a schedule described below)
will be passed to a sampler function or function object:

.. container:: api-code

    .. code-block:: cpp

            using sampler_function =
                std::function<void (cell_member_type, probe_tag, size_t, const sample_record*)>;

where the parameters are respectively the probe id, the tag, the number
of samples and a pointer to the sequence of sample records.

The ``probe_tag`` is the key given in the ``probe_info`` returned by
the recipe.

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
probe address.

The data pointed to by ``data``, and the sample records themselves, are
only guaranteed to be valid for the duration of the call to the sampler
function. A simple sampler implementation for ``double`` data might be:

.. container:: example-code

    .. code-block:: cpp

            using sample_data = std::map<cell_member_type, std::vector<std::pair<double, double>>>;

            struct scalar_sampler {
                sample_data& samples;

                explicit scalar_sample(sample_data& samples): samples(samples) {}

                void operator()(cell_member_type id, probe_tag, size_t n, const sample_record* records) {
                    for (size_t i=0; i<n; ++i) {
                        const auto& rec = records[i];

                        const double* data = any_cast<const double*>(rec.data);
                        assert(data);
                        samples[id].emplace_back(rec.time, *data);
                    }
                }
            };

The use of ``any_ptr`` allows type-checked access to the sample data, which
may differ in type from probe to probe.


Model and cell group interface
------------------------------

Polling rates, policies and sampler functions are set through the
``simulation`` interface, after construction from a recipe.

.. container:: api-code

    .. code-block:: cpp

            using sampler_association_handle = std::size_t;
            using cell_member_predicate = std::function<bool (cell_member_type)>;

            sampler_association_handle simulation::add_sampler(
                cell_member_predicate probe_ids,
                schedule sched,
                sampler_function fn,
                sampling_policy policy = sampling_policy::lax);

            void simulation::remove_sampler(sampler_association_handle);

            void simulation::remove_all_samplers();

Multiple samplers can then be associated with the same probe locations.
The handle returned is only used for managing the lifetime of the
association. The ``cell_member_predicate`` parameter defines the
set of probe ids in terms of a membership test.

Two helper functions are provided for making ``cell_member_predicate`` objects:

.. container:: api-code

   .. code-block:: cpp

           // Match all probe ids.
           cell_member_predicate all_probes = [](cell_member_type pid) { return true; };

           // Match just one probe id.
           cell_member_predicate one_probe(cell_member_type pid) {
               return [pid](cell_member_type x) { return pid==x; };
           }


The ``sampling_policy`` policy is used to modify sampling behaviour: by
default, the ``lax`` policy is to perform a best-effort sampling that
minimizes sampling overhead and which will not change the numerical
behaviour of the simulation. Other policies may be implemented in the
future, e.g. ``interpolated`` or ``exact``.

The simulation object will pass on the sampler setting request to the cell
group that owns the given probe id. The ``cell_group`` interface will be
correspondingly extended:

.. container:: api-code

   .. code-block:: cpp

           void cell_group::add_sampler(sampler_association_handle h, cell_member_predicate probe_ids, sample_schedule sched, sampler_function fn, sampling_policy policy);

           void cell_group::remove_sampler(sampler_association_handle);

           void cell_group::remove_all_samplers();

Cell groups will invoke the corresponding sampler function directly, and
may aggregate multiple samples with the same probe id in one call to the
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
---------

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
-------------------------------------------

The ``simulation`` and ``mc_cell_group`` classes use classes defined in ``scheduler_map.hpp`` to simplify
the management of sampler--probe associations and probe metdata.

``sampler_association_map`` wraps an ``unordered_map`` between sampler association
handles and tuples (*schedule*, *sampler*, *probe set*), with thread-safe
accessors.

``probe_association_map<Handle>`` is a type alias for an unordered map between
probe ids and tuples (*probe handle*, *probe tag*), where the *probe handle*
is a cell-group specific accessor that allows efficient polling.


Batched sampling in ``mc_cell_group``
-------------------------------------

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
Given an association tuple (*schedule*, *sampler*, *probe set*) where the *schedule*
has (non-zero) *n* sample times in the current integration interval, the ``mc_cell_group`` will
call the *sampler* callback once for probe in *probe set*, with *n* sample values.
