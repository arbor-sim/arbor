.. _cppcablecell-probesample:

Cable cell probing and sampling
===============================

.. _cppcablecell-probes:

Cable cell probes
-----------------

The probe metadata passed to the sampler will be a const pointer to:

* ``cable_probe_point_info`` for point mechanism state queries,
* ``mcable`` for cell-wide probes,
* ``mlocation`` for probes on locsets;

where the type ``cable_probe_point_info`` holds metadata for a point process
state variable

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

Value types will always be conforming to ``const double*``.

Cable cell probes that contingently do not correspond to a valid measurable
quantity are ignored: samplers attached to them will receive no values.
Mechanism state queries, however will throw a ``cable_cell_error`` exception at
simulation initialization if the requested state variable does not exist on the
mechanism.

Membrane voltage
^^^^^^^^^^^^^^^^

.. code::

    struct cable_probe_membrane_voltage {
        locset locations;
    };

Queries cell membrane potential at each site in ``locations``.

*  Sample value: ``double``. Membrane potential in millivolts.
*  Metadata: ``mlocation``. Locations of probe.

.. code::

    struct cable_probe_membrane_voltage_cell {};

Queries cell membrane potential across the whole cell.

* Sample value: the average membrane potential in millivolts across an
   unbranched component of the cell, as determined by the discretisation.
* Metadata: ``mcable``. Each cable in the cable list describes the unbranched
   component for the corresponding sample value.

Axial current
^^^^^^^^^^^^^

.. code::

    struct cable_probe_axial_current {
        locset locations;
    };

Estimate intracellular current at each site in ``locations``,
in the distal direction.

*  Sample value: Current in nanoamperes.
*  Metadata: ``mlocation``. Locations as of probe.

Transmembrane current
^^^^^^^^^^^^^^^^^^^^^

.. code::

    struct cable_probe_ion_current_density {
        locset locations;
        std::string ion;
    };

Membrane current density attributed to a particular ion at
each site in ``locations``.

*  Sample value: Current density in amperes per square metre.
*  Metadata: ``mlocation``. Location of probe.

.. code::

    struct cable_probe_ion_current_cell {
        std::string ion;
    };

Membrane current attributed to a particular ion across components of the cell.

* Sample value: the current in nanoamperes across an unbranched component of the
   cell, as determined by the discretisation.
* Metadata: ``mcable``. Each cable in the cable list describes the unbranched
   component for the corresponding sample value.

.. code::

    struct cable_probe_total_ion_current_density {
        locset locations;
    };

Membrane current density at given locations _excluding_ capacitive currents.

*  Sample value: Current density in amperes per square metre.
*  Metadata: ``mlocation``. Locations of probe.

.. code::

    struct cable_probe_total_ion_current_cell {};

Membrane current _excluding_ capacitive currents and stimuli across components of the cell.

*  Sample value: the current in
   nanoamperes across an unbranched component of the cell, as determined
   by the discretisation.
*  Metadata: ``mcable``. Each cable in the cable list describes
   the unbranched component for the corresponding sample value.

.. code::

    struct cable_probe_total_current_cell {};

Total membrane current excluding current stimuli across components of the cell.

* Sample value: Each value is the current in nanoamperes across an unbranched
   component of the cell, as determined by the discretisation.
* Metadata: ``mcable``. Each cable in the cable list describes the unbranched
   component for the corresponding sample value.

.. code::

    struct cable_probe_stimulus_current_cell {};

Total stimulus currents applied across components of the cell.

* Sample value: Each value is the current in nanoamperes across an unbranched
   component of the cell, as determined by the discretisation. Components of CVs
   where no stimulus is present will report a corresponding stimulus value of
   zero.
* Metadata: ``mcable``. Each cable in the cable list describes the unbranched
   component for the corresponding sample value.

Ion concentration
^^^^^^^^^^^^^^^^^

.. code::

    struct cable_probe_ion_int_concentration {
        locset locations;
        std::string ion;
    };

Ionic internal concentration of ion at each site in ``locations``.

* Sample value: Ion concentration in millimoles per litre.
* Metadata: ``mlocation``. Locations of probe.

.. code::

    struct cable_probe_ion_int_concentration_cell {
        std::string ion;
    };

Ionic external concentration of ion across components of the cell.

* Sample value: the concentration in millimoles per lire across an unbranched
   component of the cell, as determined by the discretisation.
* Metadata: ``mcable``. Each cable in the cable list describes the unbranched
   component for the corresponding sample value.

.. code::

    struct cable_probe_ion_ext_concentration {
        locset location;
        std::string ion;
    };

Ionic external concentration of ion at each site in ``locations``.

*  Sample value: Ion concentration in millimoles per litre.
*  Metadata: ``mlocation``. Locations of probe.

.. code::

    struct cable_probe_ion_ext_concentration_cell {
        std::string ion;
    };

Ionic external concentration of ion across components of the cell.

* Sample value: the concentration in millimoles per litre across an unbranched
   component of the cell, as determined by the discretisation.
*  Metadata: ``mcable``. Each cable in the cable list describes
   the unbranched component for the corresponding sample value.

Ionic diffusion concrentration.

.. code::

    struct cable_probe_ion_diff_concentration {
        locset locations;
        std::string ion;
    };

Diffusive ionic concentration of the given ``ion`` at the sites specified by
``locations``.

* Sample value: the concentration in millimoles per litre across an unbranched
   component of the cell, as determined by the discretisation.
*  Metadata: ``mcable``. Each cable in the cable list describes
   the unbranched component for the corresponding sample value.

.. code::

    struct cable_probe_ion_diff_concentration_cell {
        std::string ion;
    };

Ionic diffusion concrentration attributed to a particular ``ion`` across CVs of
the cell.

* Sample value: the concentration in millimoles per litre across an unbranched
   component of the cell, as determined by the discretisation.
*  Metadata: ``mcable``. Each cable in the cable list describes
   the unbranched component for the corresponding sample value.

Mechanism state
^^^^^^^^^^^^^^^

.. code::

    struct cable_probe_density_state {
        locset locations;
        std::string mechanism;
        std::string state;
    };

Value of state variable in a density mechanism in each site in ``locations``. If
the mechanism is not defined at a particular site, that site is ignored.

*  Sample value: State variable value.
* Metadata: ``mlocation``. Locations as given in the probeset address.

.. code::

    struct cable_probe_density_state_cell {
        std::string mechanism;
        std::string state;
    };

Value of state variable in a density mechanism across components of the cell.

* Sample value: State variable values from the mechanism across unbranched
   components of the cell, as determined by the discretisation and mechanism
   extent.
*  Metadata: ``mcable``. Each cable in the cable list describes
   the unbranched component for the corresponding sample value.

.. code::

    struct cable_probe_point_state {
        cell_lid_type target;
        std::string mechanism;
        std::string state;
    };

Value of state variable in a point mechanism associated with the given target.
If the mechanism is not associated with this target, the probe is ignored.

*  Sample value: State variable value.
*  Metadata: ``cable_probe_point_info``. Target number, multiplicity and location.

.. code::

    struct cable_probe_point_state_cell {
        std::string mechanism;
        std::string state;
    };

Value of state variable in a point mechanism for each of the targets in the cell
with which it is associated.

* Sample value: State variable values at each associated target.
* Metadata: ``cable_probe_point_info``. Target metadata for each associated
   target.

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
^^^^^^

Probes are specified in the recipe objects that are used to initialize a
simulation; the specification of the item or value that is subjected to a
probe will be specific to a particular cell type.

.. container:: api-code

   .. code-block:: cpp

            struct probe_info {
               cell_tag_type tag;     // opaque key, returned in sample record
               any address;           // cell-type specific location info

               template <typename X>
               probe_info(X&& x, nullptr_t) = delete;

               template <typename X>
               probe_info(X&& x, const cell_tag_type& tag):
                   tag(tag), address(std::forward<X>(x)) {}
           };

           std::vector<probe_info> recipe::get_probes(cell_gid_type gid);


The ``tag`` field identifies the probe locally on this global id ``gid``, e.g.
it is used in conjunction to attach samplers,  as ``cell_address_type{gid, tag}``.

Probeset addresses are decoupled from the cell descriptions themselves — this
allows a recipe implementation to construct probes independently of the cells
themselves. It is the responsibility of a cell group implementation to parse the
probeset address objects wrapped in the ``any address`` field, thus the order of
probes returned is important.

One probeset address may describe more than one concrete probe, depending upon
the interpretation of the probeset address by the cell group. In this instance,
each of the concrete probes will be associated with the same probe-id. Samplers
can distinguish between different probes with the same id by their probe index
(see below).

Samplers and sample records
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Data collected from probes (according to a schedule described below)
will be passed to a sampler function or function object:

.. container:: api-code

    .. code-block:: cpp

            struct probe_metadata {
                cell_address_type id;  // probeset id
                unsigned index;        // index of probe source within those supplied by probeset id
                std::size_t width = 0; // count of sample _columns_
                util::any_ptr meta;    // probe-specific metadata
            };

            struct sample_records {
                std::size_t n_sample;  // count of sample _rows_
                std::size_t width;     // count of sample _columns_
                const time_type* time; // pointer to time data
                std::any values;       // resolves to pointer of probe-specific data D[n_sample][width]
            };

            using sampler_function = std::function<void(const probe_metadata&, const sample_records&)>;

where the parameters are respectively the probe metadata, the number of
samples, and finally a pointer to the sequence of sample records.

The ``probeset_id``, identifies the probe by its probe-id (see above).

The ``index`` identifies which of the possibly multiple probes associated
with the probe-id is the source of the samples.

The ``any_ptr`` value in the metadata points to const probe-specific metadata;
the type of the metadata will depend upon the probeset address specified in the
``probe_info`` provided by the recipe.

The raw data in ``values`` can --- given knowledge of the correct type
information --- be cast to the correct type ``const T*`` and read traversing in
order ``T[n_sample][width]``. Likewise, ``meta`` can be cast to the metadata
type ``const M*`` and yields an array ``M[width]``.

Each probe type has type definitions for the associated value and metadata
types, e.g.

.. container:: example-code

    .. code-block:: cpp

        struct cable_probe_membrane_voltage {
            using value_type = cable_sample_type;
            using meta_type = cable_state_meta_type;
            locset locations;
        };

Access is made much more convenient through ``sample_reader``, see next section.

Sample Data Access
^^^^^^^^^^^^^^^^^^

The ``sample_reader`` provides a convenient way of accessing data retrieved in a
sampler callback, taking care of casting and the data layout. It can be used as
follows, provided the probe is known

.. container:: example-code

    .. code-block:: cpp

        // This is the probe type we will attach to
        using probe_type = cable_probe_membrane_voltage_cell;

        // This is the callback to attach
        void callback(const probe_metadata& pm, const sample_records& recs) {
            auto reader = sample_reader<probe_type::meta_type>(pm.meta, recs);

            for (std::size_t ix = 0ul; ix < reader.n_row(); ++ix) {
                auto time = reader.time(ix);
                for (std::size_t iy = 0ul; iy < reader.n_column(); ++iy) {
                    auto value = reader.value(ix, iy);
                    auto cable = reader.metadata(iy);
                    // ... use time, cable, value ...
                }
            }

In general, it provides safe access to the raw samples, time, and metadata and allows
treating ``sample_records`` like tabular data with ``width`` columns containing the
``metadata`` and ``n_sample`` rows containing ``time`` and ``values``.

.. container:: example-code

    .. code-block:: cpp

        template<typename M>
        struct sample_reader {
            using meta_type = M;
            using value_type = probe_value_type_of_t<M>;

            std::size_t n_row() const { return n_sample_; }
            std::size_t n_column() const { return width_; }

            // Retrieve sample value corresponding to
            // - time=time(i)
            // - location=metadata(j)
            value_type value(std::size_t i, std::size_t j = 0) const;
            // Retrieve i'th time
            time_type time(std::size_t i) const;
            // Retrieve metadata at j
            meta_type metadata(std::size_t j) const;
        };

Note that for many cases a ``simple_sampler`` is provided which records tabular
data into ``simple_sampler_result`` and can be attached to probes like this

.. container:: example-code

    .. code-block:: cpp

        // The schedule for sampling every 1 ms.
        auto sched = arb::regular_schedule(1*arb::units::ms);
        // This is where the voltage samples will be stored as (time, value) pairs
        sample_result voltage;
        // Now attach the sampler at probeset_id, with sampling schedule sched, writing to voltage
        sim.add_sampler(arb::one_probe(probeset_id), sched, arb::make_simple_sampler(voltage));

Then,

.. cpp:class:: simple_sampler_result


    .. cpp:member:: std::size_t n_sample

        number of rows

    .. cpp:member:: std::size_t width

        number of columns

    .. cpp:member:: std::vector<time_type> time

        sample times, one entry per row

    .. cpp:member:: std::vector<std::remove_const_t<M>> metadata

        probe specific metadata, one entry per column

    .. cpp:member:: std::vector<std::vector<std::remove_const_t<value_type>>> values

        values, one entry per row, each entry is a vector with one entry per column

can be used to retrieve the data.

Model and cell group interface
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Polling rates and sampler functions are set through the
``simulation`` interface, after construction from a recipe.

.. container:: api-code

    .. code-block:: cpp

            using sampler_association_handle = std::size_t;
            using cell_member_predicate = std::function<bool (cell_member_type)>;

            sampler_association_handle simulation::add_sampler(cell_member_predicate probeset_ids,
                                                               schedule sched,
                                                               sampler_function fn)

            void simulation::remove_sampler(sampler_association_handle);

            void simulation::remove_all_samplers();

Multiple samplers can then be associated with the same probe locations.
The handle returned is only used for managing the lifetime of the
association. The ``cell_member_predicate`` parameter defines the
set of probeset ids in terms of a membership test.

We provide a few helper functions are provided for making ``cell_member_predicate`` objects:

.. container:: api-code

   .. code-block:: cpp

           // Match all probeset ids.
           cell_member_predicate
           all_probes = [](const cell_address_type& pid) { return true; };

           // Match just one probeset id.
           cell_member_predicate
           one_probe(const cell_address_type& pid) { return [pid](const auto& x) { return pid==x; }; }

           // Match all probes on a given ``gid``.
           cell_member_predicate
           one_gid(const cell_gid_type& gid) { return [gid](const auto& x) { return gid==x.gid; }; }

           // Match all probes with a given ``tag``.
           cell_member_predicate
           one_tag(const cell_tag_type& tag) { return [tag](const auto& x) { return tag==x.tag; }; }

The simulation object will pass on the sampler setting request to the cell
group that owns the given probeset id. The ``cell_group`` interface will be
correspondingly extended:

.. container:: api-code

   .. code-block:: cpp

           void cell_group::add_sampler(sampler_association_handle h,
                                       cell_member_predicate probeset_ids,
                                       sample_schedule sched,
                                       sampler_function fn);

           void cell_group::remove_sampler(sampler_association_handle);

           void cell_group::remove_all_samplers();

Cell groups will invoke the corresponding sampler function directly, and may
aggregate multiple samples with the same probeset id in one call to the sampler.
Calls to the sampler are synchronous, in the sense that processing of the cell
group state does not proceed while the sampler function is being executed, but
the times of the samples given to the sampler will typically precede the time
corresponding to the current state of the cell group. It should be expected that
this difference in time should be no greater the the duration of the integration
period (i.e. ``mindelay/2``).

Schedules
^^^^^^^^^

Schedules represent a non-negative, monotonically increasing sequence of time
points, and are used to specify the sampling schedule in any given association
of a sampler function to a set of probes.

A ``schedule`` object has two methods:

.. container:: api-code

   .. code-block:: cpp

       void schedule::reset();

       time_event_span events(time_type t0, time_type t1)

A ``time_event_span`` is a ``std::pair`` of pointers `const time_type*`,
representing a view into an internally maintained collection of generated time
values.

The ``events(t0, t1)`` method returns a view of monotonically increasing time
values in the half-open interval ``[t0, t1)``. Successive calls to ``events`` —
without an intervening call to ``reset()`` — must request strictly subsequent
intervals.

The data represented by the returned ``time_event_span`` view is valid for the
lifetime of the ``schedule`` object, and is invalidated by any subsequent call
to ``reset()`` or ``events()``.

The ``reset()`` method resets the state such that events can be retrieved
from again from time zero. A schedule that is reset must then produce
the same sequence of time points, that is, it must exhibit repeatable
and deterministic behaviour.

The ``schedule`` object itself uses type-erasure to wrap any schedule
implementation class, which can be any copy--constructible class that provides
the methods ``reset()`` and ``events(t0, t1)`` above. Three schedule
implementations are provided by the engine:

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

The ``simulation`` and ``cable_cell_group`` classes use classes defined in
``scheduler_map.hpp`` to simplify the management of sampler--probe associations
and probe metadata.

``sampler_association_map`` wraps an ``unordered_map`` between sampler
association handles and tuples (*schedule*, *sampler*, *probe set*), with
thread-safe accessors.

Batched sampling in ``cable_cell_group``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``fvm_multicell`` implementations for CPU and GPU simulation of
multi-compartment cable neurons perform sampling in a batched manner: when their
integration is initialized, they take a sequence of ``sample_event`` objects
which are used to populate an implementation-specific ``event_stream`` that
describes for each cell the sample times and what to sample over the integration
interval.

When an integration step for a cell covers a sample event on that cell, the
sample is satisfied with the value from the cell state at the beginning of the
time step, after any postsynaptic spike events have been delivered.

It is the responsibility of the ``cable_cell_group::advance()`` method to create
the sample events from the entries of its ``sampler_association_map``, and to
dispatch the sampled values to the sampler callbacks after the integration is
complete. Given an association tuple (*schedule*, *sampler*, *probe set*) where
the *schedule* has (non-zero) *n* sample times in the current integration
interval, the ``cable_cell_group`` will call the *sampler* callback once for
probe in *probe set*, with *n* sample values.

.. note::

   When the time values returned by a call to a schedule's ``events(t0, t1)``
   method do not perfectly coincide with the boundaries of the numerical time
   step grid, :math:`[t_0, t_0 + dt, t_0 + 2\, dt, \, \cdots \, , t_1)`, the
   samples will be taken at the closest possible point in time. In particular,
   any sample times :math:`t_s \in \left( t_i - dt/2,~ t_i + dt/2\right]` are
   attributed to simulation time step :math:`t_i = t_0 + i\,dt`.
