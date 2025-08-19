.. _cppgenerators:

Schedules
=========

Generate sorted time points.

.. cpp:namespace:: arb

.. cpp:class:: schedule

    Opaque representation of a schedule.

.. cpp:function:: schedule regular_schedule(const units::quantity& t0, const units::quantity& dt, const units::quantity& t1 = std::numeric_limits<time_type>::max()*units::ms);

    Regular schedule with start ``t0``, interval ``dt``, and optional end ``t1``.

.. cpp:function:: schedule regular_schedule(const units::quantity& dt);

   Regular schedule with interval ``dt``.

.. cpp:function:: schedule explicit_schedule(const std::vector<units::quantity>& seq);

    Generate events from a predefined sorted event sequence.

.. cpp:function:: schedule explicit_schedule_from_milliseconds(const std::vector<time_type>& seq);

    Generate events from a predefined sorted event sequence given in units of ``[ms]``

.. cpp:function:: schedule poisson_schedule(const units::quantity& tstart, const units::quantity& rate, seed_type seed = default_seed, const units::quantity& tstop=terminal_time*units::ms);

    Poisson point process with rate ``rate``. The underlying Mersenne Twister pRNG is seeded with ``seed``

.. cpp:function:: schedule poisson_schedule(const units::quantity& rate, seed_type seed = default_seed, const units::quantity& tstop=terminal_time*units::ms);

    Poisson point process with rate ``rate``. The underlying Mersenne Twister pRNG is seeded with ``seed``

Event Generators
================

Wrapper class around schedules to generate spikes based on the internal schedule
with a given target and weight.

.. cpp:namespace:: arb

.. cpp:class:: event_generator

    Opaque wrapper around a schedule.

    .. cpp:function:: event_generator(cell_local_label_type target, float weight, schedule sched)

        Create generator targetting the local object ``target``, sending events
        on schedule ``sched`` with weight ``weight``.

    .. cpp:member:: void reset()

        Reset internal event sequence

    .. cpp:member:: event_seq events(time_type t0, time_type t1)

        Return events in ``[t0, t1)``

.. cpp:function:: event_generator empty_generator(cell_local_label_type target, float weight)

.. cpp:function:: event_generator regular_generator(cell_local_label_type target, float weight, const units::quantity& tstart, const units::quantity& dt, const units::quantity& tstop=terminal_time*units::ms)

     Generate events at integer multiples of ``dt`` that lie between ``tstart`` and ``tstop``.

.. cpp:function:: event_generator poisson_generator(cell_local_label_type target, float weight, const units::quantity& tstart, const units::quantity& rate, seed_type seed=default_seed, const units::quantity& tstop=terminal_time*units::ms)

    Poisson point process with rate ``rate``. The underlying Mersenne Twister pRNG is seeded with ``seed``

.. cpp:function:: template<typename S> event_generator explicit_generator(cell_local_label_type target, float weight, const S& s)

    Generate events from a predefined sorted event sequence.

.. cpp:function:: template<typename S> event_generator explicit_generator_from_milliseconds(cell_local_label_type target, float weight, const S& s)

    Generate events from a predefined sorted event sequence given in units of ``[ms]``
