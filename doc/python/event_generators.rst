.. _pygenerators:

Schedules
=========

Generate sorted time points.

.. currentmodule:: arbor

.. py:class:: schedule

    Opaque representation of a schedule.

.. py:function:: regular_schedule(t0, dt, t1 = None);

    Regular schedule with start ``t0``, interval ``dt``, and optional end ``t1``.

.. py:function:: regular_schedule(dt);

   Regular schedule with interval ``dt``.

.. py:function:: explicit_schedule(seq);

    Generate events from a predefined sorted event sequence.

.. py:function:: explicit_schedule_from_milliseconds(seq);

    Generate events from a predefined sorted event sequence given in units of ``[ms]``

.. py:function:: poisson_schedule(tstart, rate, seed=None, tstop=None);

    Poisson point process with rate ``rate``. The underlying Mersenne Twister pRNG is seeded with ``seed``

.. py:function:: schedule poisson_schedule(rate, seed=None, tstop=None);

    Poisson point process with rate ``rate``. The underlying Mersenne Twister pRNG is seeded with ``seed``

Event Generators
================

Wrapper class around schedules to generate spikes based on the internal schedule
with a given target and weight.

.. class:: event_generator

    .. function:: __init___(target, weight, schedule)

        Construct an event generator for a :attr:`target` synapse with :attr:`weight` of the events to
        deliver based on a schedule (i.e., :class:`arbor.regular_schedule`, :class:`arbor.explicit_schedule`,
        :class:`arbor.poisson_schedule`).

    .. attribute:: target

        The target synapse of type :class:`arbor.cell_local_label`.

    .. attribute:: weight

        The weight delivered to the target synapse. It is up to the target mechanism to interpret this quantity.
        For Arbor-supplied point processes, such as the ``expsyn`` synapse, a weight of ``1`` corresponds to an
        increase in conductivity in the target mechanism of ``1`` μS (micro-Siemens).

.. class:: regular_schedule

    Describes a regular schedule with multiples of :attr:`dt` within the interval [:attr:`tstart`, :attr:`tstop`).

    .. function:: regular_schedule(tstart, dt, tstop)

        Construct a regular schedule as list of times from :attr:`tstart` to :attr:`tstop` in :attr:`dt` time steps.

        By default returns a schedule with :attr:`tstart` = :attr:`tstop` = ``None`` and :attr:`dt` = 0 ms.

    .. attribute:: tstart

        The delivery time of the first event in the sequence [ms].
        Must be non-negative or ``None``.

    .. attribute:: dt

        The interval between time points [ms].
        Must be non-negative.

    .. attribute:: tstop

        No events will be delivered after this time [ms].
        Must be non-negative or ``None``.

    .. function:: events(t0, t1)

        Returns a view of monotonically increasing time values in the half-open interval [t0, t1).

.. class:: explicit_schedule

    Describes an explicit schedule at a predetermined (sorted) sequence of :attr:`times`.

    .. function:: explicit_schedule(times)

        Construct an explicit schedule.

        By default returns a schedule with an empty list of times.

    .. attribute:: times

        The list of non-negative times [ms].

    .. function:: events(t0, t1)

        Returns a view of monotonically increasing time values in the half-open interval [t0, t1).

.. class:: poisson_schedule

    Describes a schedule according to a Poisson process.

    .. function:: poisson_schedule(tstart, freq, seed)

        Construct a Poisson schedule.

        By default returns a schedule with events starting from :attr:`tstart` = 0 ms,
        with an expected frequency :attr:`freq` = 10 kHz and :attr:`seed` = 0.

    .. attribute:: tstart

        The delivery time of the first event in the sequence [ms].

    .. attribute:: freq

        The expected frequency [kHz].

    .. attribute:: seed

        The seed for the random number generator.

    .. function:: events(t0, t1)

        Returns a view of monotonically increasing time values in the half-open interval [t0, t1).

    .. attribute:: tstop

        No events will be delivered after this time [ms].

An example of an event generator reads as follows:

.. container:: example-code

    .. code-block:: python

        import arbor as A

        # define a Poisson schedule with a start time at 1 ms, expected frequency of 5 Hz,
        # and the target cell's gid as seed

        def event_generators(gid):
            # label of the synapse on target cell, if multiple distribute cyclically
            target = A.cell_local_label("syn", A.selection_policy.round_robin)
            # Poisson schedule with a start time, event rate, and the target cell's gid as seed.
            sched  = A.poisson_schedule(tstart=1*U.ms, freq=5*U.Hz, seed=gid)
            # Weight to apply to the events
            weight = 0.1
            return [A.event_generator(target, weight, sched)]        
