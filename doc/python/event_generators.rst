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

.. currentmodule:: arbor

.. py:class:: event_generator

    Opaque wrapper around a schedule.

    .. py:function:: event_generator(target, weight, sched)

        Create generator targetting the local object ``target``, sending events
        on schedule ``sched`` with weight ``weight``.
