.. _pybenchcell:

Benchmark cells
===============

.. currentmodule:: arbor

.. py:class:: benchmark_cell

    A benchmarking cell, used by Arbor developers to test communication performance.

    .. function:: benchmark_cell(source, target, schedule, realtime_ratio)

        Construct a benchmark cell with a single built-in source with label ``source``; and a
        single built-in target with label ``target``. The labels can be used for forming connections from/to
        the cell in the :py:class:`arbor.recipe` by creating a :py:class:`arbor.connection`.

        A benchmark cell generates spikes at a user-defined sequence of time points:

        - at regular intervals (using an :class:`arbor.regular_schedule`)
        - at a sequence of user-defined times (using an :class:`arbor.explicit_schedule`)
        - at times defined by a Poisson sequence (using an :class:`arbor.poisson_schedule`)

        and the time taken to integrate a cell can be tuned by setting the parameter ``realtime_ratio``.

        :param source: label of the source on the cell.

        :param target: label of the target on the cell.

        :param schedule: User-defined sequence of time points (choose from :class:`arbor.regular_schedule`, :class:`arbor.explicit_schedule`, or :class:`arbor.poisson_schedule`).

        :param realtime_ratio: Time taken to integrate a cell, for example if ``realtime_ratio`` = 2, a cell will take 2 seconds of CPU time to simulate 1 second.
