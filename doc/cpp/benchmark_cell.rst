.. _cppbenchcell:

Benchmark cells
===============

.. cpp:namespace:: arb

.. cpp:class:: lif_cell

    A benchmarking cell, used by Arbor developers to test communication performance.

    .. cpp:function:: benchmark_cell(const cell_tag_type& source, const cell_tag_type& target, schedule, double realtime_ratio)

        Construct a benchmark cell with a single built-in source with label
        ``source``; and a single built-in target with label ``target``. The
        labels can be used for forming connections from/to the cell in the
        :cpp:class:`arb::recipe` by creating a :cpp:class:`arb::connection`.

        A benchmark cell generates spikes at a user-defined sequence of time points:

        - at regular intervals (using an :cpp:class:`arb::regular_schedule`)
        - at a sequence of user-defined times (using an :cpp:class:`arb::explicit_schedule`)
        - at times defined by a Poisson sequence (using an :cpp:class:`arb::poisson_schedule`)

        and the time taken to integrate a cell can be tuned by setting the parameter ``realtime_ratio``.

    .. cpp:member:: cell_tag_type source

        Label of the source on the cell.

    .. cpp:member:: cell_tag_type target

        Label of the target on the cell.

    .. cpp:member:: schedule time_sequence

        User-defined sequence of time points, e.g.
        :cpp:class:`arb::regular_schedule`,
        :cpp:class:`arb::explicit_schedule`, or
        :cpp:class:`arb::poisson_schedule`.

    .. cpp:member:: double ratio

        Time taken to integrate a cell; for example, if ``realtime_ratio`` =
        2, a cell will take 2 seconds of CPU time to simulate 1 second.

.. cpp:type:: benchmakr_cell_editor = std::function<void(benchmark_cell&)>

    Callback function to update setting of LIF cells in place via the
    ``simulation::edit_cell`` interace. All values are changeable, except:
    ``source`` and ``target``.
