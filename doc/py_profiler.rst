Python Profiler
===============

The Arbor python library has a profiler for fine-grained timings and memory consumptions of regions of interest in the code.

Instrumenting Code
------------------

Developers manually instrument the regions to profile.
This allows the developer to only profile the parts of the code that are of interest, and choose the appropriate granularity for profiling different regions.

Once a region of code is marked for the profiler, the application will track the total time spent in the region, and how much memory (and if available energy) is consumed.

Marking Regions
~~~~~~~~~~~~~~~

For measuring time, memory (and energy) consumption Arbor's meter manager in python can be used.
First the meter manager needs to be initiated, then the metering started and checkpoints set, wherever the manager should report the meters.
The measurement starts from the start to the first checkpoint and then in between checkpoints.
Checkpoints are defined by a string describing the process to be measured.

Running the Profiler
~~~~~~~~~~~~~~~~~~~~~

The profiler does not need to be started or stopped by the user.
It needs to be initialized before entering any profiling region.
It is initialized using the information provided by the execution context.
At any point a summary of profiler region times and consumptions can be obtained.

For example, the following will record and summarize the total time and memory spent:

.. container:: example-code

    .. code-block:: python

        import arbor

        context = arbor.context()
        meter_manager = arbor.meter_manager()
        meter_manager.start(context)

        n_cells = 100
        recipe = my_recipe(n_cells)

        meter_manager.checkpoint('recipe create', context)

        decomp = arbor.partition_load_balance(recipe, context)

        meter_manager.checkpoint('load balance', context)

        sim = arbor.simulation(recipe, decomp, context)

        meter_manager.checkpoint('simulation init', context)

        tSim = 2000
        dt = 0.025
        sim.run(tSim, dt)

        meter_manager.checkpoint('simulation run', context)

        print(arbor.make_meter_report(meter_manager, context))

Profiler Output
------------------

The ``meter_report`` holds a summary of the accumulated recorders.
Calling ``make_meter_report`` will generate a profile summary, which can be printed using ``print``.

Take the example output above:

>>> ---- meters -------------------------------------------------------------------------------
>>> meter                         time(s)      memory(MB)
>>> -------------------------------------------------------------------------------------------
>>> recipe create                   0.000           0.001
>>> load balance                    0.000           0.009
>>> simulation init                 0.005           0.707
>>> simulation run                  3.357           0.028

For each region there are up to three values reported:

.. table::
    :widths: 20,50

    ============= =========================================================================
    Value         Definition
    ============= =========================================================================
    time (s)      The total accumulated time (in seconds) spent in the region.
    memory (MB)   The total memory consumption (in mega bytes) in the region.
    energy (kJ)   The total energy consumption (in kilo joule) in the region (if available).
    ============= =========================================================================

