.. _pyprofiler:

.. currentmodule:: arbor

Metering
========

Arbor's python module :py:mod:`arbor` has a :class:`meter_manager` for measuring time (and if applicable memory) consumptions of regions of interest in the python code.

Users manually instrument the regions to measure.
This allows the user to only measure the parts of the python code that are of interest.
Once a region of code is marked for the :class:`meter_manager`, the application will track the total time (and memory) spent in this region.

Marking Metering Regions
------------------------

First the :class:`meter_manager` needs to be initiated, then the metering started and checkpoints set,
wherever the :class:`meter_manager` should report the meters.
The measurement starts from the :func:`meter_manager.start` to the first :func:`meter_manager.checkpoint` and then in between checkpoints.
Checkpoints are defined by a string describing the process to be measured.

.. class:: meter_manager

    .. function:: meter_manager()

        Construct the meter manager.

    .. function:: start(context)

        Start the metering using the chosen execution :class:`arbor.context`.
        Records a time stamp, that marks the start of the first checkpoint timing region.

    .. function:: checkpoint(name, context)

        Create a new checkpoint ``name`` using the the chosen execution :class:`arbor.context`.
        Records the time since the last checkpoint (or the call to start if no previous checkpoints),
        and restarts the timer for the next checkpoint.

    .. function:: checkpoint_names

        Returns a list of all metering checkpoint names.

    .. function:: times

        Returns a list of all metering times.

At any point a summary of the timing regions can be obtained by the :func:`make_meter_report`.

.. function:: make_meter_report(meter_manager, context)

    Generate a meter report based on the :class:`meter_manager` and chosen execution :class:`arbor.context`.

For instance, the following python code will record and summarize the total time (and memory) spent:

.. container:: example-code

    .. code-block:: python

        import arbor

        context = arbor.context(threads=8, gpu_id=None)
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

Metering Output
------------------

Calling :func:`make_meter_report` will generate a profile summary, which can be printed using ``print``.
Take the example output from above:

>>> <arbor.meter_report>:
>>> ---- meters -------------------------------------------------------------------------------
>>> meter                         time(s)      memory(MB)
>>> -------------------------------------------------------------------------------------------
>>> recipe create                   0.000           0.001
>>> load balance                    0.000           0.009
>>> simulation init                 0.026           3.604
>>> simulation run                  4.171           0.021
>>> meter-total                     4.198           3.634
