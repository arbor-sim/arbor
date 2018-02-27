Profiler
========

The Arbor library has a built-in profiler for fine-grained timings of regions of interest in the code.
The time stepping code in ``arb::model`` has been instrumented, so by enabling profiler support at compile time, users of the library can generate profile reports from calls to ``arb::model::run()``.

Compilation
-----------

There are some non-trivial overheads associated with using the profiler, so it is not enabled by default.
The profiler can be enabled at compile time, by setting the CMake flag ``ARB_WITH_PROFILING``.
For example to compile a debug build with profiling turned on:

.. code-block:: bash

    cmake .. ARB_WITH_PROFILING=ON

Instrumenting Code
------------------

Developers manually instrument the regions to profile.
This allows the developer to only profile the parts of the code that are of interest, and choose
the appropriate granularity for profiling different regions.

Once a region of code is marked for the profiler, each thread in the application will track the total time spent in the region, and how many times the region is executed on that thread.

Marking Regions
~~~~~~~~~~~~~~~

To instrument a region, use ``PE`` (profiler enter) and ``PL`` (profiler leave) macros to mark the beginning and end of a region.
For example, the following will record the call count and total time spent in the ``foo()`` and ``bar()`` function calls:

.. container:: example-code

    .. code-block:: cpp

        while (t<tfinal) {
            PE(foo);
            foo();
            PL();

            PE(bar);
            bar();
            PL();

            t += dt;
        }

It is not permitted to nest regions if a profiling region is entered while recording.
For example, a ``std::runtime_error`` would be thrown if the call to ``foo()`` in the above example attempted to enter a region:

.. container:: example-code

    .. code-block:: cpp

        void foo() {
            PE(open); // error: already in the "foo" profiling region in the main time loop
            foo_open();
            PL();

            write();
        }

Whenever a profiler region is entered with a call to ``PE``, the time stamp is recorded,
and on the corresponding call to ``PL`` another time stamp is recorded,
and the difference accumulated.
If a region includes time executing other tasks, for example when calling
``arb::threading::parallel_for``, the time spent executing the other tasks will be included, which will give meaningless timings.

Organising Regions
~~~~~~~~~~~~~~~~~~

The profiler allows the user to build a hierarchy of regions by grouping related regions together.

For example, network simulations have two main regions of code to profile: those associated with `communication` and `updating cell state`. These regions each have further subdivisions.
We would like to break these regions down further, e.g. break the `communication` time into time spent performing `spike exchange` and `event binning`.

The subdivision of profiling regions is encoded in the region names.
For example, ``PE(communication_exchange)`` indicates that we are profiling the ``exchange`` sub-region of the top level ``communication`` region.

Below is an example of using sub-regions:

.. container:: example-code

    .. code-block:: cpp

        #include <profiling/profiler.hpp>

        using namespace arb;

        spike_list global_spikes;
        int num_cells = 100;

        void communicate() {
            PE(communication_sortspikes);
            auto local_spikes = get_local_spikes();
            sort(local_spikes);
            PL();

            PE(communication_exchange);
            global_spikes = exchange_spikes(local_spikes);
            PL();
        }

        void update_cell(int i) {
            PE(update_setup);
            setup_events(i);
            PL();

            PE(update_advance_state);
            update_cell_states(i);
            PL();

            PE(update_advance_current);
            update_cell_current(i);
            PL();
        }

        void run(double tfinal, double dt) {
            util::profiler_start();

            double t = 0;
            while (t<tfinal) {
                communicate();
                parallel_for(0, num_cells, update_cell);
                t += dt;
            }

            util::profiler_stop();

            // print profiler results
            util::profiler_print(util::profiler_summary());
        }

The ``communication`` region, is broken into two sub regions: ``exchange`` and ``sortspikes``.
Likewise, ``update`` is broken into ``advance`` and ``setup``, with ``advance``
further broken into ``state`` and ``current``.

Using the information encoded in the region names, the profiler can build a
hierarchical report that shows accumulated time spent in each region and its children:

::

    REGION                     CALLS      THREAD        WALL       %
    root                           -       4.705       2.353    98.0
      update                       -       4.200       2.100    87.5
        advance                    -       4.100       2.050    85.4
          state                 1000       2.800       1.400    58.3
          current               1000       1.300       0.650    27.1
        setup                   1000       0.100       0.050     2.1
      communication                -       0.505       0.253    10.5
        exchange                  10       0.500       0.250    10.4
        sortspikes                10       0.005       0.003     0.1
    WALLTIME      2.400 s

For more information on starting the profiler and interpreting its output see
`Running the Profiler`_ and `Profiler Output`_.

Running the Profiler
--------------------

Before recording time spent in regions, the profiler must first be started, and
then it must be stopped when all profiling is completed.
The time stamp is recorded at each call to stop and start to determine the total
wall time spent in the profiler. This can be used to determine the proportion of
total thread time spent in each region.

.. container:: example-code

    .. code-block:: cpp

        #include <profiling/profiler.hpp>

        using namespace arb;

        void main() {
            util::profiler_start();

            PE(init);
            // ...
            PL();

            PE(simulate);
            // ...
            PL();

            util::profiler_stop();

            // get a profile summary
            util::profile report = util::profiler_summary();

            // print a summary of the profiler to stdout
            util::profiler_profiler_print(report);

            // reset the profiler state.
            util::profiler_reset();
        }

After a call to ``util::profiler_reset``, all counters and timers are set to zero, and the profiler can be started again to collect information about a different part of the application.
This could be used, for example, to generate seperate profiler reports for model building and model executation phases of a simulation.

Profiler Output
~~~~~~~~~~~~~~~

The profiler keeps accumulated call count and time values for each region in each thread.
The ``util::profile`` type, defined in ``src/profiling/profiler.hpp``
On completion of a profiling run, a  c


.. container:: example-code

    .. code-block:: cpp

            // get a profile summary
            util::profile report = util::profiler_summary();

            // print a summary of the profiler to stdout
            util::profiler_profiler_print(report);

Take the example output above:

::

    REGION                     CALLS      THREAD        WALL       %
    root                           -       4.705       2.353    98.0
      update                       -       4.200       2.100    87.5
        advance                    -       4.100       2.050    85.4
          state                 1000       2.800       1.400    58.3
          current               1000       1.300       0.650    27.1
        setup                   1000       0.100       0.050     2.1
      communication                -       0.505       0.253    10.5
        exchange                  10       0.500       0.250    10.4
        sortspikes                10       0.005       0.003     0.1

For each region there are four values reported:

.. table::
    :widths: 10,60

    ====== ======================================================================
    Value  Definition
    ====== ======================================================================
    CALLS  The number of times each region was profiled, summed over all
           threads. Only the call count for the leaf regions is presented.
    THREAD The total accumulated time (in seconds) spent in the region,
           summed over all threads.
    WALL   The thread time divided by the number of threads.
    %      The proportion of the total thread time.
    ====== ======================================================================

The proportion of time spent in the root region is not 100%. This is because there are some parts of the application that are not covered by a region, including:
    * Parts of the library code not between ``PE()`` and ``PL()``;
    * Profiler overheads;
    * Threading runtime overheads;
    * Idle threads waiting at barriers 
