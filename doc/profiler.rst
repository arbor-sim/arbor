Profiler
========

The Arbor library has a built-in profiler for fine-grained timings of regions of interest in the code.
The time stepping code in ``arb::simulation`` has been instrumented, so by enabling profiler support at
compile time, users of the library can generate profile reports from calls to ``arb::simulation::run()``.

Compilation
-----------

There are some non-trivial overheads associated with using the profiler, so it is not enabled by default.
The profiler can be enabled at compile time, by setting the CMake flag ``ARB_WITH_PROFILING``.
For example to compile a debug build with profiling turned on:

.. code-block:: bash

    cmake .. -DARB_WITH_PROFILING=ON

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
            double t = 0;
            while (t<tfinal) {
                communicate();
                parallel_for(0, num_cells, update_cell);
                t += dt;
            }

            // print profiler results
            std::cout << util::profiler_summary() << "\n";
        }

The ``communication`` region, is broken into two sub regions: ``exchange`` and ``sortspikes``.
Likewise, ``update`` is broken into ``advance`` and ``setup``, with ``advance``
further broken into ``state`` and ``current``.

Using the information encoded in the region names, the profiler can build a
hierarchical report that shows accumulated time spent in each region and its children:

::

    _p_ REGION                     CALLS      THREAD        WALL       %
    _p_ root                           -       4.705       2.353   100.0
    _p_   update                       -       4.200       2.100    89.3
    _p_     advance                    -       4.100       2.050    87.1
    _p_       state                 1000       2.800       1.400    59.5
    _p_       current               1000       1.300       0.650    27.6
    _p_     setup                   1000       0.100       0.050     2.1
    _p_   communication                -       0.505       0.253    10.7
    _p_     exchange                  10       0.500       0.250    10.6
    _p_     sortspikes                10       0.005       0.003     0.1

For _p_ more information on interpreting the profiler's output see
`Running the Profiler`_ and `Profiler Output`_.

Running the Profiler
--------------------

The profiler does not need to be started or stopped by the user.
It needs to be initialized before entering any profiling region.
It is initialized using the information provided by the simulation's thread pool.
At any point a summary of profiler region counts and times can be obtained,
and the profiler regions can be reset.

.. container:: example-code

    .. code-block:: cpp

        #include <profiling/profiler.hpp>

        using namespace arb;

        void main() {
            execution_context context;

            // Initialize the profiler with thread information from the execution context
            profile::profiler_initialize(context.thread_pool);

            PE(init);
            // ...
            PL();

            PE(simulate);
            // ...
            PL();

            // Print a summary of the profiler to stdout
            std::cout << profile::profiler_summary() << "\n";

            // Clear the profiler state, which can then be used to record
            // profile information for a different part of the code.
            profile::profiler_clear();
        }

After a call to ``util::profiler_clear``, all counters and timers are set to zero.
This could be used, for example, to generate separate profiler reports for model building and model execution phases.

Profiler Output
~~~~~~~~~~~~~~~

The profiler keeps accumulated call count and time values for each region in each thread.
The ``util::profile`` type, defined in ``src/profiling/profiler.hpp`` holds a summary of
the accumulated recorders. Calling ``util::profiler_summary()`` will generate a profile
summary, which can be printed using the ``operator<<`` for ``std::ostream``.

.. container:: example-code

    .. code-block:: cpp

            // get a profile summary
            util::profile report = util::profiler_summary();

            // print a summary of the profiler to stdout
            std::cout << report << "\n";

Take the example output above:

::

    _p_ REGION                     CALLS      THREAD        WALL       %
    _p_ root                           -       5.379       1.345   100.0
    _p_   advance                      -       5.368       1.342    99.8
    _p_     integrate                  -       5.367       1.342    99.8
    _p_       current              26046       3.208       0.802    59.6
    _p_       state                26046       1.200       0.300    22.3
    _p_       matrix                   -       0.808       0.202    15.0
    _p_         solve              26046       0.511       0.128     9.5
    _p_         build              26046       0.298       0.074     5.5
    _p_       events               78138       0.123       0.031     2.3
    _p_       ionupdate            26046       0.013       0.003     0.2
    _p_       samples              26046       0.007       0.002     0.1
    _p_       threshold            26046       0.005       0.001     0.1
    _p_   communication                -       0.012       0.003     0.2
    _p_     enqueue                    -       0.011       0.003     0.2
    _p_       sort                    88       0.011       0.003     0.2

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
    %      The proportion of the total thread time spent in the region
    ====== ======================================================================

