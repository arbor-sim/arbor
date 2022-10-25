.. _cpphardware:

Hardware context
================

Arbor provides two library APIs for working with hardware resources:

* The core *libarbor* is used to *describe* the hardware resources
  and their contexts for use in Arbor simulations.
* The *libarborenv* provides an API for querying available hardware
  resources (e.g. the number of available GPUs), and initializing MPI.


libarborenv
-------------------

The *libarborenv* API for querying and managing hardware resources is in the
:cpp:any:`arbenv` namespace.
This functionality is kept in a separate library to enforce
separation of concerns, so that users have full control over how hardware resources
are selected, either using the functions and types in *libarborenv*, or writing their
own code for managing MPI, GPUs, and thread counts.

Functions for determining environment defaults based on system information and
user-supplied values in environment values are in the header ``arborenv/default_env.hpp``.

.. cpp:namespace:: arbenv

.. cpp:function:: unsigned long get_env_num_threads()

    Retrieve user-specified number of threads to use from the environment variable
    ARBENV_NUM_THREADS.

    Return value:

    * Returns zero if ARBENV_NUM_THREADS is unset or empty.
    * Returns positive unsigned long value on ARBENV_NUM_THREADS if set.

    Throws:

    * Throws :cpp:any:`arbenv::invalid_env_value` if ARBENV_NUM_THREADS is set, non-empty, and not a valid representation of a positive unsigned long value.

    .. container:: example-code

       .. code-block:: cpp

         #include <arborenv/concurrency.hpp>

         if (auto nt = arbenv::get_env_num_threads()) {
            std::cout << "requested " << nt.value() << "threads \n";
         }
         else {
            std::cout << "environment variable empty or unset\n";
         }

.. cpp:function:: arb::proc_allocation default_allocation()

   Return a :cpp:any:`proc_allocation` with thread count from :cpp:any:`default_concurrency()`
   and gpu id from :cpp:any:`default_gpu()`.

.. cpp:function:: unsigned long default_concurrency()

    Returns number of threads to use from :cpp:any:`get_env_num_threads()`, or else from
    :cpp:any:`thread_concurrency()` if :cpp:any:`get_env_num_threads()` returns zero.

.. cpp:function:: int default_gpu()

   Determine GPU id to use from the ARBENV_GPU_ID environment variable, or from the first available
   GPU id of those detected.

   Return value:

   * Return -1 if Arbor has no GPU support, or if the ARBENV_GPU_ID environment variable is set to a negative number, or if ARBENV_GPU_ID is empty or unset and no GPUs are detected.
   * Return a non-negative GPU id equal to ARBENV_GPU_ID if it is set to a non-negative value that is a valid GPU id, or else to the first valid GPU id detected (typically zero).

   Throws:

   * Throws :cpp:any:`arbenv::invalid_env_value` if ARBENV_GPU_ID contains a non-integer value.
   * Throws :cpp:any:`arbenv::no_such_gpu` if ARBENV_GPU_ID contains a non-negative integer that does not correspond to a detected GPU.

The header ``arborenv/concurrency.hpp`` supplies lower-level functions for querying the threading environment.

.. cpp:function:: unsigned long thread_concurrency()

   Attempts to detect the number of available CPU cores. Returns 1 if unable to detect
   the number of cores.

.. cpp:function:: std::vector<int> get_affinity()

   Returns the list of logical processor ids where the calling thread has affinity,
   or an empty vector if unable to determine.

The header ``arborenv/gpu_env.hpp`` supplies lower-level functions for querying the GPU environment.

.. cpp:function:: int find_private_gpu(MPI_Comm comm)

   A helper function that assigns a unique GPU to every MPI rank.

   .. Note::

      Arbor allows at most one GPU per MPI rank, and furthermore requires that
      an MPI rank has exclusive access to a GPU, i.e. two MPI ranks can not
      share a GPU.
      This function assigns a unique GPU to each rank when more than one rank
      has access to the same GPU(s).
      An example use case is on systems with "fat" nodes with multiple GPUs
      per node, in which case Arbor should be run with multiple MPI ranks
      per node.
      Uniquely assigning GPUs is quite difficult, and this function provides
      what we feel is a robust implementation.

   All MPI ranks in the MPI communicator :cpp:any:`comm` should call to
   avoid a deadlock.

   Return value:

     * **non-negative integer**: the identifier of the GPU assigned to this rank.
     * **-1**: no GPU was available for this MPI rank.

   Throws:

     * :cpp:any:`arbenv::gpu_uuid_error`: if there was an error in the CUDA runtime
       on the local or remote MPI ranks, i.e. if one rank throws, all ranks
       will throw.

The header ``arborenv/with_mpi.hpp`` provides an RAII interface for initializing MPI
and handling exceptions on MPI exit.

.. cpp:class:: with_mpi

   The :cpp:class:`with_mpi` type is a simple RAII scoped guard for MPI initialization
   and finalization. On creation :cpp:class:`with_mpi` will call :cpp:any:`MPI_Init_thread`
   to initialize MPI with the minimum level thread support required by Arbor, that is
   ``MPI_THREAD_SERIALIZED``. When it goes out of scope it will automatically call
   :cpp:any:`MPI_Finalize`.

   .. cpp:function:: with_mpi(int& argcp, char**& argvp, bool fatal_errors = true)

      The constructor takes the :cpp:any:`argc` and :cpp:any:`argv` arguments
      passed to main of the calling application, and an additional flag
      :cpp:any:`fatal_errors` that toggles whether errors in MPI API calls
      should return error codes or terminate.

   .. Warning::

      Handling exceptions is difficult in MPI applications, and it is the users
      responsibility to do so.

      The :cpp:class:`with_mpi` scope guard attempts to facilitate error reporting of
      uncaught exceptions, particularly in the case where one rank throws an exception,
      while the other ranks continue executing. In this case there would be a deadlock
      if the rank with the exception attempts to call :cpp:any:`MPI_Finalize` and
      other ranks are waiting in other MPI calls. If this happens inside a try-catch
      block, the deadlock stops the exception from being handled.
      For this reason the destructor of :cpp:class:`with_mpi` only calls
      :cpp:any:`MPI_Finalize` if there are no uncaught exceptions.
      This isn't perfect because the other MPI ranks still deadlock,
      however it gives the exception handling code to report the error for debugging.

   An example workflow that uses the MPI scope guard. Note that this code will
   print the exception error message in the case where only one MPI rank threw
   an exception, though it would either then deadlock or exit with an error code
   that one or more MPI ranks exited without calling :cpp:any:`MPI_Finalize`.

    .. container:: example-code

        .. code-block:: cpp

            #include <exception>
            #include <iostream>

            #include <arborenv/with_mpi.hpp>

            int main(int argc, char** argv) {
                try {
                    // Constructing guard will initialize MPI with a
                    // call to MPI_Init_thread()
                    arbenv::with_mpi guard(argc, argv, false);

                    // Do some work with MPI here

                    // When leaving this scope, the destructor of guard will
                    // call MPI_Finalize()
                }
                catch (std::exception& e) {
                    std::cerr << "error: " << e.what() << "\n";
                    return 1;
                }
                return 0;
            }

Functions and methods in the ``arborenv`` library may throw exceptions specific to the library.
These are declared in the ``arborenv/arbenvexcept.hpp`` header, and all derive from the
class ``arborenv::arborenv_exception``, itself derived from ``std::runtime_error``.

libarbor
-------------------

The core Arbor library *libarbor* provides an API for:

  * prescribing which hardware resources are to be used by a
    simulation using :cpp:class:`arb::proc_allocation`.
  * opaque handles to hardware resources used by simulations called
    :cpp:class:`arb::context`.

.. cpp:namespace:: arb

.. cpp:class:: proc_allocation

    Enumerates the computational resources on a node to be used for simulation,
    specifically the number of threads and identifier of a GPU if available.

    .. Note::

       Each MPI rank in a distributed simulation uses a :cpp:class:`proc_allocation`
       to describe the subset of resources on its node that it will use.

    .. container:: example-code

        .. code-block:: cpp

            #include <arbor/context.hpp>

            // default: 1 thread and no GPU selected
            arb::proc_allocation resources;

            // 8 threads and no GPU
            arb::proc_allocation resources(8, -1);

            // 4 threads and the first available GPU
            arb::proc_allocation resources(8, 0);

            // Construct with
            auto num_threads = arbenv::thread_concurrency();
            auto gpu_id = arbenv::default_gpu();
            arb::proc_allocation resources(num_threads, gpu_id);


    .. cpp:function:: proc_allocation() = default

        By default selects one thread and no GPU.

    .. cpp:function:: proc_allocation(unsigned threads, int gpu_id)

        Constructor that sets the number of :cpp:var:`threads` and the id :cpp:var:`gpu_id` of
        the available GPU.

    .. cpp:member:: unsigned num_threads

        The number of CPU threads available.

    .. cpp:member:: int gpu_id

        The identifier of the GPU to use.
        The gpu id corresponds to the ``int device`` parameter used by CUDA API calls
        to identify gpu devices.
        Set to -1 to indicate that no GPU device is to be used.
        See ``cudaSetDevice`` and ``cudaDeviceGetAttribute`` provided by the
        `CUDA API <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html>`_.

    .. cpp:function:: bool has_gpu() const

        Indicates whether a GPU is selected (i.e. whether :cpp:member:`gpu_id` is ``-1``).

.. cpp:namespace:: arb

.. cpp:class:: context

    An opaque handle for the hardware resources used in a simulation.
    A :cpp:class:`context` contains a thread pool, and optionally the GPU state
    and MPI communicator. Users of the library do not directly use the functionality
    provided by :cpp:class:`context`, instead they create contexts, which are passed to
    Arbor interfaces for domain decomposition and simulation.

Arbor contexts are created by calling :cpp:func:`make_context`, which returns an initialized
context. There are two versions of :cpp:func:`make_context`, for creating contexts
with and without distributed computation with MPI respectively.

.. cpp:function:: context make_context(proc_allocation alloc=proc_allocation())

    Create a local :cpp:class:`context`, with no distributed/MPI,
    that uses local resources described by :cpp:any:`alloc`.
    By default it will create a context with one thread and no GPU.

.. cpp:function:: context make_context(proc_allocation alloc, MPI_Comm comm)

    Create a distributed :cpp:class:`context`.
    A context that uses the local resources described by :cpp:any:`alloc`, and
    uses the MPI communicator :cpp:var:`comm` for distributed calculation.

Contexts can be queried for information about which features a context has enabled,
whether it has a GPU, how many threads are in its thread pool, using helper functions.

.. cpp:function:: bool has_gpu(const context&)

   Query whether the context has a GPU.

.. cpp:function:: unsigned num_threads(const context&)

   Query the number of threads in a context's thread pool.

.. cpp:function:: bool has_mpi(const context&)

   Query whether the context uses MPI for distributed communication.

.. cpp:function:: unsigned num_ranks(const context&)

   Query the number of distributed ranks. If the context has an MPI
   communicator, return is equivalent to :cpp:any:`MPI_Comm_size`.
   If the communicator has no MPI, returns 1.

.. cpp:function:: unsigned rank(const context&)

   Query the rank of the calling rank. If the context has an MPI
   communicator, return is equivalent to :cpp:any:`MPI_Comm_rank`.
   If the communicator has no MPI, returns 0.

Here are some simple examples of how to create a :cpp:class:`arb::context` using
:cpp:func:`make_context`.

.. container:: example-code

  .. code-block:: cpp

      #include <arbor/context.hpp>

      // Construct a context that uses 1 thread and no GPU or MPI.
      auto context = arb::make_context();

      // Construct a context that:
      //  * uses 8 threads in its thread pool;
      //  * does not use a GPU, regardless of whether one is available;
      //  * does not use MPI.
      arb::proc_allocation resources(8, -1);
      auto context = arb::make_context(resources);

      // Construct one that uses:
      //  * 4 threads and the first GPU;
      //  * MPI_COMM_WORLD for distributed computation.
      arb::proc_allocation resources(4, 0);
      auto mpi_context = arb::make_context(resources, MPI_COMM_WORLD)

Here is a more complicated example of creating a :cpp:class:`context` on a
system where support for GPU and MPI support are conditional.

.. container:: example-code

  .. code-block:: cpp

      #include <arbor/context.hpp>
      #include <arbor/version.hpp>   // for ARB_MPI_ENABLED

      #include <arborenv/concurrency.hpp>
      #include <arborenv/gpu_env.hpp>

      int main(int argc, char** argv) {
          try {
              arb::proc_allocation resources;

              // try to detect how many threads can be run on this system
              resources.num_threads = arbenv::thread_concurrency();

              // override thread count if the user set ARB_NUM_THREADS
              if (auto nt = arbenv::get_env_num_threads()) {
                  resources.num_threads = nt;
              }

      #ifdef ARB_WITH_MPI
              // initialize MPI
              arbenv::with_mpi guard(argc, argv, false);

              // assign a unique gpu to this rank if available
              resources.gpu_id = arbenv::find_private_gpu(MPI_COMM_WORLD);

              // create a distributed context
              auto context = arb::make_context(resources, MPI_COMM_WORLD);
              root = arb::rank(context) == 0;
      #else
              resources.gpu_id = arbenv::default_gpu();

              // create a local context
              auto context = arb::make_context(resources);
      #endif

              // Print a banner with information about hardware configuration
              std::cout << "gpu:      " << (has_gpu(context)? "yes": "no") << "\n";
              std::cout << "threads:  " << num_threads(context) << "\n";
              std::cout << "mpi:      " << (has_mpi(context)? "yes": "no") << "\n";
              std::cout << "ranks:    " << num_ranks(context) << "\n" << std::endl;

              // run some simulations!
          }
          catch (std::exception& e) {
              std::cerr << "exception caught in ring miniapp: " << e.what() << "\n";
              return 1;
          }

          return 0;
      }


.. _cppdistcontext:

Distributed context
-------------------

To support running on systems from laptops and workstations to large distributed
HPC clusters, Arbor uses  *distributed contexts* to:

    * Describe the distributed computer system that a simulation is to be
      distributed over and run on.
    * Perform collective operations over the distributed system, such as gather
      and synchronization.
    * Query information about the distributed system, such as the number of
      distributed processes and the index/rank of the calling process.

The global context used to run a simulation is determined at run time, not at compile time.
This means that if Arbor is compiled with support for MPI enabled, then at run time the
user can choose between using a non-distributed (local) context, or an distributed MPI
context.

An execution context is created by a user before building and running a simulation.
This context is then used to perform domain decomposition and initialize the simulation
(see :ref:`cppsimulation` for more about the simulation building workflow).
In the example below, a context that uses MPI is used to run a distributed simulation:

The public API does not directly expose :cpp:class:`arb::distributed_context` or any of its
implementations.
By default :cpp:class:`arb::context` uses only local "on-node" resources. To use an MPI
communicator for distributed communication, it can be initialised with the communicator:

.. container:: example-code

    .. code-block:: cpp

        arb::proc_allocation resources;
        my_recipe recipe;

        // Create a context that uses the local resources enumerated in resources,
        // and that uses the standard MPI communicator MPI_COMM_WORLD for
        // distributed communication.
        arb::context context = arb::make_context(resources, MPI_COMM_WORLD);

        // Partition model over the distributed system.
        arb::domain_decomposition decomp = arb::partition_load_balance(recipe, context);

        // Instantiate the simulation over the distributed system.
        arb::simulation sim(recipe, decomp, context);

        // Run the simulation for 100ms over the distributed system.
        sim.run(100, 0.01);

In the back end :cpp:class:`arb::distributed_context` defines the interface for distributed contexts,
for which two implementations are provided: :cpp:class:`arb::local_context` and :cpp:class:`arb::mpi_context`.
Distributed contexts are wrapped in shared pointers:

.. cpp:type:: distributed_context_handle = std::shared_ptr<distributed_context>

A distributed context can then be generated using helper functions :cpp:func:`arb::make_local_context` and
:cpp:func:`arb::make_mpi_context`:

.. container:: example-code

    .. code-block:: cpp

        // Create a context that uses only local resources (is non-distributed).
        auto dist_ctx  arb::make_local_context();

        // Create an MPI context that uses the standard MPI_COMM_WORLD communicator.
        auto dist_ctx = arb::make_mpi_context(MPI_COMM_WORLD);


Class documentation
^^^^^^^^^^^^^^^^^^^

.. cpp:namespace:: arb

.. cpp:class:: distributed_context

    Defines the interface used by Arbor to query and perform collective
    operations on distributed systems.

    Uses value-semantic type erasure. The main benefit of this approach is that
    classes that implement the interface can use duck typing instead of
    deriving from :cpp:class:`distributed_context`.

    **Constructor:**

    .. cpp:function:: distributed_context()

        Default constructor initializes the context as a :cpp:class:`local_context`.

    .. cpp:function:: distributed_context(distributed_context&& other)

        Move constructor.

    .. cpp:function:: distributed_context& operator=(distributed_context&& other)

        Copy from rvalue.

    .. cpp:function:: template <typename Impl> distributed_context(Impl&& impl)

        Initialize with an implementation that satisfies the interface.

    **Interface:**

    .. cpp:function:: int id() const

        Each distributed process has a unique integer identifier, where the identifiers
        are numbered contiguously in the half open range [0, size).
        (for example ``MPI_Rank``).

    .. cpp:function:: int size() const

        The number of distributed processes (for example ``MPI_Size``).

    .. cpp:function:: void barrier() const

        A synchronization barrier where all distributed processes wait until every
        process has reached the barrier (for example ``MPI_Barrier``).

    .. cpp:function:: std::string name() const

        The name of the context implementation. For example, if using MPI returns ``"MPI"``.

    .. cpp:function:: std::vector<std::string> gather(std::string value, int root) const

        Overload for gathering a string from each domain into a vector
        of strings on domain :cpp:any:`root`.

    .. cpp:function:: T min(T value) const

        Reduction operation over all processes.

        The type ``T`` is one of ``float``, ``double``, ``int``,
        ``std::uint32_t``, ``std::uint64_t``.

    .. cpp:function:: T max(T value) const

        Reduction operation over all processes.

        The type ``T`` is one of ``float``, ``double``, ``int``,
        ``std::uint32_t``, ``std::uint64_t``.

    .. cpp:function:: T sum(T value) const

        Reduction operation over all processes.

        The type ``T`` is one of ``float``, ``double``, ``int``,
        ``std::uint32_t``, ``std::uint64_t``.

    .. cpp:function:: std::vector<T> gather(T value, int root) const

        Gather operation. Returns a vector with one entry for each process.

        The type ``T`` is one of ``float``, ``double``, ``int``,
        ``std::uint32_t``, ``std::uint64_t``, ``std::string``.

.. cpp:class:: local_context

    Implements the :cpp:class:`arb::distributed_context` interface for
    non-distributed computation.

    This is the default :cpp:class:`arb::distributed_context`, and should be used
    when running on laptop or workstation systems with one NUMA domain.

    .. Note::
        :cpp:class:`arb::local_context` provides the simplest possible distributed context,
        with only one process, and where all reduction operations are the identity operator.

    **Constructor:**

    .. cpp:function:: local_context()

        Default constructor.

.. cpp:function:: distributed_context_handle make_local_context()

    Convenience function that returns a handle to a local context.

.. cpp:class:: mpi_context

    Implements the :cpp:class:`arb::distributed_context` interface for
    distributed computation using the MPI message passing library.

    **Constructor:**

    .. cpp:function:: mpi_context(MPI_Comm comm)

        Create a context that will uses the MPI communicator :cpp:any:`comm`.

.. cpp:function:: distributed_context_handle make_mpi_context(MPI_Comm comm)

    Convenience function that returns a handle to a :cpp:class:`arb::mpi_context`
    that uses the MPI communicator comm.




.. _cppdryrun:

.. Note::
    This is a developer feature for benchmarking, and is not useful for scientific use cases.

Dry-run mode
------------

Dry-run mode is used to mimic the performance of running an MPI distributed simulation
without having access to an HPC cluster or even MPI support. It is verifiable against an MPI
run with the same parameters. In dry-run mode, we describe the model on a single domain and
translate it to however many domains we want to mimic. This allows us to know the exact
behavior of the entire system by only running the simulation on a single node.
To support dry-run mode we use the following classes:

.. cpp:namespace:: arb

.. cpp:class:: dry_run_context

    Implements the :cpp:class:`arb::distributed_context` interface for a fake distributed
    simulation.

    .. cpp:member:: unsigned num_ranks_

        Number of domains we are mimicking.

    .. cpp:member:: unsigned num_cells_per_tile_

        Number of cells assigned to each domain.


    **Constructor:**

    .. cpp:function:: dry_run_context_impl(unsigned num_ranks, unsigned num_cells_per_tile)

        Creates the dry run context and sets up the information needed to fake communication
        between domains.

    **Interface:**

    .. cpp:function:: int id() const

        Always 0. We are only performing the simulation on the local domain which will be root.

    .. cpp:function:: int size() const

        Equal to :cpp:member:`num_ranks_`.

    .. cpp:function:: std::string name() const

        Returns ``"dry_run"``.

    .. cpp:function:: std::vector<std::string> gather(std::string value, int root) const

        Duplicates the vector of strings from local domain, :cpp:member:`num_ranks_` times.
        Returns the concatenated vector.

    .. cpp:function:: gathered_vector<arb::spike>  gather_spikes(const std::vector<arb::spike>& local_spikes) const

        The vector of :cpp:any:`local_spikes` represents the spikes obtained from running a
        simulation of :cpp:member:`num_cells_per_tile_` on the local domain.
        The returned vector should contain the spikes obtained from all domains in the dry-run.
        The spikes from the non-simulated domains are obtained by copying :cpp:any:`local_spikes`
        and modifying the gids of each spike to refer to the corresponding gids on each domain.
        The obtained vectors of spikes from each domain are concatenated along with the original
        :cpp:any:`local_spikes` and returned.

    .. cpp:function:: distributed_context_handle make_dry_run_context(unsigned num_ranks, unsigned num_cells_per_tile)

        Convenience function that returns a handle to a :cpp:class:`dry_run_context`.

.. cpp:class:: tile: public recipe

    .. Note::
        While this class inherits from :cpp:class:`arb::recipe`, it breaks one of its implicit
        rules: it allows connection from gids greater than the total number of cells in a recipe,
        :cpp:any:`ncells`.

    :cpp:class:`arb::tile` describes the model on a single domain containing :cpp:expr:`num_cells =
    num_cells_per_tile` cells, which is to be duplicated over :cpp:any:`num_ranks`
    domains in dry-run mode. It contains information about :cpp:any:`num_ranks` which is provided
    by the following function:

    .. cpp:function:: cell_size_type num_tiles() const

    Most of the overloaded functions in :cpp:class:`arb::tile` describe a recipe on the local
    domain, as if it was the only domain in the simulation, except for the following two
    functions that accept :cpp:any:`gid` arguments in the half open interval
    ``[0, num_cells*num_tiles)``:

    .. cpp:function:: std::vector<cell_connection> connections_on(cell_gid_type gid) const

    .. cpp:function:: std::vector<event_generator> event_generators(cell_gid_type gid) const

.. cpp:class:: symmetric_recipe: public recipe

    A symmetric_recipe mimics having a model containing :cpp:var:`num_tiles()`
    instances of :cpp:class:`arb::tile` in a simulation of one tile per domain.

    .. cpp:member:: std::unique_ptr<tile> tiled_recipe_

        `symmetric_recipe` owns a unique pointer to a :cpp:class:`arb::tile`, and uses
        :cpp:member:`tiled_recipe_` to query information about the tiles on the local
        and mimicked domains.

        Most functions in `symmetric_recipe` only need to call the underlying functions
        of `tiled_recipe_` for the corresponding gid in the simulated domain. This is
        done with a simple modulo operation. For example:

        .. code-block:: cpp

            cell_kind get_cell_kind(cell_gid_type i) const override {
                return tiled_recipe_->get_cell_kind(i % tiled_recipe_->num_cells());
            }

    The exception is again the following 2 functions:

    .. cpp:function:: std::vector<cell_connection> connections_on(cell_gid_type i) const

        Calls

        .. code-block:: cpp

            tiled_recipe_.connections_on(i % tiled_recipe_->num_cells())

        But the obtained connections have to be translated to refer to the correct
        gids corresponding to the correct domain.

    .. cpp:function:: std::vector<event_generator> event_generators(cell_gid_type i) const

        Calls

        .. code-block:: cpp

            tiled_recipe_.event_generators(i)

        Calls on the domain gid without the modulo operation, because the function has a
        knowledge of the entire network.

