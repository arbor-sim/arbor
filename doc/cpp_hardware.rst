.. _cpphardware:

Hardware Management
===================

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

.. cpp:namespace:: arbenv

.. cpp:function:: arb::util::optional<int> get_env_num_threads()

    Tests whether the number of threads to use has been set in an environment variable.
    First checks ``ARB_NUM_THREADS``, and if that is not set checks ``OMP_NUM_THREADS``.

    Return value:

    * **no value**: the :cpp:any:`optional` return value contains no value if the
      no thread count was specified by an environment variable.
    * **has value**: the number of threads set by the environment variable.

    Throws:

    * throws :cpp:any:`std::runtime_error` if environment variable set with invalid
      number of threads.

    .. container:: example-code

       .. code-block:: cpp

         #include <arborenv/concurrency.hpp>

         if (auto nt = arbenv::get_env_num_threads()) {
            std::cout << "requested " << nt.value() << "threads \n";
         }
         else {
            std::cout << "no environment variable set\n";
         }

.. cpp:function:: int thread_concurrency()

   Attempts to detect the number of available CPU cores. Returns 1 if unable to detect
   the number of cores.

    .. container:: example-code

       .. code-block:: cpp

         #include <arborenv/concurrency.hpp>

         // Set num_threads to value from environment variable if set,
         // otherwise set it to the available number of cores.
         int num_threads = 0;
         if (auto nt = arbenv::get_env_num_threads()) {
            num_threads = nt.value();
         }
         else {
            num_threads = arbenv::thread_concurrency();
         }

.. cpp:function:: int default_gpu()

   Returns the integer identifier of the first available GPU, if a GPU is available 

   Return value:

   * **non-negative value**: if a GPU is available, the index of the selected GPU is returned. The index will be in the range ``[0, num_gpus)`` where ``num_gpus`` is the number of GPUs detected using the ``cudaGetDeviceCount`` `CUDA API call <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html>`_.
   * **-1**: if no GPU available, or if Arbor was built without GPU support.

    .. container:: example-code

       .. code-block:: cpp

         #include <arborenv/gpu_env.hpp>

         if (arbenv::default_gpu()>-1) {}
            std::cout << "a GPU is available\n";
         }

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

     * :cpp:any:`std::runtime_error`: if there was an error in the CUDA runtime
       on the local or remote MPI ranks, i.e. if one rank throws, all ranks
       will throw.

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

