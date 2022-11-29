.. _pyhardware:

Hardware context
================

Arbor provides two ways for working with hardware resources:

* *Prescribe* the hardware resources and their contexts for use in Arbor simulations.
* *Query* available hardware resources (e.g. the number of available GPUs), and initializing MPI.

Note that to utilize some hardware features Arbor must be built and installed with the feature enabled, for example MPI or a GPU.
Please refer to the :ref:`installation guide <in_build_install>` for information on how to enable hardware support.

Available resources
-------------------

Helper functions for checking cmake or environment variables, as well as configuring and checking MPI are the following:

.. currentmodule:: arbor

.. function:: config()

    Returns a dictionary to check which options the Arbor library was configured with at compile time:

    * ``ARB_MPI_ENABLED``
    * ``ARB_WITH_MPI4PY``
    * ``ARB_GPU_ENABLED``
    * ``ARB_VECTORIZE``
    * ``ARB_WITH_PROFILING``
    * ``ARB_USE_BUNDLED_LIBS``
    * ``ARB_VERSION``
    * ``ARB_ARCH``

    .. container:: example-code

        .. code-block:: python

            import arbor
            arbor.config()

            {'mpi': True, 'mpi4py': True, 'gpu': False, 'vectorize': True, 'profiling': True, 'bundled': True, 'version': '0.5.3-dev', 'arch': 'native'}

.. function:: mpi_init()

    Initialize MPI with ``MPI_THREAD_SINGLE``, as required by Arbor.

.. function:: mpi_is_initialized()

    Check if MPI is initialized.

.. class:: mpi_comm

    .. method:: mpi_comm()

        By default sets MPI_COMM_WORLD as communicator.

    .. method:: mpi_comm(object)
        :noindex:

        :param object: Converts a Python object to an MPI Communicator.

.. function:: mpi_finalize()

    Finalize MPI by calling ``MPI_Finalize``.

.. function:: mpi_is_finalized()

    Check if MPI is finalized.
    :rtype: bool

Env: Helper functions
---------------------

The ``arbor.env`` module collects helper functions for interacting with the environment.

.. function:: find_private_gpu(comm)

   Requires GPU and MPI. Will return an integer id of a GPU such that each GPU
   is mapped to at most one MPI task (on the same node as the GPU). Raises an
   exception if
   - not built with GPU or MPI support
   - unable to satisfy the constraints above
   - handed an invalid or unknown MPI communicator object

Prescribed resources
---------------------

The Python wrapper provides an API for:

  * prescribing which hardware resources are to be used by a
    simulation using :class:`proc_allocation`.
  * opaque handles to hardware resources used by simulations called
    :class:`context`.

.. class:: proc_allocation

    Enumerates the computational resources on a node to be used for a simulation,
    specifically the number of threads and identifier of a GPU if available.

    .. method:: proc_allocation([threads=1, gpu_id=None])

        :param int threads: Number of threads.
        :param int gpu_id: Device ID.

    .. attribute:: threads

        The number of CPU threads available, 1 by default. Must be set to 1 at minimum.

    .. attribute:: gpu_id

        The identifier of the GPU to use.
        Must be ``None``, or a non-negative integer.

        The :attr:`gpu_id` corresponds to the ``int device`` parameter used by CUDA API calls
        to identify gpu devices.
        Set to ``None`` to indicate that no GPU device is to be used.
        See ``cudaSetDevice`` and ``cudaDeviceGetAttribute`` provided by the
        `CUDA API <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html>`_.

    .. method:: has_gpu()

        Indicates whether a GPU is selected (i.e., whether :attr:`gpu_id` is ``None``).

    Here are some examples of how to create a :class:`proc_allocation`.

    .. container:: example-code

        .. code-block:: python

            import arbor

            # default: one thread and no GPU selected
            alloc1 = arbor.proc_allocation()

            # 8 threads and no GPU
            alloc2 = arbor.proc_allocation(8, None)

            # reduce alloc2 to 4 threads and use the first available GPU
            alloc2.threads = 4
            alloc2.gpu_id  = 0

.. class:: context

    An opaque handle for the hardware resources used in a simulation.
    A :class:`context` contains a thread pool, and optionally the GPU state
    and MPI communicator. Users of the library do not directly use the functionality
    provided by :class:`context`, instead they configure contexts, which are passed to
    Arbor interfaces for domain decomposition and simulation.

    .. method:: context(threads, gpu_id, mpi)
        
        Create a distributed context.

        :param int threads:
            The number of threads available locally for execution.
            Must be set to 1 at minimum. 1 by default.
            Passing ``"avail_threads"`` (as string) will query and use the maximum number
            of threads the system makes available.
        :param int gpu_id:
            The non-negative identifier of the GPU to use, ``None`` by default.
            Can only be set when Arbor was built with GPU support.
        :type gpu_id: int or None
        :param mpi:
            The MPI communicator, ``None`` by default for distributed calculation.
            Can only be set when Arbor was built with MPI support.
        :type mpi: :py:class:`arbor.mpi_comm` or None.

    .. function:: context(alloc, mpi)
        :noindex:

        Create a distributed context.

        :param alloc:
            The computational resources, one thread and no GPU by default.
        :type alloc: :py:class:`proc_allocation`
        :param mpi:
            The MPI communicator, ``None`` by default for distributed calculation.
            Can only be set when Arbor was built with MPI support.
        :type mpi: :py:class:`arbor.mpi_comm` or None.


    Contexts can be queried for information about which features a context has enabled,
    whether it has a GPU, how many threads are in its thread pool.

    .. attribute:: has_gpu

        Query whether the context has a GPU.

    .. attribute:: has_mpi

        Query whether the context uses MPI for distributed communication.

    .. attribute:: threads

        Query the number of threads in the context's thread pool.

    .. attribute:: ranks

        Query the number of distributed domains.
        If the context has an MPI communicator, return is equivalent to ``MPI_Comm_size``.
        If the communicator has no MPI, returns 1.

    .. attribute:: rank

        The numeric id of the local domain.
        If the context has an MPI communicator, return is equivalent to ``MPI_Comm_rank``.
        If the communicator has no MPI, returns 0.

    Here are some simple examples of how to create a :class:`context`:

    .. container:: example-code

        .. code-block:: python

            import arbor
            import mpi4py.MPI as mpi

            # Construct a context that uses 1 thread and no GPU or MPI.
            context = arbor.context()

            # Construct a context that:
            #  * uses 8 threads in its thread pool;
            #  * does not use a GPU, reguardless of whether one is available
            #  * does not use MPI.
            alloc   = arbor.proc_allocation(8, None)
            context = arbor.context(alloc)

            # Construct a context that uses:
            #  * 4 threads and the first GPU;
            #  * MPI_COMM_WORLD for distributed computation.
            alloc   = arbor.proc_allocation(4, 0)
            comm    = arbor.mpi_comm(mpi.COMM_WORLD)
            context = arbor.context(alloc, comm)
