.. _pydomdec:

Domain Decomposition
====================

Decomposition
-------------

Documentation for the data structures used to describe domain decompositions.

.. module:: arbor

.. class:: backend_kind

    Enumeration class used to indicate which hardware backend to use for running a :class:`cell_group`.

    .. attribute:: multicore

        Use multicore backend.

    .. attribute:: gpu

        Use GPU back end.

    .. Note::
        Setting the GPU back end is only meaningful if the
        :class:`cell_group` type supports the GPU backend.

.. class:: domain_decomposition

    Describes a domain decomposition and is soley responsible for describing the
    distribution of cells across cell groups and domains.
    It holds cell group descriptions (:attr:`groups`) for cells assigned to
    the local domain, and a helper function (:func:`gid_domain`) used to
    look up which domain a cell has been assigned to.
    The :class:`domain_decomposition` object also has meta-data about the
    number of cells in the global model, and the number of domains over which
    the model is destributed.

    .. Note::
        The domain decomposition represents a division of **all** of the cells in
        the model into non-overlapping sets, with one set of cells assigned to
        each domain.
        A domain decomposition is generated either by a load balancer or is
        directly specified by a user, and it is a requirement that the
        decomposition is correct:

        * Every cell in the model appears once in one and only one cell :attr:`groups` on one and only one local :class:`domain_decomposition` object.
        * :attr:`num_local_cells` is the sum of the number of cells in each of the :attr:`groups`.
        * The sum of :attr:`num_local_cells` over all domains matches :attr:`num_global_cells`.

    .. function:: gid_domain(gid)

        A function for querying the domain id that a cell assigned to
        (using global identifier :attr:`arbor.cell_member.gid`).

    .. attribute:: num_domains

        Number of domains that the model is distributed over.

    .. attribute:: domain_id

        The index of the local domain.
        Always 0 for non-distributed models, and corresponds to the MPI rank
        for distributed runs.

    .. attribute:: num_local_cells

        Total number of cells in the local domain.

    .. attribute:: num_global_cells

        Total number of cells in the global model
        (sum of :attr:`num_local_cells` over all domains).

    .. attribute:: groups

        Descriptions of the cell groups on the local domain.
        See :class:`arbor.group_description`.

.. class:: group_description

    The indexes of a set of cells of the same kind that are grouped together in a cell group in an :class:`arbor.simulation`.

        .. function:: group_description(kind, gids, backend)

            Constructor.

        .. attribute:: kind

            The kind of cell in the group.

        .. attribute:: gids

            The (list of) gids of the cells in the cell group, **sorted in ascending order**.

        .. attribute:: backend

            The back end on which the cell group is to run.


Load Balancers
--------------

Load balancing generates an :class:`arbor.domain_decomposition` given an :class:`arbor.recipe`
and a description of the hardware on which the model will run. Currently Arbor provides
one load balancer, :func:`partition_load_balance`, and more will be added over time.

If the model is distributed with MPI, the partitioning algorithm for cells is
distributed with MPI communication. The returned :class:`arbor.domain_decomposition`
describes the cell groups on the local MPI rank.

.. Note::
    The :class:`arbor.domain_decomposition` type is simple and
    independent of any load balancing algorithm, so users can supply their
    own domain decomposition without using one of the built-in load balancers.
    This is useful for cases where the provided load balancers are inadequate,
    and when the user has specific insight into running their model on the
    target computer.

.. function:: partition_load_balance(recipe, context)

    Construct an :class:`arbor.domain_decomposition` that distributes the cells
    in the model described by an :class:`arbor.recipe` over the distributed and local hardware
    resources described by an :class:`arbor.context`.

    The algorithm counts the number of each cell type in the global model, then
    partitions the cells of each type equally over the available nodes.
    If a GPU is available, and if the cell type can be run on the GPU, the
    cells on each node are put one large group to maximise the amount of fine
    grained parallelism in the cell group.
    Otherwise, cells are grouped into small groups that fit in cache, and can be
    distributed over the available cores.

    .. Note::
        The partitioning assumes that all cells of the same kind have equal
        computational cost, hence it may not produce a balanced partition for
        models with cells that have a large variance in computational costs.

Hardware
--------

.. class:: proc_allocation

    Enumerates the computational resources to be used for a simulation, typically a
    subset of the resources available on a physical hardware node.

    .. container:: example-code

        .. code-block:: python

            # Default construction uses all detected cores/threads, and the first GPU, if available.
            import arbor
            alloc = arbor.proc_allocation()

            # Remove any GPU from the resource description.
            alloc.gpu_id = -1


    .. function:: proc_allocation() = default

        Sets the number of threads to the number available locally for execution, and
        chooses either the first available GPU, or no GPU if none are available.

    .. function:: proc_allocation(threads, gpu_id)

        Constructor that sets the number of :attr:`threads` and selects :attr:`gpu_id`.

    .. attribute:: threads

        The number of CPU threads available locally for execution.

    .. attribute:: gpu_id

        The identifier of the GPU to use.
        The :attr:`gpu_id` corresponds to the ``int device`` parameter used by CUDA API calls
        to identify gpu devices.
        Set to -1 to indicate that no GPU device is to be used.
        See ``cudaSetDevice`` and ``cudaDeviceGetAttribute`` provided by the
        `CUDA API <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html>`_.

    .. cpp:function:: has_gpu()

        Indicates (with True/ False) whether a GPU is selected (i.e. whether :attr:`gpu_id` is ``-1``).

Execution Context
-----------------

The :class:`arbor.proc_allocation` class enumerates the hardware resources on the local hardware
to use for a simulation.

.. class:: context

    A handle for the interfaces to the hardware resources used in a simulation.
    A :class:`context` contains the local thread pool, and optionally the GPU state
    and MPI communicator, if available. Users of the library do not directly use the functionality
    provided by :class:`context`, instead they configure contexts, which are passed to
    Arbor methods and types.

    .. function:: context()

        Local context that uses all detected threads and a GPU if any are available.

    .. function:: context(proc_allocation)

        Local context that uses the local resources described by :class:`arbor.proc_allocation`.

    .. function:: context(proc_allocation, mpi_comm)

        A context that uses the local resources described by :class:`arbor.proc_allocation`, and
        uses the MPI communicator :class:`arbor.mpi_comm` for distributed calculation.

    .. function:: context(threads, gpu)

        A context that uses a set number of :attr:`threads` and gpu id :attr:`gpu`.

        .. attribute:: threads

            The number of threads available locally for execution (default: 1).

        .. attribute:: gpu

            The index of the GPU to use (default: none for no GPU).

    .. function:: context(threads, gpu, mpi)

        A context that uses a set number of :attr:`threads` and gpu id :attr:`gpu`.

        .. attribute:: threads

            The number of threads available locally for execution (default: 1).

        .. attribute:: gpu

            The index of the GPU to use (default: none for no GPU).

        .. attribute:: mpi

            The MPI communicator :class:`arbor.mpi_comm` (default: none for no MPI).

    .. attribute:: has_mpi

        Whether the context uses MPI for distributed communication.

    .. attribute:: has_gpu

        Whether the context has a GPU.

    .. attribute:: threads

        The number of threads available locally for execution.

    .. attribute:: ranks

        The number of distributed domains (equivalent to the number of MPI ranks).

    .. attribute:: rank

        The numeric id of the local domain (equivalent to MPI rank).

Here are some examples of how to create a :class:`context`:

    .. container:: example-code

        .. code-block:: python

            import arbor

            # Construct a non-distributed context that uses all detected available resources.
            context = arbor.context()

            # Construct a context that:
            #  * does not use a GPU, reguardless of whether one is available;
            #  * uses 8 threads in its thread pool.
            alloc = proc_allocation(8, -1)
            context = arbor.context(alloc)

            # Construct a context that:
            #  * uses all available local hardware resources;
            #  * uses the standard MPI communicator MPI_COMM_WORLD for distributed computation.
            alloc = proc_allocation() # defaults to all detected local resources
            comm = arb.mpi_comm()
            context = arbor.context(alloc, comm);
