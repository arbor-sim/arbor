.. _cppdomdec:

Domain Decomposition
====================

Definitions
-----------

Domain decomposition
    A description of the distribution of the model over the available
    computational resources. The description partitions the
    cells in the model as follows:

        * group the cells into *cell groups* of the same kind of cell;
        * assign each cell group to either a CPU core or GPU on a specific MPI rank.

    The number of cells in each cell group depends on different factors,
    including the type of the cell, and whether the cell group will run on a CPU
    core or the GPU.

    See :cpp:class:`arb::domain_decomposition`.

Load balancer
    A distributed algorithm that determines the domain decomposition using the
    model recipe and a description of the available computational resources as
    inputs.

    See :cpp:func:`arb::partition_load_balance`.

Hardware
--------

.. cpp:namespace:: arb

.. cpp:class:: local_resources

    Enumerates the computational resources available locally, specifically the
    number of hardware threads and the number of GPUs.

    The function :cpp:func:`arb::get_local_resources` can be used to automatically
    detect the available resources are available :cpp:class:`local_resources` 

    .. container:: example-code

        .. code-block:: cpp

            auto resources = arb::get_local_resources();
            std::cout << "This node supports " << resources.num_threads " threads," <<
                      << " and " << resources.num_gpus << " gpus.";

    .. cpp:function:: local_resources(unsigned threads, unsigned gpus)

        Constructor.

    .. cpp:member:: const unsigned num_threads

        The number of threads available.

    .. cpp:member:: const unsigned num_gpus

        The number of GPUs available.

.. cpp:function:: local_resources get_local_resources()

    Returns an instance of :cpp:class:`local_resources` with the following:

    * ``num_threads`` is determined from the ``ARB_NUM_THREADS`` environment variable if
      set, otherwise Arbor attempts to detect the number of available hardware cores.
      If Arbor can't determine the available threads it defaults to 1 thread.
    * ``num_gpus`` is the number of GPUs detected using the CUDA ``cudaGetDeviceCount`` that
      `API call <https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html>`_.

.. cpp:class:: proc_allocation

    Enumerates the computational resources to be used for a simulation, typically a
    subset of the resources available on a physical hardware node.

    .. container:: example-code

        .. code-block:: cpp

            // Default construction uses all detected cores/threads, and the first GPU, if available.
            arb::proc_allocation resources;

            // Remove any GPU from the resource description.
            resources.gpu_id = -1;


    .. cpp:function:: proc_allocation() = default

        Sets the number of threads to the number detected by :cpp:func:`get_local_resources`, and
        chooses either the first available GPU, or no GPU if none are available.

    .. cpp:function:: proc_allocation(unsigned threads, int gpu_id)

        Constructor that sets the number of :cpp:var:`threads` and selects :cpp:var:`gpus` available.

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

Execution Context
-----------------

The :cpp:class:`proc_allocation` class enumerates the hardware resources on the local hardware
to use for a simulation.

.. cpp:namespace:: arb

.. cpp:class:: context

    A handle for the interfaces to the hardware resources used in a simulation.
    A :cpp:class:`context` contains the local thread pool, and optionally the GPU state
    and MPI communicator, if available. Users of the library do not directly use the functionality
    provided by :cpp:class:`context`, instead they configure contexts, which are passed to
    Arbor methods and types.

.. cpp:function:: context make_context()

    Local context that uses all detected threads and a GPU if any are available.

.. cpp:function:: context make_context(proc_allocation alloc)

    Local context that uses the local resources described by :cpp:var:`alloc`.

.. cpp:function:: context make_context(proc_allocation alloc, MPI_Comm comm)

    A context that uses the local resources described by :cpp:var:`alloc`, and
    uses the MPI communicator :cpp:var:`comm` for distributed calculation.


Here are some examples of how to create a :cpp:class:`arb::context`:

    .. container:: example-code

        .. code-block:: cpp

            #include <arbor/context.hpp>

            // Construct a non-distributed context that uses all detected available resources.
            auto context = arb::make_context();

            // Construct a context that:
            //  * does not use a GPU, reguardless of whether one is available;
            //  * uses 8 threads in its thread pool.
            arb::proc_allocation resources(8, -1);
            auto context = arb::make_context(resources);

            // Construct a context that:
            //  * uses all available local hardware resources;
            //  * uses the standard MPI communicator MPI_COMM_WORLD for distributed computation.
            arb::proc_allocation resources; // defaults to all detected local resources
            auto context = arb::make_context(resources, MPI_COMM_WORLD);

Load Balancers
--------------

Load balancing generates a :cpp:class:`domain_decomposition` given a :cpp:class:`recipe`
and a description of the hardware on which the model will run. Currently Arbor provides
one load balancer, :cpp:func:`partition_load_balance`, and more will be added over time.

If the model is distributed with MPI, the partitioning algorithm for cells is
distributed with MPI communication. The returned :cpp:class:`domain_decomposition`
describes the cell groups on the local MPI rank.

.. Note::
    The :cpp:class:`domain_decomposition` type is simple and
    independent of any load balancing algorithm, so users can supply their
    own domain decomposition without using one of the built-in load balancers.
    This is useful for cases where the provided load balancers are inadequate,
    and when the user has specific insight into running their model on the
    target computer.

.. cpp:namespace:: arb

.. cpp:function:: domain_decomposition partition_load_balance(const recipe& rec, const arb::context& ctx)

    Construct a :cpp:class:`domain_decomposition` that distributes the cells
    in the model described by :cpp:any:`rec` over the distributed and local hardware
    resources described by :cpp:any:`ctx`.

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

Decomposition
-------------

Documentation for the data structures used to describe domain decompositions.

.. cpp:namespace:: arb

.. cpp:enum-class:: backend_kind

    Used to indicate which hardware backend to use for running a :cpp:class:`cell_group`.

    .. cpp:enumerator:: multicore

        Use multicore backend.

    .. cpp:enumerator:: gpu

        Use GPU back end.

        .. Note::
            Setting the GPU back end is only meaningful if the
            :cpp:class:`cell_group` type supports the GPU backend.

.. cpp:class:: domain_decomposition

    Describes a domain decomposition and is soley responsible for describing the
    distribution of cells across cell groups and domains.
    It holds cell group descriptions (:cpp:member:`groups`) for cells assigned to
    the local domain, and a helper function (:cpp:member:`gid_domain`) used to
    look up which domain a cell has been assigned to.
    The :cpp:class:`domain_decomposition` object also has meta-data about the
    number of cells in the global model, and the number of domains over which
    the model is destributed.

    .. Note::
        The domain decomposition represents a division **all** of the cells in
        the model into non-overlapping sets, with one set of cells assigned to
        each domain.
        A domain decomposition is generated either by a load balancer or is
        directly specified by a user, and it is a requirement that the
        decomposition is correct:

            * Every cell in the model appears once in one and only one cell
              :cpp:member:`groups` on one and only one local
              :cpp:class:`domain_decomposition` object.
            * :cpp:member:`num_local_cells` is the sum of the number of cells in
              each of the :cpp:member:`groups`.
            * The sum of :cpp:member:`num_local_cells` over all domains matches
              :cpp:member:`num_global_cells`.

    .. cpp:member:: std::function<int(cell_gid_type)> gid_domain

        A function for querying the domain id that a cell assigned to
        (using global identifier :cpp:var:`gid`).
        It must be a pure function, that is it has no side effects, and hence is
        thread safe.

    .. cpp:member:: int num_domains

        Number of domains that the model is distributed over.

    .. cpp:member:: int domain_id

        The index of the local domain.
        Always 0 for non-distributed models, and corresponds to the MPI rank
        for distributed runs.

    .. cpp:member:: cell_size_type num_local_cells

        Total number of cells in the local domain.

    .. cpp:member:: cell_size_type num_global_cells

        Total number of cells in the global model
        (sum of :cpp:member:`num_local_cells` over all domains).

    .. cpp:member:: std::vector<group_description> groups

        Descriptions of the cell groups on the local domain.
        See :cpp:class:`group_description`.

.. cpp:class:: group_description

    The indexes of a set of cells of the same kind that are group together in a
    cell group in a :cpp:class:`arb::simulation`.

    .. cpp:function:: group_description(cell_kind k, std::vector<cell_gid_type> g, backend_kind b)

        Constructor.

    .. cpp:member:: const cell_kind kind

        The kind of cell in the group.

    .. cpp:member:: const std::vector<cell_gid_type> gids

        The gids of the cells in the cell group, **sorted in ascending order**.

    .. cpp:member:: const backend_kind backend

        The back end on which the cell group is to run.
