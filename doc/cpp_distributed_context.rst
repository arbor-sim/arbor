.. _cppdistcontext:

Distributed Context
===================

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


Class Documentation
-------------------

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

