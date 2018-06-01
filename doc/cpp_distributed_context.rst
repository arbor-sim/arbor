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

A global context is created by a user before building and running a simulation.
The context is then used to perform domain decomposition and initialize the simulation
(see :ref:`cppsimulation` for more about the simulation building workflow).
In the example below, a context that uses MPI is used to run a distributed simulation:

.. container:: example-code

    .. code-block:: cpp

        arb::hw::node_info node;
        my_recipe recipe;

        // Get an MPI communication context
        arb::distributed_context context = arb::mpi_context();

        // Partition model over the distributed system
        arb::domain_decomposition decomp = arb::partition_load_balance(recipe, node, &context);

        // Instatitate the simulation over the distributed system
        arb::simulation sim(recipe, decomp, &context);

        // Run the simulation for 100ms over the distributed system
        sim.run(100, 0.01);

By default :cpp:class:`arb::distributed_context` uses an :cpp:class:`arb::local_context`, which
runs on the local computer or node, that is, it is not distributed.

To run on a distributed system, use :cpp:class:`arb::mpi_context`, which uses
MPI for distributed communication.
By default the context will use the default MPI communicator ``MPI_COMM_WORLD``,
though it can be initialised with a user-supplied communicator.

.. container:: example-code

    .. code-block:: cpp

        arb::distributed_context context;

        // This is equivelent to default constructed context above
        arb::distributed_context context = arb::local_context();

        // Create an MPI context that uses MPI_COMM_WORLD
        arb::distributed_context context = arb::mpi_context();

        // create an MPI context with a user-supplied MPI_Comm
        arb::distributed_context context = arb::mpi_context(communicator);


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

        Default contstructor initializes the context as a :cpp:class:`local_context`.

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

        Special overload for gathering a string provided by each domain into a vector
        of strings on domain :cpp:var:`root`.

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

    Implements the :cpp:class:`arb::distributed_context` interface
    non-distributed computation.

    This is the default :cpp:class:`arb::distributed_context`, and should be used
    when running on laptop or workstation systems with one NUMA domain.

    .. Note::
        :cpp:class:`arb::local_context` provides the simplest possible distributed context,
        with only one process, and where all reduction operations are the identity operator.

    **Constructor:**

    .. cpp:function:: local_context()

        Default constructor.

.. cpp:class:: mpi_context

    Implements the :cpp:class:`arb::distributed_context` interface
    distributed computation using the MPI message passing library.

    **Constructor:**

    .. cpp:function:: mpi_context(MPI_Comm comm=MPI_COMM_WORLD)

        Create a context that will uses the MPI communicator :cpp:var:`comm`.
        By default uses the global communicator ``MPI_COMM_WORLD``.
