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

    See :cpp:class:`arb::domain_decimposition`.

Load balancer
    A distributed algorithm that determines the domain decomposition using the
    model recipe and and description of the available computational resources as
    inputs.

    See :cpp:func:`arb::partition_load_balance`.

Hardware
--------

.. cpp:namespace:: arb::hw

.. cpp:class:: node_info

    Information about the computational resources available to a simulation, typically a
    subset of the resources available on a physical hardware node.
    When used for distributed simulations, where a model will be distributed over more than
    one node, a :cpp:class:`hw::node_info` represents the resources available to the local
    MPI rank.

    .. container:: example-code

        .. code-block:: cpp

            // Make node that uses one thread for each available hardware thread,
            // and one GPU if any GPUs are available.
            hw::node_info node;
            node.num_cpu_cores = threading::num_threads();
            node.num_gpus = hw::num_gpus()>0? 1: 0;

    .. cpp:function:: node_info() = default

        Default constructor (sets 1 CPU core and 0 GPUs).

    .. cpp:function:: node_info(unsigned cores, unsigned gpus)

        Constructor that sets the number of :cpp:var:`cores` and :cpp:var:`gpus` available.

    .. cpp:member:: unsigned num_cpu_cores = 1

        The number of CPU cores available.

        By default it is assumed that there is one core available.

    .. cpp:member:: unsigned num_gpus = 0

        The number of GPUs available.

        By default it is assumed that there are no GPUs.

Load Balancers
--------------

Load balancing generates a :cpp:class:`domain_decomposition` given a :cpp:class:`recipe`
and a description of the hardware on which the model will run. Currently Arbor provides
one load balancer, :cpp:func:`partition_load_balance`, and more will be added over time.

If the model is distributed with MPI, the partitioning algorithm for cells is
distributed with MPI communication. The returned :cpp:class:`domain_decomposition`
describes the cell groups on the local MPI rank.

.. Note::
    Power users know their model and the hardware resources that
    they are running on very well, and know best how to partition their model
    workload to get optimal performance.
    The :cpp:class:`domain_decomposition` type is kept very simple, and is
    independent of any load balancing algorithm, so power users can supply their
    own domain decomposition without using one of the built-in load balancers.

.. cpp:namespace:: arb

.. cpp:function:: domain_decomposition partition_load_balance(const recipe& rec, hw::node_info nd)

    Construct a :cpp:class:`domain_decomposition` that distributes the cells
    in the model described by :cpp:var:`rec` over the hardware resources described
    by `hw::node_info`.

Decomposition
-------------

Documentation for the data structures used to describe domain decompositions.

.. cpp:namespace:: arb

.. cpp:enum-class:: backend_kind

    Used to indicate which hardware backend to use for running a :cpp:class:`cell_group`.

    .. cpp:enumerator:: multicore

        Use multicore backend.

    .. cpp:enumerator:: gpu

        Use gpu back end.

        .. Note::
            Can only be set if the :cpp:class:`cell_group` type supports the GPU backend.

.. cpp:class:: domain_decomposition

    Describes a domain decomposition. Solely responsible for describing the
    distribution of cells across cell groups and domains.
    Holds cell group descriptions (:cpp:member:`groups`) for cells assigned to
    the local domain, and a helper function (:cpp:member:`gid_domain`) used to
    look up which domain a cell has been assigned to.
    Also has meta-data about the number of cells in the global model, and the
    number of domains over which the model is destributed.

    A load balancing algorithm generates the domain decomposition, for example see
    :cpp:func:`partition_load_balance`.

    .. cpp:function:: bool is_local_gid(cell_gid_type gid) const

        Tests whether the cell with :cpp:var:`gid` is on the local domain.

    .. cpp:member:: std::function<int(cell_gid_type)> gid_domain

        A callback function that returns the domain id that a cell assigned to
        (using global identifier :cpp:var:`gid`).
        Set by the load balancing algorithm that generates the domain decomposition.

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
