.. _cppdomdec:

.. cpp:namespace:: arb

Domain decomposition
====================

The C++ API for partitioning a model over distributed and local hardware is described here.

Decomposition
-------------

Documentation for the data structures used to describe domain decompositions.

.. cpp:namespace:: arb

.. cpp:class:: domain_decomposition

    Describes a domain decomposition and is solely responsible for describing the
    distribution of cells across cell groups and domains.
    It holds cell group descriptions (:cpp:member:`groups`) for cells assigned to
    the local domain, and a helper member (:cpp:member:`gid_domain`) used to
    look up which domain a cell has been assigned to.
    The :cpp:class:`domain_decomposition` object also has meta-data about the
    number of cells in the global model, and the number of domains over which
    the model is distributed.

    .. Note::
        The domain decomposition represents a division of **all** of the cells in
        the model into non-overlapping sets, with one set of cells assigned to
        each domain.
        A domain decomposition is generated either by a load balancer or is
        directly constructed by a user, the following conditions must be met,
        if not, an exception will be thrown:

        * Every cell in the model appears once in one and only one cell
          :cpp:class:`group <group_description>` on one and only one local
          :cpp:class:`domain_decomposition` object.
        * The total number of cells across all cell
          :cpp:class:`groups <group_description>` on all
          :cpp:class:`domain_decomposition` objects must match the total
          number of cells in the :cpp:class:`recipe`.
        * Cells that are connected via gap-junction must be present in the
          same cell :cpp:class:`group <group_description>`.

    .. cpp:function:: domain_decomposition(const recipe& rec, const context& ctx, const std::vector<group_description>& groups)

        The constructor takes:

        *   a :cpp:class:`arb::recipe` that describes the model;
        *   a :cpp:class:`arb::context` that describes the hardware resources;
        *   a vector of :cpp:class:`arb::group_description` that contains the indices of the cells
            to be executed on the local rank, categorized into groups.

        It's expected that a different :cpp:class:`arb::domain_decomposition` object will be constructed on
        each rank in a distributed simulation containing that selected cell groups for that rank.
        For example, in a simulation of 10 cells on 2 MPI ranks where cells {0, 2, 4, 6, 8} of kind
        :class:`cable_cell` are meant to be in a single group executed on the GPU on rank 0;
        and cells {1, 3, 5, 7, 9} of kind :class:`lif_cell` are expected to be in a single group executed
        on the CPU on rank 1:

        Rank 0 should run:

        .. code-block:: c++

            std::vector<arb::group_description> groups = {
                {arb::cell_kind::cable, {0, 2, 4, 6, 8}, arb::backend_kind::gpu}
            };
            auto decomp = arb::domain_decomposition(recipe, context, groups);

        And Rank 1 should run:

        .. code-block:: c++

            std::vector<arb::group_description> groups = {
                {arb::cell_kind::lif,   {1, 3, 5, 7, 9}, arb::backend_kind::multicore}
            };
            auto decomp = arb::domain_decomposition(recipe, context, groups);

        .. _domdecnotes:
        .. Important::
            Constructing a balanced :cpp:class:`domain_decomposition` quickly
            becomes a difficult task for large and diverse networks. This is why
            arbor provides :ref:`load balancing algorithms<domdecloadbalance>`
            that automatically generate a :cpp:class:`domain_decomposition` from
            a :cpp:class:`recipe` and :cpp:class:`context`.
            A user-defined :cpp:class:`domain_decomposition` using the constructor
            is useful for cases where the provided load balancers are inadequate,
            or when the user has specific insight into running their model on the
            target computer.

        .. Important::
            When creating your own :cpp:class:`domain_decomposition` of a network
            containing **Gap Junction connections**, be sure to place all cells that
            are connected via gap junctions in the same :cpp:member:`group <groups>`.
            Example:
            ``A -gj- B -gj- C``  and ``D -gj- E``.
            Cells A, B and C need to be in a single group; and cells D and E need to be in a
            single group. They may all be placed in the same group but not necessarily.
            Be mindful that smaller cell groups perform better on multi-core systems and
            try not to overcrowd cell groups if not needed.
            Arbor provided load balancers such as :cpp:func:`partition_load_balance`
            guarantee that this rule is obeyed.

    .. cpp:member:: int gid_domain(cell_gid_type gid)

        Returns the domain id of the cell with id ``gid``.

    .. cpp:member:: int num_domains()

        Returns the number of domains that the model is distributed over.

    .. cpp:member:: int domain_id()

        Returns the index of the local domain.
        Always 0 for non-distributed models, and corresponds to the MPI rank
        for distributed runs.

    .. cpp:member:: cell_size_type num_local_cells()

        Returns the total number of cells in the local domain.

    .. cpp:member:: cell_size_type num_global_cells()

        Returns the total number of cells in the global model
        (sum of :cpp:member:`num_local_cells` over all domains).

    .. cpp:member:: cell_size_type num_groups()

        Returns the total number of cell groups on the local domain.

    .. cpp:member:: const group_description& group(unsigned idx)

        Returns the description of the cell group at index ``idx`` on the local domain.
        See :cpp:class:`group_description`.

    .. cpp:member:: const std::vector<group_description>& groups()

        Returns the descriptions of the cell groups on the local domain.
        See :cpp:class:`group_description`.

.. cpp:class:: group_description

    The indexes of a set of cells of the same kind that are group together in a
    cell group in a :cpp:class:`arb::simulation`.

    .. cpp:function:: group_description(cell_kind k, std::vector<cell_gid_type> g, backend_kind b)

        Constructor.

    .. cpp:member:: const cell_kind kind

        The kind of cell in the group.

    .. cpp:member:: const std::vector<cell_gid_type> gids

        The gids of the cells in the cell group.

    .. cpp:member:: const backend_kind backend

        The back end on which the cell group is to run.

.. cpp:enum-class:: backend_kind

    Used to indicate which hardware backend to use for running a :cpp:class:`cell_group`.

    .. cpp:enumerator:: multicore

        Use multicore backend.

    .. cpp:enumerator:: gpu

        Use GPU back end.

        .. Note::
            Setting the GPU back end is only meaningful if the
            :cpp:class:`cell_group` type supports the GPU backend.

.. _domdecloadbalance:

Load balancers
--------------

Load balancing generates a :cpp:class:`domain_decomposition` given an :cpp:class:`arb::recipe`
and a description of the hardware on which the model will run. Currently Arbor provides
one load balancer, :cpp:func:`partition_load_balance`, and more will be added over time.

If the model is distributed with MPI, the partitioning algorithm for cells is
distributed with MPI communication. The returned :cpp:class:`domain_decomposition`
describes the cell groups on the local MPI rank.

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
