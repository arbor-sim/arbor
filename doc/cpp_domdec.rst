.. _cppdomdec:

.. cpp:namespace:: arb

Domain Decomposition
====================

The C++ API for partitioning a model over distributed and local hardware is described here.

Load Balancers
--------------

Load balancing generates a :cpp:class:`domain_decomposition` given an :cpp:class:`arb::recipe`
and a description of the hardware on which the model will run. Currently Arbor provides
one load balancer, :cpp:func:`partition_load_balance`, and more will be added over time.

If the model is distributed with MPI, the partitioning algorithm for cells is
distributed with MPI communication. The returned :cpp:class:`domain_decomposition`
describes the cell groups on the local MPI rank.

.. Note::
    The :cpp:class:`domain_decomposition` type is
    independent of any load balancing algorithm, so users can define a
    domain decomposition directly, instead of generating it with a load balancer.
    This is useful for cases where the provided load balancers are inadequate,
    or when the user has specific insight into running their model on the
    target computer.

.. Important::
    When users supply their own :cpp:class:`domain_decomposition`, if they have
    **Gap Junction connections**, they have to be careful to place all cells that
    are connected via gap junctions in the same group.
    Example:
    ``A -gj- B -gj- C``  and ``D -gj- E``.
    Cells A, B and C need to be in a single group; and cells D and E need to be in a
    single group. They may all be placed in the same group but not necessarily.
    Be mindful that smaller cell groups perform better on multi-core systems and
    try not to overcrowd cell groups if not needed.
    Arbor provided load balancers such as :cpp:func:`partition_load_balance`
    guarantee that this rule is obeyed.

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

    Describes a domain decomposition and is solely responsible for describing the
    distribution of cells across cell groups and domains.
    It holds cell group descriptions (:cpp:member:`groups`) for cells assigned to
    the local domain, and a helper function (:cpp:member:`gid_domain`) used to
    look up which domain a cell has been assigned to.
    The :cpp:class:`domain_decomposition` object also has meta-data about the
    number of cells in the global model, and the number of domains over which
    the model is distributed.

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

        The gids of the cells in the cell group.

    .. cpp:member:: const backend_kind backend

        The back end on which the cell group is to run.
