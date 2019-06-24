.. _pydomdec:

Domain Decomposition
====================

Load Balancers
--------------

Load balancing generates a :class:`domain_decomposition` given a :class:`recipe`
and a description of the hardware on which the model will run. Currently Arbor provides
one load balancer, :func:`partition_load_balance`, and more will be added over time.

If the model is distributed with MPI, the partitioning algorithm for cells is
distributed with MPI communication. The returned :class:`domain_decomposition`
describes the cell groups on the local MPI rank.

.. function:: partition_load_balance(recipe, context)

    Construct a :class:`domain_decomposition` that distributes the cells
    in the model described by a :class:`recipe` over the distributed and local hardware
    resources described by a :class:`context`.

    The algorithm counts the number of each cell type in the global model, then
    partitions the cells of each type equally over the available nodes.
    If a GPU is available, and if the cell type can be run on the GPU, the
    cells on each node are put into one large group to maximise the amount of fine
    grained parallelism in the cell group.
    Otherwise, cells are grouped into small groups that fit in cache, and can be
    distributed over the available cores.

    .. Note::
        The partitioning assumes that all cells of the same kind have equal
        computational cost, hence it may not produce a balanced partition for
        models with cells that have a large variance in computational costs.

Decomposition
-------------
As defined in :ref:`modeldomdec` a domain decomposition is a description of the distribution of the model over the available computational resources.
Therefore, the following data structures are used to describe domain decompositions.

.. currentmodule:: arbor

.. class:: backend

    Enumeration used to indicate which hardware backend to execute a cell group on.

    .. attribute:: multicore

        Use multicore backend.

    .. attribute:: gpu

        Use GPU backend.

    .. Note::
        Setting the GPU back end is only meaningful if the cell group type supports the GPU backend.

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

    .. function:: gid_domain(gid)

        A function for querying the domain id that a cell is assigned to (using global identifier :attr:`cell_member.gid`).

    .. attribute:: num_domains

        The number of domains that the model is distributed over.

    .. attribute:: domain_id

        The index of the local domain.
        Always 0 for non-distributed models, and corresponds to the MPI rank
        for distributed runs.

    .. attribute:: num_local_cells

        The total number of cells in the local domain.

    .. attribute:: num_global_cells

        The total number of cells in the global model
        (sum of :attr:`num_local_cells` over all domains).

    .. attribute:: groups

        The descriptions of the cell groups on the local domain.
        See :class:`group_description`.

.. class:: group_description

    Return the indexes of a set of cells of the same kind that are grouped together in a cell group in an :class:`arbor.simulation`.

        .. function:: group_description(kind, gids, backend)

            Construct a group description with parameters :attr:`kind`, :attr:`gids` and :attr:`backend`.

        .. attribute:: kind

            The kind of cell in the group.

        .. attribute:: gids

            The list of gids of the cells in the cell group.

        .. attribute:: backend

            The hardware backend on which the cell group will run.
