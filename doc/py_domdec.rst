.. _pydomdec:

Domain Decomposition
====================

The Python API for partitioning a model over distributed and local hardware is described here.

Load Balancers
--------------

.. currentmodule:: arbor

Load balancing generates a :class:`domain_decomposition` given an :class:`arbor.recipe`
and a description of the hardware on which the model will run. Currently Arbor provides
one load balancer, :func:`partition_load_balance`, and more will be added over time.

If the model is distributed with MPI, the partitioning algorithm for cells is
distributed with MPI communication. The returned :class:`domain_decomposition`
describes the cell groups on the local MPI rank.

.. function:: partition_load_balance(recipe, context, hints)

    Construct a :class:`domain_decomposition` that distributes the cells
    in the model described by an :class:`arbor.recipe` over the distributed and local hardware
    resources described by an :class:`arbor.context`.

    The algorithm counts the number of each cell type in the global model, then
    partitions the cells of each type equally over the available nodes.
    If a GPU is available, and if the cell type can be run on the GPU, the
    cells on each node are put into one large group to maximise the amount of fine
    grained parallelism in the cell group.
    Otherwise, cells are grouped into small groups that fit in cache, and can be
    distributed over the available cores.
    Optionally, provide a dictionary of :class:`partition_hint` s for certain cell kinds, by default this dictionary is empty.

    .. Note::
        The partitioning assumes that all cells of the same kind have equal
        computational cost, hence it may not produce a balanced partition for
        models with cells that have a large variance in computational costs.

.. class:: partition_hint

    Provide a hint on how the cell groups should be partitioned.

    .. function:: partition_hint(cpu_group_size, gpu_group_size, prefer_gpu)

        Construct a partition hint with arguments :attr:`cpu_group_size` and :attr:`gpu_group_size`, and whether to :attr:`prefer_gpu`.

        By default returns a partition hint with :attr:`cpu_group_size` = ``1``, i.e., each cell is put in its own group, :attr:`gpu_group_size` = ``max``, i.e., all cells are put in one group, and :attr:`prefer_gpu` = ``True``, i.e., GPU usage is preferred.

    .. attribute:: cpu_group_size

        The size of the cell group assigned to CPU.
        Must be positive, else set to default value.

    .. attribute:: gpu_group_size

        The size of the cell group assigned to GPU.
        Must be positive, else set to default value.

    .. attribute:: prefer_gpu

        Whether GPU usage is preferred.

    .. attribute:: max_size

        Get the maximum size of cell groups.

An example of a partition load balance with hints reads as follows:

.. container:: example-code

    .. code-block:: python

        import arbor

        # Get a communication context (with 4 threads, no GPU)
        context = arbor.context(threads=4, gpu_id=None)

        # Initialise a recipe of user defined type my_recipe with 100 cells.
        n_cells = 100
        recipe = my_recipe(n_cells)

        # The hints perfer the multicore backend, so the decomposition is expected
        # to never have cell groups on the GPU, regardless of whether a GPU is
        # available or not.
        cable_hint                  = arb.partition_hint()
        cable_hint.prefer_gpu       = False
        cable_hint.cpu_group_size   = 3
        spike_hint                  = arb.partition_hint()
        spike_hint.prefer_gpu       = False
        spike_hint.cpu_group_size   = 4
        hints = dict([(arb.cell_kind.cable, cable_hint), (arb.cell_kind.spike_source, spike_hint)])

        decomp = arb.partition_load_balance(recipe, context, hints)


Decomposition
-------------
As defined in :ref:`modeldomdec` a domain decomposition is a description of the distribution of the model over the available computational resources.
Therefore, the following data structures are used to describe domain decompositions.

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

        A function for querying the domain id that a cell is assigned to (using global identifier :attr:`arbor.cell_member.gid`).

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

            The hardware :class:`backend` on which the cell group will run.
