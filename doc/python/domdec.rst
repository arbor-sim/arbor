.. _pydomdec:

Domain decomposition
====================

The Python API for partitioning a model over distributed and local hardware is described here.

Load balancers
--------------

.. currentmodule:: arbor

Load balancing generates a :class:`domain_decomposition` given an :class:`arbor.recipe`
and a description of the hardware on which the model will run. Currently Arbor provides
one load balancer, :func:`partition_load_balance`; and a function for creating
custom decompositions, :func:`partition_by_group`.
More load balancers will be added over time.

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

.. function:: partition_by_group(recipe, context, groups)

   Construct a :class:`domain_decomposition` that assigns the groups described by
   the provided list of :class:`group_description` to the local hardware of the calling rank.

   The function expects to be called by each rank in a distributed simulation with the
   selected groups for that rank. For example, in a simulation of 10 cells on 2 MPI ranks
   where cells {0, 2, 4, 6, 8} of kind :class:`cable_cell` are expected to be in a single group executed
   on the GPU on rank 0; and cells {1, 3, 5, 7, 9} of kind :class:`lif_cell` are expected to be in a single
   group executed on the CPU on rank 1:

   Rank 0 should run:

   .. code-block:: python

        import arbor

        # Get a communication context (with 4 threads, and 1 GPU with id 0)
        context = arbor.context(threads=4, gpu_id=0)

        # Initialise a recipe of user defined type my_recipe with 10 cells.
        n_cells = 10
        recipe = my_recipe(n_cells)

        groups = [arbor.group_description(arbor.cell_kind.cable, [0, 2, 4, 6, 8], arbor.backend.gpu)]
        decomp = arbor.partition_by_group(recipe, context, groups)

   And Rank 1 should run:

   .. code-block:: python

        import arbor

        # Get a communication context (with 4 threads, and no GPU)
        context = arbor.context(threads=4, gpu_id=-1)

        # Initialise a recipe of user defined type my_recipe with 10 cells.
        n_cells = 10
        recipe = my_recipe(n_cells)

        groups = [arbor.group_description(arbor.cell_kind.lif, [1, 3, 5, 7, 9], arbor.backend.multicore)]
        decomp = arbor.partition_by_group(recipe, context, groups)

   The function expects that cells connected by gap-junction are in the same group. An exception will be raised
   if this is not the case.

   The function doesn't perform any checks on the validity of the generated :class:`domain_decomposition`.
   The validity is only checked when a :class:`simulation` object is constructed using that :class:`domain_decomposition`.

   .. Note::
        This function is intended for users who have a good understanding of the computational
        cost of simulating the cells in their network and want fine-grained control over the
        partitioning of cells across ranks. It is recommended to start off by using
        :func:`partition_load_balance` and switch to this function if the observed performance
        across ranks is unbalanced (for example, if the performance of the network is not scaling
        well with the number of nodes.)

   .. Note::
        This function relies on the user to decide the size of the cell groups. It is therefore important
        to keep in mind that smaller cell groups have better performance on the multicore backend and
        larger cell groups have better performance on the GPU backend.

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

        # The hints prefer the multicore backend, so the decomposition is expected
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

    Describes a domain decomposition and is solely responsible for describing the
    distribution of cells across cell groups and domains.
    It holds cell group descriptions (:attr:`groups`) for cells assigned to
    the local domain, and a helper function (:func:`gid_domain`) used to
    look up which domain a cell has been assigned to.
    The :class:`domain_decomposition` object also has meta-data about the
    number of cells in the global model, and the number of domains over which
    the model is distributed.

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

    .. attribute:: num_groups

        The total number of cell groups on the local domain.

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
