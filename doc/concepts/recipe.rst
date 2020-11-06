.. _modelrecipe:

Recipes
===============

An Arbor *recipe* is a description of a model. The recipe is queried during the model
building phase to provide information about cells in the model, such as:

  * the number of cells in the model;
  * the type of a cell;
  * a description of a cell, e.g. with soma, synapses, detectors, stimuli;
  * the number of spike targets;
  * the number of spike sources;
  * the number of gap junction sites;
  * incoming network connections from other cells terminating on a cell;
  * gap junction connections on a cell.
  * probes on a cell.

Why recipes?
--------------

The interface and design of Arbor recipes was motivated by the following aims:

    * Building a simulation from a recipe description must be possible in a
      distributed system efficiently with minimal communication.
    * Minimising the amount of memory used in model building, making it
      possible to build and run simulations in one run.

Recipe descriptions are cell-oriented, in order that the building phase can
be efficiently distributed and that the model can be built independently of any
runtime execution environment.

During model building, the recipe is queried first by a load balancer,
then later when building the low-level cell groups and communication network.
The cell-centered recipe interface, whereby cell and network properties are
specified "per-cell", facilitates this.

The steps of building a simulation from a recipe are:

.. topic:: 1. Load balancing

    First, the cells are partitioned over MPI ranks, and each rank parses
    the cells assigned to it to build a cost model.
    The ranks then coordinate to redistribute cells over MPI ranks so that
    each rank has a balanced workload. Finally, each rank groups its local
    cells into :cpp:type:`cell_group` s that balance the work over threads (and
    GPU accelerators if available).

.. topic:: 2. Model building

    The model building phase takes the cells assigned to the local rank, and builds the
    local cell groups and the part of the communication network by querying the recipe
    for more information about the cells assigned to it.

.. Note::
    An example of how performance considerations impact Arbor's architecture:
    you will notice cell kind and cell description are separately added to a recipe.
    Consider the following conversation between an Arbor simulation, recipe and hardware back-end:

    | Simulator: give me cell 37.
    | Recipe: here you go, it's of C++ type s3cr1ts4uc3.
    | Simulator: wot? What is the cell kind for cell 37?
    | Recipe: it's a foobar.
    | Simulator: Okay.
    | Cell group implementations: which one of you lot deals with foobars?
    | Foobar_GPUFTW_lolz: That'd be me, if we've got GPU enabled.
    | Simulator: Okay it's up to you then to deal with this s3cr1ts4uc3 object.

General best practices
----------------------

.. topic:: Think of the cells

    When formulating a model, think cell-first, and try to formulate the model and
    the associated workflow from a cell-centred perspective. If this isn't possible,
    please contact the developers, because we would like to develop tools that help
    make this simpler.

.. _recipe_lazy:

.. topic:: Be lazy

    A recipe does not have to contain a complete description of the model in
    memory. Precompute as little as possible, and use
    `lazy evaluation <https://en.wikipedia.org/wiki/Lazy_evaluation>`_ to generate
    information only when requested.
    This has multiple benefits, including:

        * thread safety;
        * minimising the memory footprint of the recipe.

.. topic:: Be reproducible

    Arbor is designed to give reproducible results when the same model is run on a
    different number of MPI ranks or threads, or on different hardware (e.g. GPUs).
    This only holds when a recipe provides a reproducible model description, which
    can be a challenge when a description uses random numbers, e.g. to pick incoming
    connections to a cell from a random subset of a cell population.
    To get a reproducible model, use the cell `gid` (or a hash based on the `gid`)
    to seed random number generators, including those for :cpp:type:`event_generator` s.


API
---

* :ref:`Python <pyrecipe>`
* :ref:`C++ <cpprecipe>`
