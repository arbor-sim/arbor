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

Why Recipes?
--------------

The interface and design of Arbor recipes was motivated by the following aims:

    * Building a simulation from a recipe description must be possible in a
      distributed system efficiently with minimal communication.
    * To minimise the amount of memory used in model building, to make it
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


General Best Practices
----------------------

.. topic:: Think of the cells

    When formulating a model, think cell-first, and try to formulate the model and
    the associated workflow from a cell-centered perspective. If this isn't possible,
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


Mechanisms
----------------------
The description of multi-compartment cells also includes the specification of ion channel and synapse dynamics.
In the recipe, these specifications are called *mechanisms*.
Implementations of mechanisms are either hand-coded or a translator (modcc) is used to compile a
subset of NEURON's mechanism specification language NMODL.

Examples
    Common examples are the *passive/ leaky integrate-and-fire* model, the *Hodgkin-Huxley* mechanism, the *(double-) exponential synapse* model, or the *Natrium current* model for an axon.

Detailed documentation for Python recipes can be found in :ref:`pyrecipe`.
C++ :ref:`cpprecipe` are documented and best practices are shown as well.
