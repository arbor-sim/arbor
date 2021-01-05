.. _modelrecipe:

Recipes
=======

An Arbor *recipe* is a description of a model. The recipe is queried during the model
building phase to provide information about individual cells in the model, such as:

  * The **number of cells** in the model.
  * The **kind** of each cell.
  * The **description** of each cell, e.g. with morphology, dynamics, synapses, detectors,
    stimuli etc.
  * The number of **spike targets** on each cell.
  * The number of **spike sources** on each cell.
  * The number of **gap junction sites** on each cell.
  * Incoming **network connections** from other cells terminating on a cell.
  * **Gap junction connections** on each cell.
  * **Probes** on each cell.

Recipes are structured to provide a consistent interface for describing each cell in the
network using their global identifier (`gid`).
This allows the simulator to be able to quickly look-up properties related to the connections
going in and out of a cell (think of synapses, gap junctions, but also probes and spike inputs);
which helps make Arbor fast and easily distributable over many nodes.

To better illustrate the content of a recipe, let's consider the following network of
three cells:

-  | ``Cell 0``: Is a single soma, with ``hh`` (Hodgkin-huxley) dynamics. In the middle
     of the soma, a spike detector is attached, it generates a spiking event when the
     voltage goes above 10 mV. In the same spot on the soma, a current clamp is also
     attached, with the intention of triggering some spikes. All of the preceding info:
     the morphology, dynamics, spike detector and current clamp are what is referred to in
     Arbor as the **description** of the cell.
   | ``Cell 0`` should be modelled as a :ref:`cable cell<modelcablecell>`,
     (because cable cells allow complex dynamics such as ``hh``). This is referred to as
     the **kind** of the cell.
   | It's quite expensive to build cable cells, so we don't want to do this too often.
     But when the simulation is first set up, it needs to know how cells interact with
     one another in order to distribute the simulation over the available computational
     resources. This is why the number of **targets**, **sources** and **gap junction sites**
     is needed separately from the cell description: with them, the simulation can tell
     that ``cell 0`` has 1 **spike source** (the detector), 0 **spike targets**, and 0
     **gap junction sites**, without having to build the cell.
-  | ``Cell 1``: Is a soma and a single dendrite, with ``passive`` dynamics everywhere.
     It has a single synapse at the end of the dendrite and a gap junction site in the
     middle of the soma. This is the **description** of the cell.
     It's also a cable cell, which is its **cell kind**. It has 0 **spike sources**, 1
     **spike target** (the synapse) and 1 **gap junction site**.
-  | ``Cell 2``: Is a soma and a single dendrite, with ``passive`` dynamics everywhere.
     It has a gap junction site in the middle of the soma. This is the **description**
     of the cell. It's also a cable cell, which is its **cell kind**. It has 0
     **spike sources**, 0 **spike targets** and 1 **gap junction site**.

The total **number of cells** in the model is 3. The **kind**, **description** and
number of **spike sources**, **spike targets** and **gap junction sites** on each cell
is known and can be registered in the recipe. Next is the cell interaction.

The model is designed such that ``cell 0`` has a spike source, ``cell 1`` has
a spike target and gap junction site, and ``cell 2`` has a gap junction site. A
**network connection** can be formed from ``cell 0`` to ``cell 1``; and a
**gap junction connection** from ``cell 1`` to ``cell 2``. If ``cell 0`` spikes,
a spike should be observed on ``cell 2`` after some delay. To monitor
the voltage on ``cell 2`` and record the spike, a **probe** can be set up
on ``cell 2``. All this information is also registered via the recipe.

There are additional docs on :ref:`cell kinds <modelcellkind>`;
:ref:`cell descriptions <modelcelldesc>`; :ref:`network connections <modelconnections>`;
:ref:`gap junction connections <modelgapjunctions>`; :ref:`probes <modelprobes>`

The recipe is used to distribute the model across machines and is used in the simulation.
Technical details of the recipe class are presented in the  :ref:`Python <pyrecipe>` and
:ref:`C++ <cpprecipe>` APIs.

Are recipes always necessary?
------------------------------

Yes. However, we provide a python :class:`single_cell_model <py_single_cell_model>`
that abstracts away the details of a recipe for simulations of  single, stand-alone
:ref:`cable cells<modelcablecell>`, which absolves the users from having to create the
recipe themselves. This is possible because the number of cells, spike targets, spike sources
and gap junction sites is fixed and known, as well as the fact that there can be no connections
or gap junctions on a single cell. The single cell model is able to fill out the details of the
recipe under the hood, and the user need only provide the cell description, and any probes they
wish to place on the cell.

Why recipes?
------------

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

    It might seem like overkill to have a separate call that returns the cell
    kind, when one could determine the kind by requesting the cell description,
    then querying the kind of the result.

    Some phases of model construction, however, only require the cell kind, and
    not the full cell description, which can be quite expensive to
    assemble; for example, a Purkinje cell model can have very complex geometry,
    a rich collection of ion channels, and thousands of synapses.

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
