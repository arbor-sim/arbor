.. _modelrecipe:

Recipes
===============

An Arbor *recipe* is a description of a model. The recipe is queried during the model
building phase to provide cell information, such as:

  * the number of cells in the model;
  * the type of a cell;
  * a description of a cell, e.g. with soma, synapses, detectors, stimuli;

and optionally, e.g.:

  * the number of spike targets;
  * the number of spike sources;
  * incoming network connections from other cells terminating on a cell.

General Best Practices
----------------------

.. topic:: Think of the cells

    When formulating a model, think cell-first, and try to formulate the model and
    the associated workflow from a cell-centered perspective. If this isn't possible,
    please contact the developers, because we would like to develop tools that help
    make this simpler.

.. topic:: Be reproducible

    Arbor is designed to give reproduceable results when the same model is run on a
    different number of MPI ranks or threads, or on different hardware (e.g. GPUs).
    This only holds when a recipe provides a reproducible model description, which
    can be a challenge when a description uses random numbers, e.g. to pick incoming
    connections to a cell from a random subset of a cell population.
    To get a reproduceable model, use the cell global identifyer `gid` to seed random number generators.

Mechanisms
----------------------
The description of multi-compartment cells also includes the specification of ion channel and synapse dynamics.
In the recipe, these specifications are called *mechanisms*.
Implementations of mechanisms are either hand-coded or a translator (modcc) is used to compile a
subset of NEURONs mechanism specification language NMODL.

Examples
    Common examples are the *passive/ leaky integrate-and-fire* model, the *Hodgkin-Huxley* mechanism, the *(double-)exponential synapse* model, or the *Natrium current* model for an axon.

The detailed documentations and specific best practices for C++ recipes can be found in :ref:`cpprecipe` and in :ref:`pyrecipe` covering python recipes.
