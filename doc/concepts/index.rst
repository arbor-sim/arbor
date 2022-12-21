.. _modelintro:

Concepts overview
=================

Arbor is a library that lets you to model neural networks with morphologically
detailed cells; which it then executes the resulting simulation on a variety of
hardware. The execution can optionally be configured in high detail but comes
with sensible defaults.

.. raw:: html
   :file: index-diag-1.html

To learn how to use Arbor, it is helpful to understand some of its concepts.
Arbor's design aims to enable scalability through abstraction.
To achieve this, Arbor makes a distinction between the **description** of a model, and the
**execution** of a model:
a *recipe* describes a model, and a *simulation* is an executable instantiation of a model.

To be able to simulate a model, three basic steps need to be considered:

1. Describe the model by defining a recipe;
2. Define the computational resources available to execute the model;
3. Initiate and execute a simulation of the recipe on the chosen hardware resources.

The Python front-end further abstracts away some of these steps for single cell models, where users only need to
describe the cell and simulation; and the details of the recipe and computational resources construction are
handled under the hood. Generally speaking though, these 3 steps are the building blocks of an Arbor application.

:ref:`Recipes <modelrecipe>` represent a set of neuron constructions and connections with :ref:`mechanisms <mechanisms>` specifying
ion channel and synapse dynamics in a cell-oriented manner. This has the advantage that cell data can be initiated
in parallel.

A cell represents the smallest unit of computation and forms the smallest unit of work distributed across processes.
Arbor has built-in support for different :ref:`cell types <modelcells>`, which can be extended by adding new cell
types to the C++ cell group interface.

:ref:`modelsimulation` manage the instantiation of the model and the scheduling of spike exchange as well as the
integration for each cell group. A cell group represents a collection of cells of the same type computed together
on the GPU or CPU. The partitioning into cell groups is provided by :ref:`modeldomdec` which describes the distribution
of the model over the locally available computational resources.

In order to visualize the result of detected spikes a spike recorder can be used, and to analyse Arbor's performance a
meter manager is available.

:ref:`probesample` shows how to extract data from simulations.
