.. _modelintro:

Overview
=========
Arbor's design aims to enable scalability through abtraction.

To achieve this, Arbor makes a distinction between the **description** of a model, and the
**execution** of a model:
a *recipe* describes a model, and a *simulation* is an executable instatiation of a model.

To be able to simulate a model, three basic steps need to be considered:

* first, describe the neuron model by defining a recipe;
* then, define the computational resources available to execute the model;
* finally, initiate and execute a simulation of the recipe on the chosen hardware resources.

.. topic:: Concepts

    :ref:`modelrecipe` represent a set of neuron constructions and connections with *mechanisms* specifying ion channel and synapse dynamics in a cell-oriented manner. This has the advantage that cell data can be initiated in parallel.

    A cell represents the smallest unit of computation and forms the smallest unit of work distributed across processes. Different :ref:`modelcells` can be utilized.

    :ref:`modelsimulation` manage the instantiation of the model and the scheduling of spike exchange as well as the integration for each cell group. A cell group represents a collection of cells of the same type computed together on the GPU or CPU. The partitioning into cell groups is provided by :ref:`modeldomdec` which describes the distribution of the model over the locally available computational resources.

    In order to visualise the result of detected spikes a spike recorder can be used and to analyse Arbor's performance a meter manager is available.
