.. _tutorialbrunel:

Brunel network
==============

In this tutorial, we will build a classic Brunel network using LIF cells in Arbor, and you can compare it with the simulation using LIF cells in the NEST simulator.

.. Note::
    
    **Concepts covered in this example:**
 1. Build LIF cells by loading certain neuron parameters from the parameter file.
 2. Connect neurons in a fixed in-degree manner based on a connection probability.
 3. Add Possonian input to drive the network activity.
 4. Record spikes and plot raster plot and peristimulus time histogram (PSTH).


Start a recipe and initiate the parameters
------------------------------------------

Here, we will follow the protocol for building the recipe, which has been instructed before.

.. literalinclude:: ../../python/example/brunel/arbor_brunel.py
   :language: python
   :dedent:
   :lines: 4-59

We can find all the parameters listed in a seperate parameter files:

.. literalinclude:: ../../python/example/brunel/parameters.py
   :language: python
   :dedent:
   :lines: 1-53


We define the network size with **num_cells** and cell type with **cell_kind**. Then load all the neuron parameters to the LIF cells with **cell_description** function.

.. literalinclude:: ../../python/example/brunel/arbor_brunel.py
   :language: python
   :dedent:
   :lines: 62-81


The Brunel network is randomly sparsely connected with a fixed in-degree regulated by a connection probability (:math:`\epsilon`). We, therefore, define a function to enable random connectivity.

.. literalinclude:: ../../python/example/brunel/arbor_brunel.py
   :language: python
   :dedent:
   :lines: 93-112


To enable the network activity, we apply Poissonian input via event_generatorto to each neuron in the network. It aims to achieve a similar effect as the `Poisson_generator` in the NEST simulator.

.. literalinclude:: ../../python/example/brunel/arbor_brunel.py
   :language: python
   :dedent:
   :lines: 115-123


In the end, we build the network, run the simulation, and record the spikes.

.. literalinclude:: ../../python/example/brunel/arbor_brunel.py
   :language: python
   :dedent:
   :lines: 126-178

One can also use the code below to visualize the raster plot of the entire nework and a few selected cells, and the peristimulus time histogram (PSTH) of the entire network. The parameters used here are supposed to achieve asynchronous irregular dynamics.

.. literalinclude:: ../../python/example/brunel/analysis.py
   :language: python
   :dedent:
   :lines: 1-60


The full code
-------------
You can find the same network architecture simulated in the NEST simulator in the same repo ``python/examples/brunel/nest_brunel.py``. The average firing rate of neurons and network dynamics look similar in both cases.


References
----------
.. [1] Brunel, Journal of Computational Neuroscience 8: 183-208 (2000); `<https://link.springer.com/article/10.1023/A:1008925309027>`_.