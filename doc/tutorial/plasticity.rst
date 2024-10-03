.. _tutorial_plasticity:

In this tutorial, we are going to demonstrate how a network can be built using
plasticity and homeostatic connection rules. Despite not playing towards Arbor's
strengths, we choose a LIF (Leaky Integrate and Fire) neuron model, as we are
primarily interested in examining the required scaffolding.

We will build up the simulation in stages, starting with an unconnected network
and finishing with a dynamically built connectome.

An Unconnected Network
----------------------

Consider a collection of ``N`` LIF cells. This will be the starting point for
our exploration. For now, we set up each cell with a Poissonian input such that
it will produce spikes periodically at a low frequency.

The Python file ``01-setup.py`` is the scaffolding we will build our simulation
around and thus contains some passages that might seem redundant now, but will
be helpful in later steps.

We begin by defining the global settings

.. literalinclude:: ../../python/example/plasticity/unconnected.py
  :language: python
  :lines: 9-17

- ``T`` is the total runtime of the simulation in ``ms``
- ``dT`` defines the _interval_ such that the simulation is advance in discrete
  steps ``[0, dT, 2 dT, ..., T]``. Later, this will be the timescale of
  plasticity.
- ``dt`` is the numerical timestep on which cells evolve

These parameters are used here

.. literalinclude:: ../../python/example/plasticity/step-01.py
  :language: python

Next, we define the ``recipe`` used to describe the 'network' which is currently
unconnected.

We also proceed to add spike recording and generating raster/rate plots.

A Randomly Wired Network
------------------------

We use inheritance to derive a new recipe that contains all the functionality of
the ``unconnected`` recipe. We add a random connectivity matrix during
construction, fixed connection weights, and deliver the resulting connections
via the callback, with the only extra consideration of allowing multiple
connections between two neurons.

Adding Homeostasis
------------------

Under the homeostatic model, each cell was a setpoint for the firing rate :math:`\nu^*`
which is used to determine the creation or destruction of synaptic connections via

.. math::

   \frac{dC}{dt} = \alpha(\nu - \nu^*)

Thus we need to add some extra information to our simulation; namely the
setpoint :math:`\nu^*_i` for each neuron :math:`i` and the sensitivity parameter
:math:`\alpha`. We will also use a simplified version of the differential
equation above, namely adding/deleting exactly one connection if the difference
of observed to desired spiking frequency exceeds :math:`\pm\alpha`. This is both
for simplicity and to avoid sudden changes in the network structure.

We do this by tweaking the connection table in between calls to ``run``. In
particular, we walk the potential pairings of targets and sources in random
order and check whether the targets requires adding or removing connections. If
we find an option to fulfill that requirement, we do so and proceed to the next
target. The randomization is important here, espcially for adding connections as
to avoid biases, in particular when there are too few eglible connection
partners.
