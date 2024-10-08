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
  :lines: 7-15

- ``N`` is the cell count of the simulation
- ``T`` is the total runtime of the simulation in ``ms``
- ``t_interval`` defines the _interval_ such that the simulation is advance in
  discrete steps ``[0, 1, 2, ...] t_interval``. Later, this will be the timescale of
  plasticity.
- ``dt`` is the numerical timestep on which cells evolve

These parameters are used here

.. literalinclude:: ../../python/example/plasticity/unconnected.py
  :language: python
  :lines: 52-62

where we run the simulation in increments of ``t_interval``.

Back to the recipe; we set a prototypical cell

.. literalinclude:: ../../python/example/plasticity/unconnected.py
  :language: python
  :lines: 23

and deliver it for all ``gid`` s

.. literalinclude:: ../../python/example/plasticity/unconnected.py
  :language: python
  :lines: 42-43

Also, each cell has an event generator attached

.. literalinclude:: ../../python/example/plasticity/unconnected.py
  :language: python
  :lines: 33-40

using a Poisson point process seeded with the cell's ``gid``. All other
parameters are set in the constructor

.. literalinclude:: ../../python/example/plasticity/unconnected.py
  :language: python
  :lines: 19-28

We also proceed to add spike recording and generate plots using a helper
function ``plot_spikes`` from ``util.py``. You can skip the following details
for now and come back later if you are interested how it works. We generate
raster plots via ``scatter``. Rates are computed by binning spikes into
``t_interval`` and the neuron id; the mean rate is the average across the
neurons smoothed using a Savitzky-Golay filter (``scipy.signal.savgol_filter``).
We plot per-neuron and mean rates.

A Randomly Wired Network
------------------------

We use inheritance to derive a new recipe that contains all the functionality of
the ```unconnected`` recipe. We then add a random connectivity matrix during
construction, fixed connection weights, and deliver the resulting connections
via the ``connections_on`` callback, with the only extra consideration of
allowing multiple connections between two neurons.

In detail, the recipe stores the connection matrix, the current
incoming/outgoing connections per neuron, and the maximum for both directions

.. literalinclude:: ../../python/example/plasticity/random_network.py
  :language: python
  :lines: 26-31

The connection matrix is used to construct connections

.. literalinclude:: ../../python/example/plasticity/random_network.py
  :language: python
  :lines: 33-38

together with the fixed connection parameters

.. literalinclude:: ../../python/example/plasticity/random_network.py
  :language: python
  :lines: 24-25

We define helper functions ``add|del_connections`` to manipulate the connection
table while upholding these invariants:

- no self-connections, i.e. ``connection[i, i] == 0``
- ``inc[i]`` the sum of ``connections[:, i]``
- no more incoming connections than allowed by ``max_inc``, i.e. ``inc[i] <= max_inc``
- ``out[i]`` the sum of ``connections[i, :]``
- no more outgoing connections than allowed by ``max_out``, i.e. ``out[i] <= max_out``

These methods return ``True`` on success and ``False`` otherwise

.. literalinclude:: ../../python/example/plasticity/random_network.py
  :language: python
  :lines: 40-54

Both are used in ``rewire`` to produce a random connection matrix

.. literalinclude:: ../../python/example/plasticity/random_network.py
  :language: python
  :lines: 56-65

We then proceed to run the simulation and plot the results as before

.. literalinclude:: ../../python/example/plasticity/random_network.py
  :language: python
  :lines: 68-79

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
