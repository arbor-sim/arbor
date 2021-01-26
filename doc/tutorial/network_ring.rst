.. _tutorialnetworkring:

A ring network
==============

In this example, a small *network* of cells, arranged in a ring, will be created and the simulation distributed over multiple threads or GPUs if available. Data on execution time and memory consumption will be recorded.

.. Note::

   **Concepts covered in this example:**

   1. Building a basic :class:`arbor.cell` with a synapse site.
   2. Building a :class:`arbor.recipe` with a network of interconnected cells.
   3. Create and execute a :class:`arbor.simulation`.
   4. Running the simulation and visualizing the results,

We outline the following steps of this example:

1. Define the **cell**.
2. Define the **recipe** of the model.
3. Define the **execution context** and **domain decomposition** of the recipe.
4. Define and run the **simulation**.
5. Collect and visualize the **results**.

The cell
********

Step **(1)** shows how a simple cell with a dendrite is created. We construct the following :term:`morphology` and label the soma and dendrite:

.. figure:: ../gen-images/tutorial_network_ring_morph.svg
   :width: 400
   :align: center

   A 4-segment cell with a soma (pink) and a branched dendrite (light blue).

.. code-block:: python

   # (1) Build a segment tree
   tree = arbor.segment_tree()

   # Soma (tag=1) with radius 6 μm, modelled as cylinder of length 2*radius
   s = tree.append(arbor.mnpos, arbor.mpoint(-12, 0, 0, 6), arbor.mpoint(0, 0, 0, 6), tag=1)

   # Single dendrite (tag=3) of length 50 μm and radius 2 μm attached to soma.
   b0 = tree.append(s, arbor.mpoint(0, 0, 0, 2), arbor.mpoint(50, 0, 0, 2), tag=3)

   # Attach two dendrites (tag=3) of length 50 μm to the end of the first dendrite.
   # Radius tapers from 2 to 0.5 μm over the length of the dendrite.
   b1 = tree.append(b0, arbor.mpoint(50, 0, 0, 2), arbor.mpoint(50+50/sqrt(2), 50/sqrt(2), 0, 0.5), tag=3)
   # Constant radius of 1 μm over the length of the dendrite.
   b2 = tree.append(b0, arbor.mpoint(50, 0, 0, 1), arbor.mpoint(50+50/sqrt(2), -50/sqrt(2), 0, 1), tag=3)

   # Associate labels to tags
   labels = arbor.label_dict()
   labels['soma'] = '(tag 1)'
   labels['dend'] = '(tag 3)'

In step **(2)** we create a :term:`label` for both the root, and the site where we'll be connecting the cell to another:

.. figure:: ../gen-images/tutorial_network_ring_synapse_site.svg
   :width: 400
   :align: center

   We'll create labels for the root (red) and a synapse_site (black).

.. code-block:: python

   # (2) Mark location for synapse at the midpoint of branch 1 (the first dendrite).
   labels['synapse_site'] = '(location 1 0.5)'
   # Mark the root of the tree.
   labels['root'] = '(root)'

Step **(3)** defines a basic cell decor and creates the :ref:`cable cell <cablecell>` :ref:`description <modelcelldesc>`. In the decor, a synapse with an exponential decay (``'expsyn'``) is placed on the ``'synapse_site'`` **(4)**. A spike detector is placed at the ``'root'``.

.. code-block:: python

   # (3) Create a decor and a cable_cell
   decor = arbor.decor()

   # Put hh dynamics on soma, and passive properties on the dendrites.
   decor.paint('"soma"', 'hh')
   decor.paint('"dend"', 'pas')

   # (4) Attach a single synapse.
   decor.place('"synapse_site"', 'expsyn')

   # Attach a spike detector with threshold of -10 mV.
   decor.place('"root"', arbor.spike_detector(-10))

   cell = arbor.cable_cell(tree, labels, decor)

The recipe
**********

To create a model with multiple connected cells, we need to use a :class:`recipe <arbor.recipe>` that describes the model.
The recipe is where the different cells and the :ref:`connections <interconnectivity>` between them are defined.

Before we go there, let's first create a function that returns the above cell. This tutorial's objective is to demonstrate creating the network after all. Simply wrap the above code in a function definition, and let's add the imports while we're at it:

.. code-block:: python

   import arbor
   import pandas, seaborn #used for plotting
   from math import sqrt

   def make_cable_cell(gid):
      {{ The above cell }}
      return cell

Now that we can generate as many copies of this cell as we need, let's set the recipe up.

Step **(5)** shows a class definition for a recipe with multiple cells. Instantiating the class requires the desired number of cells as input. Compared to the :ref:`simple cell recipe <tutorialsinglecellrecipe>`, the main difference, apart from connecting the cells, is returning a variable number of cells **(6)** and returning a new cell per ``gid`` **(7)**.

Step **(8)** creates a :class:`arbor.connection` between this cell and the previous (the ``gid`` of the previous cell is ``(gid-1)%self.ncells``), with a weight of 0.1 μS and a delay of 5 ms. The two arguments to :class:`arbor.cell_member` refer to the cell ``gid`` (first argument) and the index of the synapse (second argument). Only one synapse was defined (step **4**), so the index is always 0. :func:`arbor.cable_cell.num_targets` and :func:`arbor.cable_cell.num_sources` must be set to 1: each cell has one connection coming in and one going out.

Step **(9)** creates an :class:`arbor.event_generator` on the 0th cell. The :class:`arbor.explicit_schedule` in instantiated with a list of times with unit ms, so a schedule with a period of a millisecond is created.

Step **(10)** instantiates the recipe with 4 cells.

.. code-block:: python

   # (5) Create a recipe that generates a network of connected cells.
   class ring_recipe (arbor.recipe):
      def __init__(self, ncells):
         # The base C++ class constructor must be called first, to ensure that
         # all memory in the C++ class is initialized correctly.
         arbor.recipe.__init__(self)
         self.ncells = ncells
         self.props = arbor.neuron_cable_properties()
         self.cat = arbor.default_catalogue()
         self.props.register(self.cat)

      # (6) The num_cells method that returns the total number of cells in the model
      # must be implemented.
      def num_cells(self):
         return self.ncells

      # (7) The cell_description method returns a cell
      def cell_description(self, gid):
         return make_cable_cell(gid)

      # The kind method returns the type of cell with gid.
      # Note: this must agree with the type returned by cell_description.
      def cell_kind(self, gid):
         return arbor.cell_kind.cable

      # (8) Make a ring network
      def connections_on(self, gid):
         src = (gid-1)%self.ncells
         w = 0.01
         d = 5
         return [arbor.connection(arbor.cell_member(src,0), arbor.cell_member(gid,0), w, d)]

      def num_targets(self, gid):
         return 1

      def num_sources(self, gid):
         return 1

      # (9) Attach a generator to the first cell in the ring.
      def event_generators(self, gid):
         if gid==0:
               sched = arbor.explicit_schedule([1])
               return [arbor.event_generator(arbor.cell_member(0,0), 0.1, sched)]
         return []

      def probes(self, gid):
         return [arbor.cable_probe_membrane_voltage('"root"')]

      def global_properties(self, kind):
         return self.props

   # (10) Instantiate recipe
   ncells = 4
   recipe = ring_recipe(ncells)

The execution context and the domain decomposition
**************************************************

We have almost all the components needed to create an :class:`arbor.simulation` object. First, we must create an
:class:`arbor.context` and :class:`arbor.domain_decomposition`. An execution context tells Arbor something about the
hardware on which to run the simulation. A domain decomposition defines how to distribute the different components of
a recipe over the hardware in the execution context. A follow-up tutorial will detail this further; for now we'll stick to Arbor's defaults.

Step **(11)** creates a default execution context, and uses the :func:`arbor.partition_load_balance` to create a
default domain decomposition. You can print the objects to see what defaults they produce on your system.

.. code-block:: python

   # (11) Create a default execution context and a default domain decomposition.
   context = arbor.context()
   print(context)
   decomp = arbor.partition_load_balance(recipe, context)
   print(decomp)

The simulation
**************

In step **(12)** we create the simulation. We set all spike recorders to record, and set all samplers to record at a frequency of 10 kHz. We save the handles to the samplers to be able to analyse their results later.

Step **(13)** executes the simulation for a duration of 100 ms.

.. code-block:: python

   # (12) Simulation init
   sim = arbor.simulation(recipe, decomp, context)
   sim.record(arbor.spike_recording.all)

   # Attach a sampler to the voltage probe on cell 0.
   # Sample rate of 10 sample every ms.
   handles = [sim.sample((gid, 0), arbor.regular_schedule(0.1)) for gid in range(ncells)]

   # (13) Run simulation
   sim.run(100)
   print('Simulation finished')

The results
***********

We can print the times of the spikes:

.. code-block:: python

   # Print spike times
   print('spikes:')
   for sp in sim.spikes():
      print(' ', sp)

Let's have a plot of the sampling data:

.. code-block:: python

   # Plot the recorded voltages over time.
   print("Plotting results ...")
   df_list = []
   for gid in range(ncells):
      samples, meta = sim.samples(handles[gid])[0]
      df_list.append(pandas.DataFrame({'t/ms': samples[:, 0], 'U/mV': samples[:, 1], 'Cell': f"cell {gid}"}))

   df = pandas.concat(df_list)
   seaborn.relplot(data=df, kind="line", x="t/ms", y="U/mV",hue="Cell",ci=None).savefig('network_ring_result.svg')

``sim.samples()`` takes a ``handle`` of the probe we wish to examine. It returns a list
of ``(data, meta)`` terms: ``data`` being the time and value series of the probed quantity; and
``meta`` being the location of the probe. The size of the returned list depends on the number of
discrete locations pointed to by the handle, which in this case is 1 (only one sampler), so we can take the first element.

Since we have create ``ncells`` cells, we have ``ncells`` traces. We should be seeing phase shifted traces, as the action potential propagated through the network.

We plot the results using pandas and seaborn:

.. figure:: network_ring_result.svg
    :width: 400
    :align: center


The full code
*************

You can find the full code of the example at ``python/examples/network_ring.py``.
