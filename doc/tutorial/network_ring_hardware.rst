.. _tutorialnetworkringhardware:

A ring network
==============

.. todo::
   In this or the a follow-up tutorial, cells with more than one in- and outgoing connection will be described. In calling `arbor.cell_member`, source and dest indices are actually different.

In this example, we will build on the :ref:`ring network example <tutorialnetworkring>`.
We'll run a larger network on GPUs and using MPI on multiple nodes in a supercomputer.

.. Note::

   **Concepts covered in this example:**

   1. Building an :class:`arbor.context` and a :class:`arbor.domain_decomposition`
      that distributes the simulation over multiple threads or GPU.
   2. Profile the simulation using a :class:`arbor.meter_manager`.
   3.  TODO MPI
   4. Create and execute a :class:`arbor.simulation`.
   5. Running the simulation and visualizing the results,

We outline the following steps of this example:

1. Define the **cell**.
2. Define the **recipe** of the model.
3. Define the **execution context** of the model: a description of the underlying system
   on which the simulation will run.
4. Define the **meter manager** that will record execution time and memory consumption.
5. Define the **domain decomposition** of the network: how the cells are distributed on
   the different ranks of the system.
6. Define the **simulation**.
7. **Run** the simulation.
8. Collect and visualize the **results**.

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

Step **(8)** creates a :class:`arbor.connection` between this cell and the previous (the ``gid`` of the previous cell is ``(gid-1)%self.ncells``), with a weight of 0.1 μS and a delay of 5 ms. The two arguments to :class:`arbor.cell_member` refer to the cell ``gid`` (first argument) and the index of the synapse (second argument). Only one synapse was defined (step **(4)**), so the index is always 0. :func:`arbor.cable_cell.num_targets` and :func:`arbor.cable_cell.num_sources` must be set to 1: each cell has one connection coming in and one going out.

In step **(9)** an :class:`arbor.event_generator` is created on the 0th cell. The :class:`arbor.explicit_schedule` in instantiated with a list of times with unit ms, so a schedule with a period of a millisecond is created.

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

The execution context
*********************

An :ref:`execution context <modelcontext>`_ describes the hardware resources on which the simulation will run.
It contains the thread pool used to parallelise work on the local CPU, and optionally describes GPU resources
and the MPI communicator for distributed simulations. The details of the execution context can be customized by
the user. However, the ideal settings can usually be inferred from the system.

In step **(10)** we specify the number of threads in the thread pool and
let Arbor know that we have no preference for a particular GPU.

Printing out the :class:`arbor.context` object will show some information of the
capabilities of the system and the way it was configured. Most Arbor objects can
be printed out to get obtain some information about its configuration.

.. code-block:: python

   # (10) Set up the hardware context
   context = arbor.context(threads=12, gpu_id=None)
   print(context)

The meter manager
*****************

Understanding where the execution of the simulation spends time and memory is helpful for optimisation.
The :class:`arbor.meter_manager` facilitates this. You can set one up per
:class:`arbor.context` and set :func:`checkpoints <arbor.meter_manager.checkpoint>`
wherever you need them.

Step **(11)** creates the meters object.

Step **(12)** instantiates our recipe with 4 cells. Then, we set the first checkpoint.

.. code-block:: python

   # (11) Set up and start the meter manager
   meters = arbor.meter_manager()
   meters.start(context)

   # (12) Instantiate recipe
   ncells = 4
   recipe = ring_recipe(ncells)
   meters.checkpoint('recipe-create', context)

The domain decomposition
************************

The domain decomposition describes the distribution of the cells over the available computational resources.
The :class:`arbor.domain_decomposition` class can be manually created by the user, by deciding which cells
go on which ranks. Or we can use a load balancer that can partition the cells across ranks according to
some rules. Arbor provides :class:`arbor.partition_load_balance`, which, using the recipe and execution
context, creates the :class:`arbor.domain_decomposition` object for us.
:class:`arbor.partition_load_balance` can be fed some hints as to the way we
prefer to execute the simulation, if possible. For this, the :class:`arbor.partition_hint` object can be used.

Step **(13)** shows how we tell Arbor to execute the simulation of cable cells on the GPU in groups of a 1000 (GPUs consists of hundreds or thousands of parallel processing units). A dictionary that maps :class:`cell kinds <arbor.cell_kind>` to :class:`hints <arbor.partition_hint>` is what we can pass to the load balancer in the next step. Note that these are just hints, so execution might still happen differently from what you specified in your hints.

Step **(14)** sets up a load balancer for the recipe, context and hints. At this point, Arbor has configured the execution for you. By printing the initialized :class:`arbor.domain_decomposition` object produced by :class:`arbor.partition_load_balance`, you can observe how your simulation will be executed.

lastly, let's set another checkpoint.

.. code-block:: python

   # (13) Define a hint at to the execution.
   hint = arbor.partition_hint()
   hint.prefer_gpu = True
   hint.gpu_group_size = 1000
   print(hint)
   hints = {arbor.cell_kind.cable: hint}

   # (14) Domain decomp
   decomp = arbor.partition_load_balance(recipe, context, hints)
   print(decomp)

   meters.checkpoint('load-balance', context)

The simulation
**************

Finally we have the components needed to create a :class:`arbor.simulation` object.

In step **(15)** we create the simulation. We set all spike recorders to record, and set all samplers to record at a frequency of 10 kHz. We save the handles to the samplers to be able to analyse their results later. Let's have another checkpoint.

.. code-block:: python

   # (15) Simulation init
   sim = arbor.simulation(recipe, decomp, context)
   sim.record(arbor.spike_recording.all)

   # Attach a sampler to the voltage probe on cell 0.
   # Sample rate of 10 sample every ms.
   handles = [sim.sample((gid, 0), arbor.regular_schedule(0.1)) for gid in range(ncells)]

   meters.checkpoint('simulation-init', context)

The execution
*************

We can now run the simulation we just instantiated for a duration of 100ms, and measure the wall time.

.. code-block:: python

   # (16) Run simulation
   sim.run(100)
   print('Simulation finished')

   meters.checkpoint('simulation-run', context)

The results
***********

The last step is result collection. First, let's look at the profiler:

.. code-block:: python

   # Print profiling information
   print(f'{arbor.meter_report(meters, context)}')

We can print the times of the spikes:

.. code-block:: python

   # Print spike times
   print('spikes:')
   for sp in sim.spikes():
      print(' ', sp)

And let's wrap up with a plot of the sampling data:

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