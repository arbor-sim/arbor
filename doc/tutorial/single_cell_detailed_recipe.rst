.. _tutorialsinglecellswcrecipe:

A detailed single cell recipe
=============================

This example builds the same single cell model as
:ref:`the previous tutorial <tutorialsinglecellswc>`, except using a :class:`arbor.recipe`
and :class:`arbor.simulation` instead of a :class:`arbor.single_cell_model`.

.. Note::

   **Concepts covered in this example:**

   1. Building a :class:`arbor.recipe`.
   2. Building an :class:`arbor.context` and a :class:`arbor.domain_decomposition`
   3. Using the recipe, context and domain decomposition to create a :class:`arbor.simulation`
   4. Running the simulation and visualizing the results,


Recipes are an important concept in Arbor. They represent the most versatile tool
for building a complex network of cells. We will go though this example of a model
of a single cell, before using the recipe to represent more complex networks in
subsequent examples.

We outline the following steps of this example:

1. Define the **cell**. This is the same cell we have seen before.
2. Define the **recipe** of the model.
3. Define the **execution context** of the model: a description of the underlying system
   on which the simulation will run.
4. Define the **domain decomposition** of the network: how the cells are distributed on
   the different ranks of the system.
5. Define the **simulation**.
6. **Run** the simulation.
7. Collect and visualize the **results**.

The cell
********

We can immediately paste the cell description code from the
:ref:`previous example <tutorialsinglecellswc-cell>` where it is explained in detail.

.. code-block:: python

   import arbor
   from arbor import mechanism as mech

   #(1) Read the morphology from an SWC file.

   morph = arbor.load_swc_arbor("single_cell_detailed.swc")

   #(2) Create and populate the label dictionary.

   labels = arbor.label_dict()

   # Regions:

   labels['soma'] = '(tag 1)'
   labels['axon'] = '(tag 2)'
   labels['dend'] = '(tag 3)'
   labels['last'] = '(tag 4)'

   labels['all'] = '(all)'

   labels['gt_1.5'] = '(radius-ge (region "all") 1.5)'
   labels['custom'] = '(join (region "last") (region "gt_1.5"))'

   # Locsets:

   labels['root']     = '(root)'
   labels['terminal'] = '(terminal)'
   labels['custom_terminal'] = '(restrict (locset "terminal") (region "custom"))'
   labels['axon_terminal'] = '(restrict (locset "terminal") (region "axon"))'

   # (3) Create and populate the decor.

   decor = arbor.decor()

   # Set the default properties of the cell (this overrides the model defaults)

   decor.set_property(Vm =-55)

   # Override the cell defaults.

   decor.paint('"custom"', tempK=270)
   decor.paint('"soma"',   Vm=-50)

   # Paint density mechanisms.

   decor.paint('"all"', 'pas')
   decor.paint('"custom"', 'hh')
   decor.paint('"dend"',  mech('Ih', {'gbar': 0.001}))

   # Place stimuli and spike detectors.

   decor.place('"root"', arbor.iclamp(10, 1, current=2), 'iclamp0')
   decor.place('"root"', arbor.iclamp(30, 1, current=2), 'iclamp1')
   decor.place('"root"', arbor.iclamp(50, 1, current=2), 'iclamp2')
   decor.place('"axon_terminal"', arbor.spike_detector(-10), 'detector')

   # Set cv_policy

   soma_policy = arbor.cv_policy_single('"soma"')
   dflt_policy = arbor.cv_policy_max_extent(1.0)
   policy = dflt_policy | soma_policy
   decor.discretization(policy)

   # (4) Create the cell.

   cell = arbor.cable_cell(morph, labels, decor)

We will add one more thing to this section. We will create the voltage probe at the "custom_terminal" locset.
In the previous example, this probe was registered directly using the :class:`arbor.single_cell_model` object.
Now it has to be explicitly created and registered in the recipe.

.. _tutorialsinglecellswcrecipe-probe:
.. code-block:: python

   probe = arbor.cable_probe_membrane_voltage('"custom_terminal"')

The recipe
**********

The :class:`arbor.single_cell_model` of the previous example created a :class:`arbor.recipe` under
the hood, and abstracted away the details so we were unaware of its existence. In this example, we will
examine the recipe in detail: how to create one, and why it is needed.

.. code-block:: python

   # (1) Create a class that inherits from arbor.recipe
   class single_recipe (arbor.recipe):

       # (2) Define the class constructor
       def __init__(self, cell, probes):
           # The base C++ class constructor must be called first, to ensure that
           # all memory in the C++ class is initialized correctly.
           arbor.recipe.__init__(self)
           self.the_cell = cell
           self.the_probes = probes

           self.the_cat = arbor.default_catalogue()
           self.the_cat.extend(arbor.allen_catalogue(), "")

           self.the_props = arbor.cable_global_properties()
           self.the_props.set_property(Vm=-65, tempK=300, rL=35.4, cm=0.01)
           self.the_props.set_ion(ion='na', int_con=10,   ext_con=140, rev_pot=50, method='nernst/na')
           self.the_props.set_ion(ion='k',  int_con=54.4, ext_con=2.5, rev_pot=-77)
           self.the_props.set_ion(ion='ca', int_con=5e-5, ext_con=2, rev_pot=132.5)

           self.the_props.register(self.the_cat)

       # (3) Override the num_cells method
       def num_cells(self):
           return 1

       # (4) Override the cell_kind method
       def cell_kind(self, gid):
           return arbor.cell_kind.cable

       # (5) Override the cell_description method
       def cell_description(self, gid):
           return self.the_cell

       # (6) Override the probes method
       def probes(self, gid):
           return self.the_probes

       # (7) Override the connections_on method
       def connections_on(self, gid):
           return []

       # (8) Override the gap_junction_on method
       def gap_junction_on(self, gid):
           return []

       # (9) Override the event_generators method
       def event_generators(self, gid):
           return []

       # (10) Overrode the global_properties method
       def global_properties(self, gid):
          return self.the_props

Let's go through the recipe point by point.

Step **(1)** creates a ``single_recipe`` class that inherits from :class:`arbor.recipe`. The base recipe
implements all the methods defined above with default values except :meth:`arbor.recipe.num_cells`,
:meth:`arbor.recipe.cell_kind` and :meth:`arbor.recipe.cell_description` which always have to be implemented
by the user. The :meth:`arbor.recipe.global_properties` also needs to be implemented for
:class:`arbor.cell_kind.cable` cells. The inherited recipe can implement any number of additional methods and
have any number of instance or class variables.

Step **(2)** defines the class constructor. In this case, we pass a ``cell`` and a set of ``probes`` as
arguments. These will be used to initialize the instance variables ``self.the_cell`` and ``self.the_probes``,
which will be used in the overloaded ``cell_description`` and ``get_probes`` methods. Before variable
initialization, we call the base C++ class constructor ``arbor.recipe.__init__(self)``. This ensures correct
initialization of memory in the C++ class.

We also create the ``self.the_cat`` variable and set it to arbor's default mechanism catalogue. This will expose
the *hh* and *pas* mechanisms but not the *Ih* mechanism, which is present in the allen catalogue. To be able
to use *Ih*, we extend ``self.the_cat`` to include the allen catalogue.

Finally we create the ``self.the_props`` variable. This will hold the global properties of the model, which apply
to all the cells in the network. Initially it is empty. We set all the properties of the system similar to
what we did in the :ref:`previous example <tutorialsinglecellswc-gprop>`. One last important step is to register
``self.the_cat`` with ``self.the_props``.

.. Note::

   The mechanism catalogue needs to live in the recipe as an instance variable. Its lifetime needs to extend
   to the entire duration of the simulation.

Step **(3)** overrides the :meth:`arbor.recipe.num_cells` method. It takes no arguments. We simply return 1,
as we are only simulating one cell in this example.

Step **(4)** overrides the :meth:`arbor.recipe.cell_kind` method. It takes one argument: ``gid``.
Given the gid, this method returns the kind of the cell. Our defined cell is a
:class:`arbor.cell_kind.cable`, so we simply return that.

Step **(5)** overrides the :meth:`arbor.recipe.cell_description` method. It takes one argument: ``gid``.
Given the gid, this method returns the cell description which is the cell object passed to the constructor
of the recipe. We return ``self.the_cell``.

Step **(6)** overrides the :meth:`arbor.recipe.get_probes` method. It takes one argument: ``gid``.
Given the gid, this method returns all the probes on the cell. The probes can be of many different kinds
measuring different quantities on different locations of the cell. We pass these probes explicitly to the recipe
and they are stored in ``self.the_probes``, so we return that variable.

Step **(7)** overrides the :meth:`arbor.recipe.connections_on` method. It takes one argument: ``gid``.
Given the gid, this method returns all the connections ending on that cell. These are typically synapse
connections from other cell *sources* to specific *targets* on the cell with id ``gid``. Since we are
simulating a single cell, and self-connections are not possible, we return an empty list.

Step **(8)** overrides the :meth:`arbor.recipe.gap_junctions_on` method. It takes one argument: ``gid``.
Given the gid, this method returns all the gap junctions on that cell. Gap junctions require 2 separate cells.
Since we are simulating a single cell, we return an empty list.

Step **(9)** overrides the :meth:`arbor.recipe.event_generators` method. It takes one argument: ``gid``.
Given the gid, this method returns *event generators* on that cell. These generators trigger events (or
spikes) on specific *targets* on the cell. They can be used to simulate spikes from other cells, to kick-start
a simulation for example. Our cell uses a current clamp as a stimulus, and has no targets, so we return
an empty list.

Step **(10)** overrides the :meth:`arbor.recipe.global_properties` method. It takes one argument: ``kind``.
This method returns the default global properties of the model which apply to all cells in the network of
that kind. We return ``self.the_props`` which we defined in step **(1)**.

Now we can instantiate a ``single_recipe`` object using the ``cell`` and ``probe`` we created in the
previous section:

.. code-block:: python

   # Instantiate recipe
   # Pass the probe in a list because that it what single_recipe expects.
   recipe = single_recipe(cell, [probe])

The execution context
*********************

An :ref:`execution context <modelcontext>` describes the hardware resources on which the simulation will run.
It contains the thread pool used to parallelise work on the local CPU, and optionally describes GPU resources
and the MPI communicator for distributed simulations. In the previous
examples, the :class:`arbor.single_cell_model` object created the execution context :class:`arbor.context`
behind the scenes.

The details of the execution context can be customized by the user. We may specify the number of threads
in the thread pool; determine the id of the GPU to be used; or create our own MPI communicator. However,
the ideal settings can usually be inferred from the system, and Arbor can do that with a simple command.

.. code-block:: python

   context = arbor.context()

The domain decomposition
************************

The domain decomposition describes the distribution of the cells over the available computational resources.
The :class:`arbor.single_cell_model` also handled that without our knowledge in the previous examples.
Now, we have to define it ourselves.

The :class:`arbor.domain_decomposition` class can be manually created by the user, by deciding which cells
go on which ranks. Or we can use a load balancer that can partition the cells across ranks according to
some rules. Arbor provides :class:`arbor.partition_load_balance`, which, using the recipe and execution
context, creates the :class:`arbor.domain_decomposition` object for us.

Our example is a simple one, with just one cell. We don't need any sophisticated partitioning algorithms, so
we can use the load balancer, which does a good job distributing simple networks.

.. code-block:: python

   domains = arbor.partition_load_balance(recipe, context)

The simulation
**************

Finally we have the 3 components needed to create a :class:`arbor.simulation` object.

.. code-block:: python

   sim = arbor.simulation(recipe, domains, context)

Before we run the simulation, however, we need to register what results we expect once execution is over.
This was handled by the :class:`arbor.single_cell_model` object in the previous example.

We would like to get a list of the spikes on the cell during the runtime of the simulation, and we would like
to plot the voltage registered by the probe on the "custom_terminal" locset.

.. code-block:: python

   # Instruct the simulation to record the spikes
   sim.record(arbor.spike_recording.all)

   # Instruct the simulation to sample the probe (0, 0)
   # at a regular schedule with period = 0.02 ms (50 kHz)
   probe_id = arbor.cell_member(0,0)
   handle = sim.sample(probe_id, arbor.regular_schedule(0.02))

The lines handling probe sampling warrant a second look. First, we declared ``probe_id`` to be a
:class:`arbor.cell_member`, with :class:`arbor.cell_member.gid` = 0 and :class:`arbor.cell_member.index` = 0.
This variable serves as a global identifier of a probe on a cell, namely the first declared probe on the
cell with gid = 0, which is id of the :ref:`only probe <tutorialsinglecellswcrecipe-probe>` we created on
the only cell in the model.

Next, we instructed the simulation to sample ``probe_id`` at a frequency of 50 kHz. That function returns a
``handle`` which we will use to extract the results of the sampling after running the simulation.

The execution
*************

We can now run the simulation we just instantiated for a duration of 100 ms with a time step of 0.025 ms.

.. code-block:: python

   sim.run(tfinal=100, dt=0.025)


The results
***********

The last step is result collection. We instructed the simulation to record the spikes on the cell, and
to sample the probe.

We can print the times of the spikes:

.. code-block:: python

   spikes = sim.spikes()

   # Print the number of spikes.
   print(len(spikes), 'spikes recorded:')

   # Print the spike times.
   for s in spikes:
       print(s)

The probe results, again, warrant some more explanation:

.. code-block:: python

   data = []
   meta = []
   for d, m in sim.samples(handle):
      data.append(d)
      meta.append(m)

``sim.samples()`` takes a ``handle`` of the probe we wish to examine. It returns a list
of ``(data, meta)`` terms: ``data`` being the time and value series of the probed quantity; and
``meta`` being the location of the probe. The size of the returned list depends on the number of
discrete locations pointed to by the handle. We placed the probe on the "custom_terminal" locset which is
represented by 2 locations on the morphology. We therefore expect the length of ``sim.samples(handle)``
to be 2.

We plot the results using pandas and seaborn as we did in the previous example, and expect the same results:

.. code-block:: python

   df = pandas.DataFrame()
   for i in range(len(data)):
       df = df.append(pandas.DataFrame({'t/ms': data[i][:, 0], 'U/mV': data[i][:, 1], 'Location': str(meta[i])}))
   seaborn.relplot(data=df, kind="line", x="t/ms", y="U/mV",hue="Location",col="Variable",ci=None).savefig('single_cell_detailed_recipe_result.svg')

The following plot is generated. Identical to the plot of the previous example.

.. figure:: single_cell_detailed_result.svg
    :width: 400
    :align: center


The full code
*************

You can find the full code of the example at ``python/examples/single_cell_detailed_recipe.py``.
