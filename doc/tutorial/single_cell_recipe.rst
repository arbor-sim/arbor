.. _tutorialsinglecellrecipe:

A single cell recipe
--------------------

This example aims to build the same single cell model as
:ref:`a previous tutorial <tutorialsinglecellswc>`, except using a *recipe* instead
of a *single_cell_model*.

Recipes are an important concept in Arbor. They represent the most versatile tool
for building a complex network of cells. We start with this example of a model
of a single cell, before using the recipe to represent more complex networks in
subsequent examples.

We outline the following steps of this example:

1. Define the *cell*. This is the same cell we have seen before.
2. Define the *recipe* of the model.
3. Define the *execution context* of the model: a description of the underlying system
   on which the simulation will run.
4. Define the *domain decomposition* of the network: how the cells are distributed on
   the different ranks of the system.
5. Define the *simulation*.
6. *Run* the simulation.
7. Collect and visualize the *results*.

The cell
********

We can immediately paste the cell description code from the
:ref:`previous example <tutorialsinglecellswc-cell>` where it is explained in detail.

.. code-block:: python

   import arbor
   from arbor import mechanism as mech

   #(1) Read the morphology from an SWC file.

   morph = arbor.load_swc_arbor("morph.swc")

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

   # Set the default properties.

   decor.set_property(Vm =-55, tempK=300, rL=35.4, cm=0.01)
   decor.set_ion('na', int_con=10,   ext_con=140, rev_pot=50, method='nernst/na')
   decor.set_ion('k',  int_con=54.4, ext_con=2.5, rev_pot=-77)

   # Override the defaults.

   decor.paint('"custom"', tempK=270)
   decor.paint('"soma"',   Vm=-50)

   # Paint density mechanisms.

   decor.paint('"all"', 'pas')
   decor.paint('"custom"', 'hh')
   decor.paint('"dend"',  mech('Ih', {'gbar': 0.001}))

   # Place stimuli and spike detectors.

   decor.place('"root"', arbor.iclamp(10, 1, current=2))
   decor.place('"root"', arbor.iclamp(30, 1, current=2))
   decor.place('"root"', arbor.iclamp(50, 1, current=2))
   decor.place('"axon_terminal"', arbor.spike_detector(-10))

   # Set cv_policy

   soma_policy = arbor.cv_policy_single('"soma"')
   dflt_policy = arbor.cv_policy_max_extent(1.0)
   policy = dflt_policy | soma_policy
   decor.discretization(policy)

   # (4) Create the cell.

   cell = arbor.cable_cell(morph, labels, decor)

We will add one more thing to this section. We will create the voltage probe at the "custom_terminal" locset.
That was registered directly using the :class:`arbor.single_cell_model` object in the previous example.
Now it has to be explicitly created and registered in the recipe.

.. code-block:: python
   probe = arbor.cable_probe_membrane_voltage('"custom_terminal"')

The recipe
**********

The :class:`arbor.single_cell_model` of the previous example created a recipe under the hood, and
abstracted away the details so we were unaware of its existence. In this example, we will
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

       # (3) Override the num_cells method
       def num_cells(self):
           return 1

       # (4) Override the num_sources method
       def num_sources(self, gid):
           return 1

       # (5) Override the num_targets method
       def num_targets(self, gid):
           return 0

       # (6) Override the num_targets method
       def cell_kind(self, gid):
           return arbor.cell_kind.cable

       # (7) Override the cell_description method
       def cell_description(self, gid):
           return self.the_cell

       # (8) Override the get_probes method
       def get_probes(self, gid):
           return self.the_probes

       # (9) Override the connections_on method
       def connections_on(self, gid):
           return []

       # (10) Override the gap_junction_on method
       def gap_junction_on(self, gid):
           return []

       # (11) Override the event_generators method
       def event_generators(self, gid):
           return []

Let's go through the recipe point by point.

Step **(1)** creates a ``single_recipe`` class that inherits from :class:`arbor.recipe`. The base recipe
implements all the methods defined above with default values except :meth:`arbor.recipe.num_cells`,
:meth:`arbor.recipe.cell_kind` and :meth:`arbor.recipe.cell_description` which always have to be implemented
by the user. The inherited recipe can implement any number of additional methods and have any number of
instance or class variables.

Step **(2)** defines the class constructor. In this case, we pass a ``cell`` and a set of ``probes`` as
arguments. These will be used to initialize the instance variables ``self.the_cell`` and ``self.the_probes``,
which will be used in the overloaded ``cell_description`` and ``get_probes`` methods. Before variable
initialization, we call the base C++ class constructor ``arbor.recipe.__init__(self)``. This ensures correct
initialization of memory in the C++ class.

Step **(3)** overrides the :meth:`arbor.recipe.num_cells` method. It takes 0 arguments. We simply return 1,
as we are only simulating one cell in this example.

Step **(4)** overrides the :meth:`arbor.recipe.num_sources` method. It takes one argument: ``gid``.
Given this global ID of a cell, the method will return the number of spike *sources* on the cell. We have defined
our cell with one spike detector, on one location on the morphology, so we return 1.

Step **(5)** overrides the :meth:`arbor.recipe.num_targets` method. It takes one argument: ``gid``.
Given the gid, this method returns the number of *targets* on the cell. These are typically synapses on the cell
that are capable of receiving events from other cells. We have defined our cell with 0 synapses, so we return 0.

Step **(6)** overrides the :meth:`arbor.recipe.cell_kind` method. It takes one argument: ``gid``.
Given the gid, this method returns the kind of the cell. Our defined cell is a
:class:`arbor.cell_kind.cable`, so we simply return that.

Step **(7)** overrides the :meth:`arbor.recipe.cell_description` method. It takes one argument: ``gid``.
Given the gid, this method returns the cell description which is the cell object passed to the constructor
of the recipe. We return ``self.the_cell``.

Step **(8)** overrides the :meth:`arbor.recipe.get_probes` method. It takes one argument: ``gid``.
Given the gid, this method returns all the probes on the cell. The probes can be of many different kinds
measuring different quantities on different locations of the cell. We pass these probes explicitly to the recipe
and they are stored in ``self.the_probes``, so we return that variable.

Step **(9)** overrides the :meth:`arbor.recipe.connections_on` method. It takes one argument: ``gid``.
Given the gid, this method returns all the connections ending on that cell. These are typically synapse
connections from other cell *sources* to specific *targets* on the cell with id ``gid``. Since we are
simulating a single cell, and self-connections are not possible, we return an empty list.

Step **(10)** overrides the :meth:`arbor.recipe.gap_junctions_on` method. It takes one argument: ``gid``.
Given the gid, this method returns all the gap junctions on that cell. Gap junctions require 2 separate cells.
Since we are simulating a single cell, we return an empty list.

Step **(11)** overrides the :meth:`arbor.recipe.event_generators` method. It takes one argument: ``gid``.
Given the gid, this method returns *event generators* on that cell. These generators trigger events (or
spikes) on specific *targets* on the cell. They can be used to simulate spikes from other cells, to kick-start
a simulation for example. Our cell uses a current clamp as a stimulus, and has no targets, so we return
an empty list.

.. Note::

   You may wonder why the methods:  :meth:`arbor.recipe.num_sources`, :meth:`arbor.recipe.num_targets`,
   and :meth:`arbor.recipe.cell_kind` are required, since they can be inferred by examining the cell description.
   The recipe was designed to allow building simulations efficiently in a distributed system with minimum
   communication. Some parts of the model initialization require only the cell kind, or the number of
   sources and targets, not the full cell description which can be quite expensive to build. Providing these
   descriptions separately saves time and resources for the user. More information on the recipe can be found
   :ref:`here <modelrecipe>`.

Now we can intantiate a ``single_recipe`` object using the ``cell`` and ``probe`` we created in the
previous section:

.. code-block:: python

   # Instantiate recipe
   # Pass the probe in a list because that it what single_recipe expects.
   recipe = single_recipe(cell, [probe])

The execution context
*********************

The execution context contains all system-specific information needed by the simulation: it contains the
thread pool which handles multi-threaded optimization on the CPU; it knows the relevant GPU attributes
if a GPU is available; and it holds the MPI communicator for distributed simulations. In the previous
examples, the :class:`arbor.single_cell_model` object created the execution context :class:`arbor.context`
behind the scenes.

The details of the execution context can be customized by the user. We may specify the number of threads
in the thread pool; determine the id of the GPU to be used; or create our own MPI communicator. However,
the ideal settings can usually be inferred from the system, and arbor can do that with a simple command.

.. code-block:: python

   context = arbor.context()

The domain decomposition
************************

The domain decomposition describes the distribution of the cells over the available computational resources.
The :class:`arbor.single_cell_model` also handled that without our knowledge in the previous examples.
Now, we have to define it ourselves.

The :class:`arbor.domain_decomposition` class can be manually created by the user, by deciding which cells
go on which ranks. Or we can use a load balancer that can partition the cells across ranks according to
some heuristics. Arbor provides :class:`arbor.partition_load_balance`, which, using the recipe and execution
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
This was handled by the :class:`arbor.single_cell_model` object in previous examples.

We would like to get a list of the spikes on the cell during the runtime of the simulation, and we would like
to plot the voltage registered by the probe on the "custom_terminal" locset.

.. code-block:: python

   # Instruct the simulation to record the spikes
   sim.record(arbor.spike_recording.all)

   # Instruct the simulation to sample the probe (0, 0)
   # at a regular schedule with period = 0.02 ms (50000 Hz)
   probe_id = arbor.cell_member(0,0)
   handle = sim.sample(probe_id, arbor.regular_schedule(0.02))

The lines handling probe sampling warrant a second look. First, we declared ``probe_id`` to be a
:class:`arbor.cell_member`, with :class:`arbor.cell_member.gid` = 0 and :class:`arbor.cell_member.index` = 0.
This variable serves as a global identifier of a probe on a cell, namely the first declared probe on the
cell with gid = 0 which is the only cell in the model.

Next, we instructed the simulation to sample ``probe_id`` at a frequency of 50KHz. That function returns a
``handle`` which we will use to extract the results of the sampling after running the simulation.

The execution
*************

We can now run the simulation we just instantiated for a duration of 100ms with a time step of 0.025 ms.

.. code-block:: python

   sim.run(tfinal=100, dt=0.025)


The results
***********

The last step is result collection. We instructed the simulation to record the spikes on the cell, and
the sample of the probe.

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
   seaborn.relplot(data=df, kind="line", x="t/ms", y="U/mV",hue="Location", ci=None).savefig('single_cell_recipe_result.svg')

The full code
*************

.. code-block:: python

   import arbor
   import pandas
   import seaborn
   from arbor import mechanism as mech

   #(1) Creat a cell.

   # Create the morphology

   morph = arbor.load_swc_arbor("morph.swc")

   # Create and populate the label dictionary.

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

   # Create and populate the decor.

   decor = arbor.decor()

   # Set the default properties.

   decor.set_property(Vm =-55, tempK=300, rL=35.4, cm=0.01)
   decor.set_ion('na', int_con=10,   ext_con=140, rev_pot=50, method='nernst/na')
   decor.set_ion('k',  int_con=54.4, ext_con=2.5, rev_pot=-77)

   # Override the defaults.

   decor.paint('"custom"', tempK=270)
   decor.paint('"soma"',   Vm=-50)

   # Paint density mechanisms.

   decor.paint('"all"', 'pas')
   decor.paint('"custom"', 'hh')
   decor.paint('"dend"',  mech('Ih', {'gbar': 0.001}))

   # Place stimuli and spike detectors.

   decor.place('"root"', arbor.iclamp(10, 1, current=2))
   decor.place('"root"', arbor.iclamp(30, 1, current=2))
   decor.place('"root"', arbor.iclamp(50, 1, current=2))
   decor.place('"axon_terminal"', arbor.spike_detector(-10))

   # Set cv_policy

   soma_policy = arbor.cv_policy_single('"soma"')
   dflt_policy = arbor.cv_policy_max_extent(1.0)
   policy = dflt_policy | soma_policy
   decor.discretization(policy)

   # Create a cell

   cell = arbor.cable_cell(morph, labels, decor)

   # (2) Declare a probe.

   probe = arbor.cable_probe_membrane_voltage('"custom_terminal"')

   # (3) Create a recipe class and instantiate a recipe

   class single_recipe (arbor.recipe):

       def __init__(self, cell, probes):
           # The base C++ class constructor must be called first, to ensure that
           # all memory in the C++ class is initialized correctly.
           arbor.recipe.__init__(self)
           self.the_cell = cell
           self.the_probes = probes

       def num_cells(self):
           return 1

       def num_sources(self, gid):
           return 1

       def num_targets(self, gid):
           return 0

       def cell_kind(self, gid):
           return arbor.cell_kind.cable

       def cell_description(self, gid):
           return self.the_cell

       def get_probes(self, gid):
           return self.the_probes

       def connections_on(self, gid):
           return []

       def gap_junction_on(self, gid):
           return []

       def event_generators(self, gid):
           return []

   recipe = single_recipe(cell, [probe])

   # (4) Create an execution context

   context = arbor.context()

   # (5) Create a domain decomposition

   domains = arbor.partition_load_balance(recipe, context)

   # (6) Create a simulation

   sim = arbor.simulation(recipe, domains, context)

   # Instruct the simulation to record the spikes and sample the probe

   sim.record(arbor.spike_recording.all)

   probe_id = arbor.cell_member(0,0)
   handle = sim.sample(probe_id, arbor.regular_schedule(0.02))

   # (7) Run the simulation

   sim.run(tfinal=100, dt=0.025)

   # (8) Print or display the results

   spikes = sim.spikes()
   print(len(spikes), 'spikes recorded:')
   for s in spikes:
       print(s)

   data = []
   meta = []
   for d, m in sim.samples(handle):
      data.append(d)
      meta.append(m)

   df = pandas.DataFrame()
   for i in range(len(data)):
       df = df.append(pandas.DataFrame({'t/ms': data[i][:, 0], 'U/mV': data[i][:, 1], 'Location': str(meta[i])}))
   seaborn.relplot(data=df, kind="line", x="t/ms", y="U/mV",hue="Location", ci=None).savefig('single_cell_recipe_result.svg')