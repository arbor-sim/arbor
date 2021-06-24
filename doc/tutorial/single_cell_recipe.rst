.. _tutorialsinglecellrecipe:

A simple single cell recipe
===========================

This example builds the same single cell model as
:ref:`the previous tutorial <tutorialsinglecell>`, except using a :class:`arbor.recipe`
and :class:`arbor.simulation` instead of a :class:`arbor.single_cell_model`.

.. Note::

   **Concepts covered in this example:**

   1. Building a :class:`arbor.recipe`.
   2. Using the recipe, context and domain decomposition to create a :class:`arbor.simulation`
   3. Running the simulation and visualizing the results.

The cell
--------

We can immediately paste the cell description code from the
:ref:`previous example <tutorialsinglecell-cell>` where it is explained in detail.

.. code-block:: python

    import arbor

    # (1) Create a morphology with a single (cylindrical) segment of length=diameter=6 μm
    tree = arbor.segment_tree()
    tree.append(arbor.mnpos, arbor.mpoint(-3, 0, 0, 3), arbor.mpoint(3, 0, 0, 3), tag=1)

    # (2) Define the soma and its midpoint
    labels = arbor.label_dict({'soma':   '(tag 1)',
                              'midpoint': '(location 0 0.5)'})

    # (3) Create cell and set properties
    decor = arbor.decor()
    decor.set_property(Vm=-40)
    decor.paint('"soma"', 'hh')
    decor.place('"midpoint"', arbor.iclamp( 10, 2, 0.8), 'iclamp')
    decor.place('"midpoint"', arbor.spike_detector(-10), 'detector')
    cell = arbor.cable_cell(tree, labels, decor)

The recipe
----------

The :class:`arbor.single_cell_model` of the previous example created a :class:`arbor.recipe` under
the hood, and abstracted away the details so we were unaware of its existence.

Creating an analogous recipe starts with creating a class that inherits from :class:`arbor.recipe`
and overrides and implements some of :class:`arbor.recipe` methods. Not all methods
have to be overridden, but some will always have to be, such as :meth:`arbor.recipe.num_cells`.
It returns `0` by default and models without cells are quite boring!

.. code-block:: python

    # (4) Define a recipe for a single cell and set of probes upon it.
    # This constitutes the corresponding generic recipe version of
    # `single_cell_model.py`.

    class single_recipe (arbor.recipe):
        def __init__(self, cell, probes):
            # (4.1) The base C++ class constructor must be called first, to ensure that
            # all memory in the C++ class is initialized correctly.
            arbor.recipe.__init__(self)
            self.the_cell = cell
            self.the_probes = probes
            self.the_props = arbor.neuron_cable_properties()
            self.the_cat = arbor.default_catalogue()
            self.the_props.register(self.the_cat)

        def num_cells(self):
            # (4.2) Override the num_cells method
            return 1

        def cell_kind(self, gid):
            # (4.3) Override the cell_kind method
            return arbor.cell_kind.cable

        def cell_description(self, gid):
            # (4.4) Override the cell_description method
            return self.the_cell

        def probes(self, gid):
            # (4.5) Override the probes method
            return self.the_probes

        def global_properties(self, kind):
            # (4.6) Override the global_properties method
            return self.the_props

    # (5) Instantiate recipe with a voltage probe located on "midpoint".

    recipe = single_recipe(cell, [arbor.cable_probe_membrane_voltage('"midpoint"')])

Step **(4)** describes the recipe that will reflect our single cell model.

Step **(4.1)** defines the class constructor. It can take any shape you need, but it
is important to call base class' constructor. If the overridden methods of the class
need to return an object, it may be a good idea to have the returned object be a
member of the class. With this constructor, we could easily change the cell and probes
of the model, should we want to do so. Here we initialize the cell properties to match
Neuron's defaults using Arbor's built-in :meth:`arbor.neuron_cable_properties` and
extend with Arbor's own :meth:`arbor.default_catalogue`.

Step **(4.2)** defines that this model has one cell.

Step **(4.3)** returns :class:`arbor.cell_kind.cable`, the :class:`arbor.cell_kind`
associated with the cable cell defined above. If you mix multiple cell kinds and
descriptions in one recipe, make sure a particular ``gid`` returns matching cell kinds
and descriptions.

Step **(4.4)** returns the cell description passed in on class initialisation. If we
were modelling multiple cells of different kinds, we would need to make sure that the
cell returned by :meth:`arbor.recipe.cell_description` has the same cell kind as
returned by :meth:`arbor.recipe.cell_kind` for every :gen:`gid`.

Step **(4.5)** returns the probes passed in at class initialisation.

Step **(4.6)** returns the properties that will be applied to all cells of that kind in the model.

More methods can be overridden if your model requires that, see :class:`arbor.recipe` for options.

Step **(5)** instantiates the recipe with the cable cell described earlier, and a single voltage probe located at "midpoint".

The context and domain decomposition
------------------------------------

:class:`arbor.single_cell_model` does not only take care of the recipe, it also takes
care of defining how the simulation will be run. When you create and use your own
recipe, you'll need to do this manually, in the form of defining a execution context
and a domain decomposition. Fortunately, the default constructors of
:class:`arbor.context` and :class:`arbor.partition_load_balance` are sufficient for
this model, and is what :class:`arbor.single_cell_model` does under the hood! We'll
leave the details of this subject for another tutorial.

.. code-block:: python

    # (6) Create a default execution context and a default domain decomposition.

    context = arbor.context()
    domains = arbor.partition_load_balance(recipe, context)

Step **(6)** sets up a default context and domains.

The simulation
--------------

.. code-block:: python

    # (7) Create and run simulation and set up 10 kHz (every 0.1 ms) sampling on the probe.
    # The probe is located on cell 0, and is the 0th probe on that cell, thus has probe_id (0, 0).

    sim = arbor.simulation(recipe, domains, context)
    sim.record(arbor.spike_recording.all)
    handle = sim.sample((0, 0), arbor.regular_schedule(0.1))
    sim.run(tfinal=30)

Step **(7)** instantiates the simulation and sets up the probe added in step 5. In the
:class:`arbor.single_cell_model` version of this example, the probe frequency and
simulation duration are the same. Note that the frequency is set with a :class:`arbor.regular_schedule`,
which takes a time and not a frequency. Also note that spike recording must be
switched on. For extraction of the probe traces later on, we store a handle.

The results
----------------------------------------------------

Apart from creating :class:`arbor.recipe` ourselves, we have changed nothing
about this simulation compared to :ref:`the previous tutorial <tutorialsinglecell>`.
If we create the same analysis of the results we therefore expect the same results.

.. code-block:: python

    # (8) Collect results.

    spikes = sim.spikes()
    data, meta = sim.samples(handle)[0]

    if len(spikes)>0:
        print('{} spikes:'.format(len(spikes)))
        for t in spikes['time']:
            print('{:3.3f}'.format(t))
    else:
        print('no spikes')

    print("Plotting results ...")
    seaborn.set_theme() # Apply some styling to the plot
    df = pandas.DataFrame({'t/ms': data[:, 0], 'U/mV': data[:, 1]})
    seaborn.relplot(data=df, kind="line", x="t/ms", y="U/mV", ci=None).savefig('single_cell_recipe_result.svg')

    df.to_csv('single_cell_recipe_result.csv', float_format='%g')

Step **(8)** plots the measured potentials during the runtime of the simulation.
Retrieving the sampled quantities is a little different, these have to be accessed
through the simulation object: :meth:`arbor.simulation.spikes` and :meth:`arbor.simulation.samples`.

We should be seeing something like this:

.. figure:: single_cell_model_result.svg
    :width: 400
    :align: center

    Plot of the potential over time for the voltage probe added in step (5).

You can find the source code for this example in full at ``python/examples/single_cell_recipe.py``.
