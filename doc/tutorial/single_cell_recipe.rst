.. _tutorialsimplecellrecipe:

A simple single cell recipe
===========================

This example builds the same single cell model as
:ref:`the previous tutorial <tutorialsimplecell>`, except using a :class:`arbor.recipe`
and :class:`arbor.simulation` instead of a :class:`arbor.single_cell_model`.

.. Note::

   **Concepts covered in this example:**

   1. Building a :class:`arbor.recipe`.
   2. Using the recipe, context and domain decomposition to create a :class:`arbor.simulation`
   3. Running the simulation and visualizing the results.

The cell
--------

We can immediately paste the cell description code from the
:ref:`previous example <tutorialsinglecell-cell` where it is explained in detail.

.. code-block:: python

    import arbor

    # (1) Create a morphology with a single (cylindrical) segment of length=diameter=6 μm
    tree = arbor.segment_tree()
    tree.append(arbor.mnpos, arbor.mpoint(-3, 0, 0, 3), arbor.mpoint(3, 0, 0, 3), tag=1)

    # (2) Define the soma and its center
    labels = arbor.label_dict({'soma':   '(tag 1)',
                              'center': '(location 0 0.5)'})

    # (3) Create cell and set properties
    decor = arbor.decor()
    decor.set_property(Vm=-40)
    decor.paint('"soma"', 'hh')
    decor.place('"center"', arbor.iclamp( 10, 2, 0.8))
    decor.place('"center"', arbor.spike_detector(-10))

    # (4) Create cell and the single cell model based on it
    cell = arbor.cable_cell(tree, labels, decor)

The context and domain decomposition
------------------------------------

When you create and use your own recipe, you'll also need to define a context :ref:`in this tutorial<tutorialsinglecellrecipe>:`.

The simulation
--------------

The results
----------------------------------------------------

Apart from creating :class:`arbor.recipe` ourselves, we have changed nothing
about this simulation compared to :ref:`the previous tutorial <tutorialsimplecell>`.
If we create the same analysis of the results we therefore expect the same results.

.. code-block:: python

    # (9) Collect results.

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

Step **(9)** plots the measured potentials during the runtime of the simulation.
Retrieving the sampled quantities is a little different, these have to be accessed through the simulation object: meth:`arbor.simulation.spikes` and meth:`arbor.simulation.handle`.

We should be seeing something like this:

.. figure:: single_cell_model_result.svg
    :width: 400
    :align: center

    Plot of the potential over time for the voltage probe added in step (6).

You can find the source code for this example in full at ``python/examples/single_cell_model.py``.
