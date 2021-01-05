.. _tutorialsimplecell:

A simple single cell model
==========================

Building and testing detailed models of individual cells, then optimizing their
parameters is usually the first step in building models with multi-compartment cells.
Arbor supports a *single cell model* workflow for this purpose, which is a good way to
introduce Arbor's cell modelling concepts and approach.

.. Note::

   **Concepts covered in this example:**

   1. Intro to building a morphology from a :class:`arbor.segment_tree`.
   2. Intro to region and locset expressions.
   3. Intro to decors and cell decorations.
   4. Building a :class:`arbor.cable_cell` object.
   5. Building a :class:`arbor.single_cell_model` object.
   6. Running a simulation and visualising the results.

.. _tutorialsinglecell-cell:

The cell
--------

The most trivial representation of a cell in Arbor is to model the entire cell as a
single cylinder. The following example shows the steps required to construct a model of a
cylindrical cell with a length of 6 μm and a radius of 3 μm; Hodgkin–Huxley dynamics
and a current clamp stimulus, then run the model for 30 ms.

The first step is to construct the cell. In Arbor, the abstract representation used to
define a cell with branching cable morphology is a ``cable_cell``, which holds a
description of the cell's morphology, named regions and locations on the morphology, and
descriptions of ion channels, synapses, spike detectors and electrical properties.

Our *single-segment HH cell* has a simple morphology and dynamics, constructed as follows:

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

    # (4) Create cell
    cell = arbor.cable_cell(tree, labels, decor)

The recipe
----------

Once the cell description has been built, the next step is to build and run the simulation.
Arbor provides an interface for constructing single cell models with the
:class:`arbor.single_cell_model` helper that creates a model from a cell description, with
an interface for recording outputs and running the simulation.

.. code-block:: python

    class single_recipe (arbor.recipe):
        def __init__(self, cell, probes):
            # The base C++ class constructor must be called first, to ensure that
            # all memory in the C++ class is initialized correctly.
            arbor.recipe.__init__(self)
            self.the_cell = cell
            self.the_probes = probes
            self.the_props = arbor.neuron_cable_propetries()
            self.the_cat = arbor.default_catalogue()
            self.the_props.register(self.the_cat)

        def num_cells(self):
            return 1

        def num_sources(self, gid):
            return 1

        def cell_kind(self, gid):
            return arbor.cell_kind.cable

        def cell_description(self, gid):
            return self.the_cell

        def probes(self, gid):
            return self.the_probes

        def global_properties(self, kind):
            return self.the_props

The results
-----------

Our cell and model have been defined and we have run our simulation. Now we can look at what
the spike detector and a voltage probes from our model have produced.

.. code-block:: python

    # (8) Print spike times.
    if len(m.spikes)>0:
        print('{} spikes:'.format(len(m.spikes)))
        for s in m.spikes:
            print('{:3.3f}'.format(s))
    else:
        print('no spikes')

    # (9) Plot the recorded voltages over time.
    import pandas, seaborn # You may have to pip install these.
    seaborn.set_theme() # Apply some styling to the plot
    df = pandas.DataFrame({'t/ms': m.traces[0].time, 'U/mV': m.traces[0].value})
    seaborn.relplot(data=df, kind="line", x="t/ms", y="U/mV",ci=None).savefig('single_cell_model_result.svg')

Step **(8)** accesses :meth:`arbor.single_cell_model.spikes`
to print the spike times. A single spike should be generated at around the same time the stimulus
we provided in step (3) gets activated (10ms).

Step **(9)** plots the measured potentials during the runtime of the simulation. The sampled quantities
can be accessed through :meth:`arbor.single_cell_model.traces`.
We should be seeing something like this:

.. figure:: single_cell_model_result.svg
    :width: 400
    :align: center

    Plot of the potential over time for the voltage probe added in step (6).

The full code
-------------

You can find the source code for this example in full at ``python/examples/single_cell_model.py``.
