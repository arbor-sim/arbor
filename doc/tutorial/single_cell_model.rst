.. _gs_single_cell:

A single cell model
================================

Building and testing detailed models of individual cells, then optimizing their
parameters is usually the first step in building models with multi-compartment cells.
Arbor supports a *single cell model* workflow for this purpose, which is a good way to
introduce Arbor's cell modelling concepts and approach.

This guide will walk through a series of single cell models of increasing complexity.
Links are provided to separate documentation that covers relevant topics in more detail.

In an interactive Python interpreter, you can use ``help()`` on any class or function to
obtain some documentation. (Try, for example, ``help(arbor.simulation``).

.. _single_soma:

Single segment cell with HH dynamics
----------------------------------------------------

The most trivial representation of a cell in Arbor is to model the entire cell as a
single cylinder. The following example shows the steps required to construct a model of a
cylindrical cell with a length of 6 μm and a radius of 3 μm; Hodgkin–Huxley dynamics
and a current clamp stimulus, then run the model for 30 ms.

The first step is to construct the cell. In Arbor, the abstract representation used to
define a cell with branching "cable" morphology is a ``cable_cell``, which holds a
description of the cell's morphology, named regions and locations on the morphology, and
descriptions of ion channels, synapses, spike detectors and electrical properties.

Our "single-segment HH cell" has a simple morphology and dynamics, so the steps to
create the ``cable_cell`` that represents it are as follows:

.. code-block:: python

    import arbor

    # (1) Create a morphology with a single (cylindrical) segment of length=diameter=6 μm
    tree = arbor.segment_tree()
    tree.append(arbor.mnpos, arbor.mpoint(-3, 0, 0, 3), arbor.mpoint(3, 0, 0, 3), tag=1)

    # (2) Define the soma and its center
    labels = arbor.label_dict({'soma':   '(tag 1)',
                               'center': '(location 0 0.5)'})

    # (3) Create cell and set properties
    cell = arbor.cable_cell(tree, labels)
    cell.set_properties(Vm=-40)
    cell.paint('"soma"', 'hh')
    cell.place('"center"', arbor.iclamp( 10, 2, 0.8))
    cell.place('"center"', arbor.spike_detector(-10))

Let's unpack that.

Step **(1)** constructs a :class:`arbor.segment_tree` (see also :ref:`segment tree<morph-segment_tree>`).
The segment tree is the representation used to construct the morphology of a cell. A segment is
a tapered cone with a tag; the tag can be used to classify the type of the segment (for example
soma, dendrite etc). To create a segment tree representing our single-cylinder cell, we need to add
one segment to our ``tree`` object. We use the :meth:`arbor.segment_tree.append` method, which takes
4 arguments: the parent segment which does not exist for the first segment, so we use :class:`arbor.mnpos`;
the proximal :class:`arbor.mpoint` (location and radius) of the segment; the distal :class:`arbor.mpoint`
of the segment; and the tag.

Step **(2)** creates a dictionary of labels (:class:`arbor.label_dict<arbor.label_dict>`). Labels give
names to :ref:`regions<labels-region>` and :ref:`location<labels-locset>` described using a DSL
based on s-expressions. Labels from the dictionary can then be used to facilitate adding synapses,
dynamics, stimulii and probes to the cell. We add two labels:

* ``soma`` defines a *region* with ``(tag  1)``. Note that this corresponds to the
  ``tag`` parameter that was used to define the single segment in step (1).
* ``center`` defines a *location* at ``(location 0 0.5)``, which is the mid point ``0.5``
  of branch ``0``, which corresponds to the center of the soma on the morphology defined in step (1).

Step **(3)** constructs the :class:`arbor.cable_cell` from the segment tree and dictionary of labeled
regions and locations. The resulting cell's default properties can be modified, and we can use
:meth:`arbor.cable_cell.paint` and :meth:`arbor.cable_cell.place` to further customise it in the
following way:

* :meth:`arbor.cable_cell.set_properties` is used to set some default properties on the entire cell.
  In the above example we set the initial membrane potential to -40 mV.
* :meth:`arbor.cable_cell.paint` is used to set properties or add dynamics to a region of the cell.
  We call this method 'painting' to convey that we are working on sections of a cell, as opposed to
  precise locations: for example, we might want to ``paint`` an ion channel on all dendrites, and then
  ``place`` a synapse at the tip of the axon. In the above example we :meth:`arbor.cable_cell.paint`
  HH dynamics on the region we previously named 'soma' in our label dictionary.
* :meth:`arbor.cable_cell.place<arbor.cable_cell.place>` is used to add objects on a precise
  :class:`arbor.location` on a cell. Examples of objects that are ``placed`` are synapses,
  spike detectors, current stimulii, and probes. In the above example we place a current stimulus
  :class:`arbor.iclamp<arbor.iclamp>` with a duration of 2 ms and a current of 0.8 nA, starting at 10 ms
  on the location we previously labelled 'center'. We also place a :class:`arbor.spike_detector<arbor.spike_detector>`
  with a threshold of -10 mV on the same location.

Single cell model
----------------------------------------------------

Great, we have defined our cell! Now, let's move on to the simulation. Arbor is able to simulate
networks with multiple individual cells; this requires a *recipe* to describe the cells,
connections, gap junctions, etc. However, for single cell models, arbor does not require the recipe
to be provided by the user. Arbor provides a :class:`arbor.single_cell_model`
helper that wraps a cell description and creates a recipe under the hood, providing an interface for
recording potentials and running the simulation more easily.

.. code-block:: python

    # (4) Make single cell model.
    m = arbor.single_cell_model(cell)

    # (5) Attach voltage probe sampling at 10 kHz (every 0.1 ms).
    m.probe('voltage', '"center"', frequency=10000)

    # (6) Run simulation for 30 ms of simulated activity.
    m.run(tfinal=30)

Step **(4)** instantiates the :class:`arbor.single_cell_model`
with our single-compartment cell.

Step **(5)** adds a :meth:`arbor.single_cell_model.probe`
used to record variables from the model. Three pieces of information are
provided: the type of quantity we want probed (voltage), the location where we want to
probe ('"center"'), and the frequency at which we want to sample (10kHz).

Step **(6)** runs the actual simulation for a duration of 30 ms.

Results
----------------------------------------------------

Our cell and model have been defined and we have run our simulation. However, we have not seen any
results! Let's take a look at what the spike detector and a voltage probes from our model have produced.

.. code-block:: python

    # (7) Print spike times, if any.
    if len(m.spikes)>0:
        print('{} spikes:'.format(len(m.spikes)))
        for s in m.spikes:
            print('{:3.3f}'.format(s))
    else:
        print('no spikes')

    # (8) Plot the recorded voltages over time.
    import pandas, seaborn # You may have to pip install these.
    seaborn.set_theme() # Apply some styling to the plot
    df = pandas.DataFrame({'t/ms': m.traces[0].time, 'U/mV': m.traces[0].value})
    seaborn.relplot(data=df, kind="line", x="t/ms", y="U/mV",ci=None).savefig('single_cell_model_result.svg')

Step **(7)** accesses :meth:`arbor.single_cell_model.spikes`
to print the spike times. A single spike should be generated at around the same time the stimulus
we provided in step (3) gets activated (10ms).

Step **(8)** plots the measured potentials during the runtime of the simulation. The sampled quantities
can be accessed through :meth:`arbor.single_cell_model.traces`.
We should be seeing something like this:

.. figure:: single_cell_model_result.svg
    :width: 400
    :align: center

    Plot of the potential over time for the voltage probe added in step (5).

You can find the source code for this example in full at ``python/examples/single_cell_model.py``.

.. Todo::
    Add equivalent but more comprehensive recipe implementation in parallel, such that the reader learns how single_cell_model works.
