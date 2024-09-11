.. _tutorialsinglecell:

A simple single-cell model
==========================

Building and testing detailed models of individual cells, then optimizing their
parameters is usually the first step in building models with multi-compartment cells.
Arbor supports a *single-cell model* workflow for this purpose, which is a good way to
introduce Arbor's cell modelling concepts and approach.

.. Note::

   **Concepts covered in this example:**

   0. The Arbor library and units.
   1. Intro to morphologies.
   2. Intro to region and locset expressions.
   3. Intro to decors and cell decorations.
   4. Building a :class:`arbor.cable_cell` object.
   5. Building a :class:`arbor.single_cell_model` object.
   6. Running a simulation and visualising the results.

.. _tutorialsinglecell-cell:

Setup and introduction to units
-------------------------------

We begin by importing the Arbor library and its unit support.

.. literalinclude:: ../../python/example/single_cell_model.py
   :language: python
   :lines: 4-5

As we will refer to both quite often, we assign aliases ``A`` and ``U``, to
minimize typing. Over the course of this introduction, you will notice that most
of Arbor's user interface is making use of units. This requires a bit of typing,
but makes the physical quantities obvious and allows for easy conversion of
models. You can use any sensible unit for a given dimension and Arbor will
convert as needed, e.g., you can write ``5 * U.mm`` instead of ``5000 * U.um``.
Handing a mismatching dimension to a method will cause a runtime error, so in
the example above, ``5 * U.mV`` will be rejected.

The cell
--------

The most trivial representation of a cell in Arbor is to model the entire cell
as a single cylinder. The following example shows the steps required to
construct a model of a cylindrical cell with a length of 6 μm and a radius of 3
μm; Hodgkin–Huxley dynamics and a current clamp stimulus, then run the model for
30 ms.

The first step is to construct the cell. In Arbor, the abstract representation
used to define a cell with branching cable morphology is a ``cable_cell``, which
holds a description of the cell's morphology, named regions, and locations on the
morphology, and descriptions of ion channels, synapses, threshold detectors, and
electrical properties. We will go over these one by one.

Our *cell* has a simple morphology comprising a single segment, which is why we
use an explicit construction. Normally, one would read the morphology from file
and Arbor handles most standard formats natively.

.. literalinclude:: ../../python/example/single_cell_model.py
   :language: python
   :lines: 9-11

This constructs a :class:`arbor.segment_tree` (see also :ref:`segment
tree<morph-segment_tree>`) containing a single segment. You can skip the rest of
this paragraph on first reading, it explains the details of constructing a
morphology from scratch. The segment tree is the representation used to
construct the morphology of a cell. A segment is a tapered cone with a tag; the
tag can be used to classify the type of the segment (for example soma, dendrite
, etc). To create a segment tree representing our single-cylinder cell, we need to
add one segment to our ``tree`` object. We use the
:meth:`arbor.segment_tree.append` method, which takes 4 arguments: the parent
segment which does not exist for the first segment, so we use
:class:`arbor.mnpos`; the proximal :class:`arbor.mpoint` (location and radius)
of the segment; the distal :class:`arbor.mpoint` of the segment; and the tag.

.. literalinclude:: ../../python/example/single_cell_model.py
   :language: python
   :lines: 13-14

Next, we create a dictionary of labels
(:class:`arbor.label_dict<arbor.label_dict>`) to assign properties to. This is a
handy tool to connect part of your morphology to semantically meaningful names.
Labels give names to :term:`regions<region>` and :term:`location<locset>`
described using a DSL based on s-expressions. Labels from the dictionary can
then be used to facilitate adding synapses, dynamics, stimuli, and probes to the
cell. We add two labels:

* ``soma`` defines a *region* with ``(tag 1)``. Note that this corresponds to
  the ``tag`` parameter that was used to define the single segment in step (1).
* ``midpoint`` defines a *location* at ``(location 0 0.5)``, which is the midpoint
``0.5`` of branch ``0``, which corresponds to the midpoint of the soma
  on the morphology defined in step (1).

.. literalinclude:: ../../python/example/single_cell_model.py
   :language: python
   :lines: 16-23

The final piece constructs a :class:`arbor.decor` describing the distribution
and placement of dynamics and properties on a cell. The cell's default
properties can be modified, and we can use :meth:`arbor.decor.paint` and
:meth:`arbor.decor.place` to further customise it in the following way:

* :meth:`arbor.decor.set_property` is used to set some default properties on the
  entire cell. In the above example we set the initial membrane potential
  to -40 mV.
* :meth:`arbor.decor.paint` is used to set properties or add dynamics to a
  region of the cell. We call this method 'painting' to convey that we are
  working on sections of a cell, as opposed to precise locations: for example,
  we might want to *paint* a density ion channel on all dendrites, and then
  *place* a synapse at the tip of the axon. In the above example we paint HH
  dynamics on the region we previously named ``"soma"`` in our label dictionary.
* :meth:`arbor.decor.place` is used to add objects on a precise
  :class:`arbor.location` on a cell. Examples of objects that are *placed* are synapses,
  threshold detectors, current stimuli, and probes. In the above example, we place a current stimulus
  :class:`arbor.iclamp` with a duration of 2 ms and a current of 0.8 nA, starting at 10 ms
  on the location we previously labelled ``"midpoint"``. We also place a :class:`arbor.threshold_detector`
  with a threshold of -10 mV on the same location.

.. literalinclude:: ../../python/example/single_cell_model.py
   :language: python
   :lines: 25-26

The three ingredients -- morphology, labels, and decor -- are joined into a cable cell.

The single-cell model
---------------------

Once the cell description has been built, the next step is to build and run the
simulation. Arbor provides an interface for constructing single-cell models with
the :class:`arbor.single_cell_model` helper that creates a model from a cell
description, with an interface for recording outputs and running the simulation.

.. literalinclude:: ../../python/example/single_cell_model.py
   :language: python
   :lines: 28-29

The single-cell model has 4 main functions:

1. It holds the **global properties** of the model
2. It registers **probes** on specific locations on the cell to measure the voltage.
3. It **runs** the simulation.
4. It collects **spikes** from threshold detectors and voltage **traces** from registered probes.

Right now, we'll only set a probe. The model is complete without, but to see
the results, we need to extract some data.

.. literalinclude:: ../../python/example/single_cell_model.py
   :language: python
   :lines: 31-32

Note, that the probe is given a location from the label dictionary ``midpoint``,
the value to record ``voltage``, the sampling frequency, and finally a tag by
which we can reference later here ``Um``.

Now, we can start the actual simulation:

.. literalinclude:: ../../python/example/single_cell_model.py
   :language: python
   :lines: 34-35

The results
-----------

Our cell and model have been defined and we have run our simulation. Now we can
look at the resulting spikes and membrane potential.

.. literalinclude:: ../../python/example/single_cell_model.py
   :language: python
   :lines: 37-40

To print the spike times, we use :meth:`arbor.single_cell_model.spikes`. A
single spike should be generated at around the same time the stimulus we
provided in step (3) gets activated (10ms).

And, finally, we plot the membrane potential

.. literalinclude:: ../../python/example/single_cell_model.py
   :language: python
   :lines: 6-7,42-48


We should be seeing something like this:

.. figure:: single_cell_model_result.svg
    :width: 400
    :align: center

    A plot of the potential over time for the voltage probe was added in step (6).

The full code
-------------

You can find the source code for this example in full at
``python/examples/single_cell_model.py`` which comes in at roughly 10 lines of
Python to define and simulate a cell from scratch.
