.. _labels:

Labels
=========

Arbor provides a domain specific language (DSL) for labeling regions and
locations on morphologies.
Labels are used later to describe cell properties and attributes,
for example, the membrane capacitance on a region of the cell membrane,
or the location of synapse instances.

Example Cell
------------

The following morphology will be used to demonstrate the DSL.

.. _labels-morph-fig:

.. figure:: gen-images/morphlab.svg
  :width: 800
  :align: center

  **Left**: Segments of the sample morphology, colored according to tags: soma (tag 1, red), axon (tag 2, gray), dendrites (tag 3, blue). **Right**: The Soma is branch 0; the dendritic tree is composed of branches 1, 3, 3, 4 and 5; and the axon is branch 6.

Branch 0 contains the soma, which is modelled as a cylinder of length and diameter 4 μm, and the first branch of the dendritic tree which has a radius of 0.75 μm, and is attached to the distal end of the soma.
The other branches in the dendritic tree have the following properties: branch 1 tapers from 0.4 to 0.2 μm; branch 2 has radius 0.5 μm;
and branches 3 and 4 taper from 0.5 to 0.2 μm.

The axon is a single branch, composed of two cable segments: a tapering axon hillock attached to the proximal end of the soma, and the start of the axon proper with radius 0.4 μm.


Label Types
------------

Locations
~~~~~~~~~~~

A *location* is used to place countable entities
on the morphology. Examples of countable entities include synapses, gap junction
sites, voltage recorders and current clamps.
Labels are given to groups of locations, called *locsets*.
The following are examples of locsets:

* The center of the soma.
* 100 random locations on the dendrites.
* The terminal samples of a cell.
* All points in the dendritic tree that are 100 μm from the soma.

.. figure:: gen-images/locset_label_examples.svg
  :width: 800
  :align: center

  Locsets on the example morphology. **Left**: terminal samples.
  **Right**: 10 random locations on the dendritic tree.


Regions
~~~~~~~~~~~~

A *region* is a subset of a morphology, that can refer to membrane or volume of
morphology cables.
Regions are used to define membrane properties, for example the type and density
of specific ion channels, membrane capacitance capacitance, or initial reversal
potential.
Examples of regions include:

* The soma.
* The dendrites.
* An explicit reference to a specific unbranched cable, e.g. "branch 3" or "the distal half of branch 1".
* The axon hillock.
* The dendrites with radius less than 1 μm.

It is possible for a region to be empty, for example a region that defines the axon
hillock would be empty on a morphology that has no axon.
Regions do not need to be complete sub-trees of a morphology.

Examples:

* a branch
* a sub-tree
* disjoint regions

.. figure:: gen-images/region_label_examples.svg
  :width: 800
  :align: center

  Regions on the example morphology. **Left**: the dendritic tree.
  **Right**: All cables with radius less than 0.5 μm.

Representations
----------------

A *location* on a cell is uniquely described using a tuple ``(branch, pos)`` where ``branch`` is a
branch id, and ``0 ≤ pos ≤ 1`` is a relative distance along the branch, where 0 and 1 are the
proximal and distal ends of the branch respectively.

Regions are composed of unbranched *cables*, which are tuples of the form ``(branch, prox, dist)``,
where ``branch`` is the branch id, and ``0 ≤ prox ≤ dist ≤ 1`` define the relative position
of the end points of the section on the branch.

*TODO* some examples of cables and locations. (whole branch, subset of branch, root (0,0), mid point of a dendrite)

*TODO* introduce *locset* and *cable_list*

Regions
-------


