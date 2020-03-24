.. _labels:

Labels
=========

Arbor provides a domain specific language (DSL) for labeling regions and
locations on morphologies.
Properties and attributes can be assigned to a labeled region or set of locations,
for example, the membrane capacitance or specify an ion channel on a region of the cell membrane,
or place synapse instances on a set of sites on the cell.

Label Types
------------

Locations
~~~~~~~~~~~

A *location* is a single point on a morphology, used to place countable entities
on the morphology. Examples of countable entities include synapses, gap junction
sites, voltage recorders and current clamps.
Labels are given to groups of locations, called *locsets*.
The following are examples of locsets:

* The center of the soma.
* 100 random locations on the dendrites.
* The terminal points of a cell.
* All points in the dendritic tree that are 100 μm from the soma.

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

Example Cell
------------

To demonstrate how to label regions and locsets, we will use the following example morphology.

.. figure:: gen-images/morphlab.svg
  :width: 800
  :align: center

  Example morphology, composed of a soma, dendritic tree and axon.

.. figure:: gen-images/labels.svg
  :width: 800
  :align: center

  **Left**: an example illustration of a locset.
  **Right**: an example illustration of a region.

* Branch 0 is the soma, modelled as a cylinder of length and diameter 4 μm.
* The dendritic tree is decsribed by branches 1,2,3,4 and 5.

    * Description of radius in dendrites.

* Branch 6 is the axon, starting as a hillock attached to the soma, and an initial segment of axon.

Concrete
--------------

A location on a cell is uniquely described using a tuple ``(branch, pos)`` where ``branch`` is a
branch id, and ``0 ≤ pos ≤ 1`` is a relative distance along the branch, where 0 is the
proximal end of the branch, and 1 is the distal end of the branch.

Regions are composed of unbranched *cables*, which are tuples of the form ``(branch, prox, dist)``,
where ``branch`` is the branch id, and ``0 ≤ prox ≤ dist ≤ 1`` define the relative position
of the end points of the section on the branch.

*TODO* some examples of cables and locations. (whole branch, subset of branch, root (0,0), mid point of a dendrite)

*TODO* introduce *locset* and *cable_list*


